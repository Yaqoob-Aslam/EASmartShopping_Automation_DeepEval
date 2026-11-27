import asyncio
import os
import sys
import time
from pathlib import Path
import json
import requests
import httpx
import pytest
from conftest import OllamaModelNoThink
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from deepeval.test_case.mcp import MCPServer, MCPToolCall
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from deepeval.metrics import MCPUseMetric, MultiTurnMCPUseMetric
from deepeval.evaluate import evaluate

_ROOT = Path(__file__).parent.parent.parent


def _resolve_python() -> str:
    """Pick the interpreter that runs the MCP server.

    Order: MCP_SERVER_PYTHON env var -> the interpreter running pytest
    (sys.executable) -> known venv locations in this repo. The old hardcoded
    `.venv/bin/python` did not exist, causing:
    FileNotFoundError: '.../EASmartShopping/.venv/bin/python'
    """
    candidates = [
        os.getenv("MCP_SERVER_PYTHON"),
        sys.executable,
        str(_ROOT / "venv/bin/python"),          # repo-level venv
        str(_ROOT / "backend/venv/bin/python"),  # backend venv
        str(_ROOT / ".venv/bin/python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "No Python interpreter found for the MCP server. Set MCP_SERVER_PYTHON "
        f"to a valid interpreter. Tried: {candidates}"
    )


_PYTHON = _resolve_python()

_SERVER = StdioServerParameters(
    command=_PYTHON,
    args=[str(_ROOT / "mcp_server.py")],
    cwd=str(_ROOT),
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)


# --- Agent model selection -------------------------------------------------
# The agent must call MCP tools, so it needs a model with the "tools"
# capability. gemma3:1b (the judge model in LOCAL_OLLAMA_MODEL) has only
# "completion", which caused:
#   ollama._types.ResponseError: .../gemma3:1b does not support tools (400)
# Override with MCP_AGENT_MODEL; otherwise a tool-capable model is picked.
_OLLAMA_BASE_URL = os.getenv("LOCAL_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_PREFERRED_TOOL_MODELS = ("llama3.2:latest", "llama3.1:latest", "mistral:latest", "qwen2.5:7b")


def _ollama_model_capabilities(model: str) -> list[str] | None:
    """Return the model's capabilities, or None if it isn't installed/reachable."""
    try:
        resp = requests.post(f"{_OLLAMA_BASE_URL}/api/show", json={"model": model}, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.json().get("capabilities") or []


def _installed_tool_models() -> list[str]:
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    return [
        m["name"]
        for m in resp.json().get("models", [])
        if "tools" in (_ollama_model_capabilities(m["name"]) or [])
    ]


def _available_models() -> list[str]:
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return [f"<server unreachable: {exc}>"]
    return [m["name"] for m in resp.json().get("models", [])]


def _resolve_agent_model() -> str:
    explicit = os.getenv("MCP_AGENT_MODEL")
    if explicit:
        caps = _ollama_model_capabilities(explicit)
        if caps is None:
            # A common trap on this machine: two Ollama installs with different
            # model stores (snap: /var/snap/ollama/common/models, manual:
            # /usr/share/ollama/.ollama/models). Pulling a model into one store
            # makes it invisible to the server that owns :11434.
            raise RuntimeError(
                f"MCP_AGENT_MODEL={explicit!r} is not available from the Ollama "
                f"server at {_OLLAMA_BASE_URL}.\n"
                f"  Server knows: {_available_models()}\n"
                f"  Pull it into the serving instance: `sudo -u ollama ollama "
                f"pull {explicit}` (or point the service at the other model store)."
            )
        if "tools" not in caps:
            raise RuntimeError(
                f"MCP_AGENT_MODEL={explicit!r} does not support tool calling "
                f"(capabilities={caps}). Pick a model such as llama3.2:latest."
            )
        return explicit

    # Prefer LOCAL_OLLAMA_MODEL only if it can actually call tools.
    judge_model = os.getenv("LOCAL_OLLAMA_MODEL")
    if judge_model and "tools" in (_ollama_model_capabilities(judge_model) or []):
        return judge_model

    for candidate in _PREFERRED_TOOL_MODELS:
        if "tools" in (_ollama_model_capabilities(candidate) or []):
            return candidate

    installed = _installed_tool_models()
    if installed:
        return installed[0]

    raise RuntimeError(
        "No Ollama model with tool-calling support found. Install one "
        "(e.g. `ollama pull llama3.2`) or set MCP_AGENT_MODEL."
    )


_AGENT_MODEL: str | None = None


def get_agent_model() -> str:
    global _AGENT_MODEL
    if _AGENT_MODEL is None:
        _AGENT_MODEL = _resolve_agent_model()
    return _AGENT_MODEL


def _patch_structured_content(result, tool_text: str) -> None:
    """deepeval template accesses result.structuredContent['result']; patch it from the text content."""
    try:
        result.structuredContent = {"result": json.loads(tool_text)}
    except (json.JSONDecodeError, Exception):
        result.structuredContent = {"result": tool_text}


# Mistral (and other small instruct models) tend to answer "here is how you
# would call the tool" instead of actually calling it, which produced prose
# with fake examples and a 0.5 MCP Use score. A strict system prompt keeps it
# on the tool-calling path.
_AGENT_SYSTEM_PROMPT = (
    "You are a shopping assistant for this store. "
    "You MUST use the provided tools to answer. "
    "Call the tools directly with arguments based on the user's request. "
    "Never print code, JSON, function signatures, or examples of tool calls. "
    "Never invent products, stores, prices, or tool results. "
    "After the tools return, answer the user using only those results. "
    "If a tool returns no products, say the store has no matching items."
)

# Ollama/anyio can drop the connection (model was unloaded, OOM-killed, or the
# server restarted). Those are transient infra failures, not test failures.
_TRANSIENT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    McpError,
    ConnectionError,
)

# Return me the tool calls and the response from the MCP Server
async def _async_run_agent(message: str) -> tuple[list[MCPToolCall], str]:
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            schemas = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
                }
                for t in tools
            ]
            llm = ChatOllama(
                model=get_agent_model(),
                base_url=_OLLAMA_BASE_URL,
                temperature=0,
                # Keep replies short: CPU-only inference is slow and the long
                # "here is a list..." summary both wastes time and inflates the
                # DeepEval judge prompt.
                num_predict=256,
            )
            messages = [
                SystemMessage(content=_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
            resp = await llm.bind_tools(schemas).ainvoke(messages)
            calls: list[MCPToolCall] = []
            tool_texts: list[str] = []
            if resp.tool_calls:
                # Append the assistant tool-call message exactly once; appending
                # it inside the loop duplicated it for multi-call responses.
                messages.append(resp)
            for tc in resp.tool_calls:
                result = await s.call_tool(tc["name"], tc["args"])
                tool_text= " ".join(c.text for c in result.content if hasattr(c, "text"))
                _patch_structured_content(result, tool_text)
                calls.append(MCPToolCall(name=tc["name"], args=tc["args"], result=result))
                tool_texts.append(tool_text)
                messages.append(ToolMessage(content=tool_text or "{}", tool_call_id=tc["id"]))

            # Second pass: LLM to summarize the tool results as natural language
            final_resp = await llm.ainvoke(messages) if resp.tool_calls else resp
            return calls, (final_resp.content or "\n".join(tool_texts))


async def _async_build_mcp_server() -> MCPServer:
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r,w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            return MCPServer(
                        server_name="ecommerce-mcp-server",
                        transport="stdio",
                        available_tools=tools
            )
        

def run_agent(message: str, attempts: int = 3) -> tuple[list[MCPToolCall], str]:
    """Run the agent, retrying transient Ollama/MCP transport failures.

    Observed on this machine: the Ollama server gets OOM-killed mid-generation
    (llama-server ~7 GB RSS alongside the desktop apps) and httpx reports
    "Server disconnected without sending a response"; a restarted MCP stdio
    process shows up as McpError("Connection closed"). Retrying makes the suite
    survive that without hiding real assertion failures.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return asyncio.run(_async_run_agent(message))
        except _TRANSIENT_ERRORS as exc:
            last_error = exc
            if attempt == attempts:
                break
            wait = min(5 * attempt, 20)
            print(f"\n[run_agent] transient failure ({type(exc).__name__}: {exc}); "
                  f"retry {attempt}/{attempts - 1} in {wait}s")
            time.sleep(wait)
    raise last_error  # type: ignore[misc]


_MCP_SERVER_CACHE: MCPServer | None = None


def get_mcp_server(attempts: int = 3) -> MCPServer:
    """Build the MCP server description once, on first use.

    Previously this ran at import time, so any startup problem aborted
    collection for the entire module. Building it lazily lets individual
    tests report the failure instead.
    """
    global _MCP_SERVER_CACHE
    if _MCP_SERVER_CACHE is not None:
        return _MCP_SERVER_CACHE
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _MCP_SERVER_CACHE = asyncio.run(_async_build_mcp_server())
            return _MCP_SERVER_CACHE
        except _TRANSIENT_ERRORS as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(5 * attempt, 20))
    raise last_error  # type: ignore[misc]


_TEST_SESSION_ID = "test-session-100"


@pytest.fixture
def mcp_judge():
    """DeepEval judge for the MCP metrics.

    Uses the same model as the agent by default. Swapping between two
    multi-GB models on CPU-only Ollama (agent llama3.2 <-> judge gemma3:1b)
    forces a reload for every call and dominated the runtime (~19 min).
    Override with MCP_JUDGE_MODEL if a dedicated judge is preferred.
    """
    return OllamaModelNoThink(model=os.getenv("MCP_JUDGE_MODEL") or get_agent_model())


class TestSearchProducts:

    def test_invoke_mcp_server(self):
        user_input = "get me all the sports shoes"
        tool_calls, result_text = run_agent(user_input)
        print(tool_calls)
        print("\n")
        print(result_text)

    def test_search_task_returns_relevant_toolcalls(self, mcp_judge):
        user_input = "find sports gears"
        tool_calls, result_text = run_agent(user_input)
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=result_text,
            mcp_servers=[get_mcp_server()],
            mcp_tools_called=tool_calls,
        )

        metrics = MCPUseMetric(threshold=0.7, model=mcp_judge)
        evaluate(test_cases=[test_case],
                 metrics=[metrics],
                 identifier="mcp:search")
        

    def test_search_task_returns_relevant_result(self, mcp_judge):
        user_input = "I need running shoes"
        tool_calls, result_text = run_agent(user_input)
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=result_text,
            mcp_servers=[get_mcp_server()],
            mcp_tools_called=tool_calls,
        )

        metrics = MCPUseMetric(threshold=0.7, model=mcp_judge)
        evaluate(test_cases=[test_case],
                 metrics=[metrics],
                 identifier="mcp:search")


class TestMCPMultiTurnConversation:
    def _build_cart_conversation(self) -> ConversationalTestCase:
        search_msg = f"Get me ONLY one highest rated running shoe with size UK 9 and color White, my session ID is {_TEST_SESSION_ID}"
        add_msg = f"Add the highest rated running shoes to the cart, my session ID is {_TEST_SESSION_ID}"
        show_msg = f"Show me the items in the cart, my session ID is {_TEST_SESSION_ID}"

        search_call, search_response = run_agent(search_msg)
        add_call, add_response = run_agent(add_msg)
        show_call, show_response = run_agent(show_msg)

        return ConversationalTestCase(
            turns=[
                Turn(role="user", content=search_msg),
                Turn(role="assistant", content="", mcp_tools_called=search_call),
                Turn(role="assistant", content=search_response),
                Turn(role="user", content=add_msg),
                Turn(role="assistant", content="", mcp_tools_called=add_call),
                Turn(role="assistant", content=add_response),
                Turn(role="user", content=show_msg),
                Turn(role="assistant", content="", mcp_tools_called=show_call),
                Turn(role="assistant", content=show_response)
            ],
            mcp_servers=[get_mcp_server()],
            chatbot_role="Shopping assistant that suggests products, items to add in the cart and show items from the cart"
        )
    
    def test_cart_multi_turn_mcp_server(self, mcp_judge):
        test_case = self._build_cart_conversation()
        metrics = MultiTurnMCPUseMetric(threshold=0.5, model=mcp_judge)
        evaluate(test_cases=[test_case], metrics=[metrics])