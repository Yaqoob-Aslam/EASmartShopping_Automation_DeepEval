/** @type {import('next').NextConfig} */
const nextConfig = {
  // Don't auto-generate AGENTS.md on `next dev`
  agentRules: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
