import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  async rewrites() {
    // In production (Vercel), /api/* is handled by the Python serverless function.
    // In local dev, proxy to the FastAPI backend running on port 8000.
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return process.env.NODE_ENV === "production"
      ? [] // Vercel handles /api/* natively via vercel.json routes
      : [
          {
            source: "/api/:path*",
            destination: `${apiBase}/:path*`,
          },
        ];
  },

  env: {
    NEXT_PUBLIC_APP_NAME: "AI Resume Screener",
    NEXT_PUBLIC_APP_VERSION: "1.0.0",
  },
};

export default nextConfig;
