import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  async rewrites() {
    // In local dev without NEXT_PUBLIC_API_URL, proxy /api/* to FastAPI on :8000
    if (process.env.NODE_ENV !== 'production' && !process.env.NEXT_PUBLIC_API_URL) {
      return [
        {
          source: '/api/backend/:path*',
          destination: 'http://localhost:8000/:path*',
        },
      ];
    }
    return [];
  },

  env: {
    NEXT_PUBLIC_APP_NAME: "AI Resume Screener",
    NEXT_PUBLIC_APP_VERSION: "2.0.0",
    // Set NEXT_PUBLIC_API_URL in Vercel dashboard to point to Render backend
    // e.g. https://ai-resume-screener-api.onrender.com
  },
};

export default nextConfig;
