import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 60; // Max allowed for Vercel Hobby plan

/**
 * Next.js API Route — proxies to the real ML backend on Render.
 *
 * Why proxy instead of calling Render directly from the browser?
 * Browser CORS blocks cross-origin multipart/form-data requests.
 * This server-side proxy bypasses CORS entirely — the browser calls
 * /api/screen (same origin), and this route forwards to Render.
 */

const RENDER_URL = process.env.RENDER_API_URL || process.env.NEXT_PUBLIC_API_URL || 'https://ai-resume-screener-api-5iq6.onrender.com';

// ─── Proxy to real ML backend ─────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const targetUrl = `${RENDER_URL}/screen`;

  // 4-minute timeout — covers Render cold start (60s) + ML inference
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 240000);

  let response: Response;
  try {
    response = await fetch(targetUrl, {
      method: 'POST',
      body: form,
      signal: controller.signal,
      // No Content-Type header — let fetch set it with the boundary
    });
  } catch (err) {
    clearTimeout(timeout);
    const isTimeout = err instanceof Error && err.name === 'AbortError';
    console.error('Render proxy failed:', err);
    return NextResponse.json(
      { error: isTimeout ? 'ML Backend timed out (took > 4 mins)' : 'Failed to connect to ML Backend' },
      { status: 504 }
    );
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    const text = await response.text();
    return NextResponse.json(
      { error: `ML Backend returned ${response.status}`, detail: text },
      { status: response.status }
    );
  }

  const data = await response.json();
  return NextResponse.json(data);
}

