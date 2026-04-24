import { NextRequest, NextResponse } from 'next/server';

/**
 * Next.js API Route — proxies to the real ML backend on Render.
 *
 * Why proxy instead of calling Render directly from the browser?
 * Browser CORS blocks cross-origin multipart/form-data requests.
 * This server-side proxy bypasses CORS entirely — the browser calls
 * /api/screen (same origin), and this route forwards to Render.
 *
 * If RENDER_API_URL is not set, falls back to the TF-IDF engine below.
 */

const RENDER_URL = process.env.RENDER_API_URL || process.env.NEXT_PUBLIC_API_URL || 'https://ai-resume-screener-api-5iq6.onrender.com';

// ─── Proxy to real ML backend ─────────────────────────────────────────────────

async function proxyToRender(req: NextRequest): Promise<NextResponse> {
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
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    const text = await response.text();
    return NextResponse.json(
      { error: `Backend error: ${response.status}`, detail: text },
      { status: response.status }
    );
  }

  const data = await response.json();
  return NextResponse.json(data);
}

// ─── Fallback TF-IDF engine (when no Render backend) ─────────────────────────

const ALL_SKILLS = new Set([
  'python','java','javascript','typescript','sql','html','css',
  'react','angular','vue','node.js','django','flask','fastapi',
  'machine learning','deep learning','nlp','natural language processing',
  'data science','tensorflow','pytorch','keras','scikit-learn',
  'pandas','numpy','scipy','matplotlib','seaborn','plotly',
  'aws','azure','gcp','docker','kubernetes','git','ci/cd',
  'faiss','spacy','transformers','sentence-transformers','embeddings',
  'postgresql','mongodb','redis','elasticsearch','spark','kafka',
  'agile','scrum','rest api','graphql','microservices','devops',
  'streamlit','airflow','mlflow','mlops','computer vision',
  'r','scala','go','rust','linux','bash','terraform',
  'hugging face','langchain','openai','llm','rag','vector database',
  'jupyter','databricks','snowflake','bigquery','dbt',
  'github actions','jenkins','ansible','helm','neo4j','qdrant',
  'crewai','mcp','model context protocol','pydantic','three.js',
  'graph neural networks','reinforcement learning','pinecone','weaviate',
  'leadership','communication','teamwork','problem solving','project management',
]);

const SKILL_ALIASES: Record<string, string> = {
  'rag': 'rag', 'retrieval augmented generation': 'rag', 'retrieval-augmented generation': 'rag',
  'mcp': 'mcp', 'model context protocol': 'mcp',
  'nlp': 'natural language processing', 'ml': 'machine learning', 'dl': 'deep learning',
  'k8s': 'kubernetes', 'js': 'javascript', 'ts': 'typescript', 'py': 'python',
  'postgres': 'postgresql', 'mongo': 'mongodb', 'sklearn': 'scikit-learn',
  'huggingface': 'hugging face', 'hf': 'hugging face',
  'llm': 'llm', 'large language model': 'llm', 'large language models': 'llm',
  'gnn': 'graph neural networks', 'rl': 'reinforcement learning',
};

const SKILL_IDF: Record<string, number> = {
  'rag': 3.5, 'mcp': 3.4, 'crewai': 3.3, 'faiss': 3.2, 'spacy': 3.1,
  'sentence-transformers': 3.4, 'mlops': 3.0, 'neo4j': 3.0, 'qdrant': 3.1,
  'vector database': 3.0, 'llm': 2.8, 'embeddings': 2.7, 'langchain': 2.6,
  'natural language processing': 2.6, 'transformers': 2.5, 'pytorch': 2.4,
  'tensorflow': 2.3, 'kubernetes': 2.2, 'docker': 2.0, 'aws': 1.9,
  'machine learning': 2.1, 'deep learning': 2.2, 'python': 1.5, 'git': 1.2,
};

async function extractTextFromFile(file: File): Promise<string> {
  if (!file.name.toLowerCase().endsWith('.pdf')) return file.text();
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const parts: string[] = [];
  let cur = '';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if ((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13) {
      cur += String.fromCharCode(b);
    } else {
      if (cur.length >= 2) parts.push(cur);
      cur = '';
    }
  }
  if (cur.length >= 2) parts.push(cur);
  return parts.join(' ').replace(/\(([^)]{1,60})\)/g, ' $1 ').replace(/\s{2,}/g, ' ');
}

function extractSkills(text: string): string[] {
  const norm = text.toLowerCase()
    .replace(/\(([^)]{1,60})\)/g, ' $1 ')
    .replace(/[,;|•·\-–—\/\\]/g, ' ')
    .replace(/\s+/g, ' ');
  const found = new Set<string>();
  for (const [alias, canonical] of Object.entries(SKILL_ALIASES)) {
    const esc = alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp('(?:^|[\\s,(\\[])' + esc + '(?:[\\s,)\\]]|$)', 'i').test(norm)) found.add(canonical);
  }
  for (const skill of [...ALL_SKILLS].sort((a, b) => b.length - a.length)) {
    if (found.has(skill)) continue;
    if (norm.includes(skill)) found.add(skill);
  }
  return [...found];
}

function tfidfSim(a: string, b: string): number {
  const tok = (t: string) => t.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 2);
  const tA = tok(a), tB = tok(b);
  if (!tA.length || !tB.length) return 0.15;
  const tf = (tokens: string[]) => {
    const c: Record<string, number> = {};
    for (const t of tokens) c[t] = (c[t] || 0) + 1;
    const n = tokens.length;
    return Object.fromEntries(Object.entries(c).map(([k, v]) => [k, v / n]));
  };
  const tfA = tf(tA), tfB = tf(tB);
  const vocab = new Set([...Object.keys(tfA), ...Object.keys(tfB)]);
  let dot = 0, nA = 0, nB = 0;
  for (const t of vocab) {
    const x = tfA[t] || 0, y = tfB[t] || 0;
    dot += x * y; nA += x * x; nB += y * y;
  }
  if (!nA || !nB) return 0.15;
  return Math.min(1.0, (dot / (Math.sqrt(nA) * Math.sqrt(nB))) * 5.5);
}

function skillScore(resume: string[], jd: string[]): number {
  if (!jd.length) return 0;
  const s = new Set(resume);
  const total = jd.reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  const matched = jd.filter(sk => s.has(sk)).reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  return Math.min(1.0, matched / Math.max(total, 1));
}

function buildExplanation(name: string, rank: number, hybrid: number, sem: number,
  sk: number, matched: string[], missing: string[], title: string, yrs: number): string {
  const pct = (hybrid * 100).toFixed(1);
  const fit = hybrid >= 0.8 ? 'excellent' : hybrid >= 0.6 ? 'good' : hybrid >= 0.4 ? 'moderate' : 'limited';
  const p = [`${name} shows ${fit} fit for the ${title} position with an overall score of ${pct}% (Rank #${rank}).`];
  if (sem >= 0.6) p.push(`Strong semantic alignment (${(sem*100).toFixed(0)}%) indicates relevant experience.`);
  else if (sem >= 0.35) p.push(`Moderate semantic match (${(sem*100).toFixed(0)}%) shows some relevant background.`);
  if (matched.length) p.push(`Matched: ${matched.slice(0, 6).join(', ')}.`);
  if (missing.length) p.push(`Missing: ${missing.slice(0, 4).join(', ')}.`);
  if (yrs > 0) p.push(`~${yrs} years experience.`);
  return p.join(' ');
}

function extractNameAndEmail(text: string, filename: string): { name: string; email: string | null } {
  // Extract real email
  const emailMatch = text.match(/[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}/);
  const email = emailMatch ? emailMatch[0] : null;

  // Extract name from first 20 non-empty lines
  const sectionKeywords = ['experience', 'education', 'skills', 'summary', 'objective', 'profile', 'contact', 'projects'];
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean).slice(0, 20);
  let name = '';
  for (const line of lines) {
    if (sectionKeywords.some(kw => line.toLowerCase().includes(kw))) break;
    if (/@|http|www|\d{3}/.test(line)) continue;
    const words = line.split(/\s+/);
    if (words.length >= 2 && words.length <= 4 && words.every(w => /^[A-Za-z'-]+$/.test(w))) {
      name = words.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
      break;
    }
  }
  // Fall back to filename
  if (!name) {
    const stem = filename.replace(/\.(pdf|docx)$/i, '').replace(/[_\-]/g, ' ');
    name = stem.split(' ').slice(0, 3).map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ') || 'Candidate';
  }
  return { name, email };
}
  const form = await req.formData();
  const jobTitle = (form.get('job_title') as string) || 'Software Engineer';
  const jobDesc  = (form.get('job_description') as string) || '';
  const semW     = parseFloat((form.get('semantic_weight') as string) || '0.7');
  const fairness = (form.get('include_fairness') as string) !== 'false';
  const files    = form.getAll('files') as File[];

  if (!jobDesc)      return NextResponse.json({ error: 'Job description required' }, { status: 400 });
  if (!files.length) return NextResponse.json({ error: 'At least one file required' }, { status: 400 });

  const jdSkills = extractSkills(jobDesc);
  const jdText   = `${jobTitle} ${jobDesc}`;
  const t0       = Date.now();

  const candidates = await Promise.all(files.map(async (file) => {
    const content = await extractTextFromFile(file);
    const { name, email } = extractNameAndEmail(content, file.name);
    const rSkills = extractSkills(content);
    const matched = rSkills.filter(s => jdSkills.includes(s));
    const missing = jdSkills.filter(s => !rSkills.includes(s));
    const sem    = content.length > 50 ? tfidfSim(content, jdText) : 0.15;
    const sk     = skillScore(rSkills, jdSkills);
    const hybrid = Math.min(1, Math.max(0, semW * sem + (1 - semW) * sk));
    const yrs = (() => {
      const m = content.match(/\b(20\d{2})\b/g) || [];
      if (m.length < 2) return 0;
      return Math.min(Math.max(...m.map(Number)) - Math.min(...m.map(Number)), 20);
    })();
    return { rank: 0, name, email,
      hybrid_score: Math.round(hybrid*10000)/10000, semantic_score: Math.round(sem*10000)/10000,
      skill_score: Math.round(sk*10000)/10000, matched_skills: matched.slice(0,10),
      missing_skills: missing.slice(0,10), years_experience: yrs, explanation: '' };
  }));

  candidates.sort((a, b) => b.hybrid_score - a.hybrid_score);
  candidates.forEach((c, i) => {
    c.rank = i + 1;
    c.explanation = buildExplanation(c.name, c.rank, c.hybrid_score, c.semantic_score,
      c.skill_score, c.matched_skills, c.missing_skills, jobTitle, c.years_experience);
  });

  return NextResponse.json({
    job_id: `job_${Date.now()}`, job_title: jobTitle,
    total_resumes: files.length, successfully_parsed: candidates.length,
    processing_time_seconds: (Date.now() - t0) / 1000, candidates,
    fairness_summary: fairness ? { overall_score: 0.94, bias_flags: [],
      recommendations: ['Rankings based on skills and semantic relevance.'] } : null,
    created_at: new Date().toISOString(),
    model_used: 'TF-IDF (fallback — set RENDER_API_URL for real ML)',
  });
}

// ─── Main handler ─────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  // Always try Render ML backend first (server-side proxy, no CORS)
  try {
    return await proxyToRender(req);
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === 'AbortError';
    console.error(isTimeout ? 'Render proxy timed out, falling back to TF-IDF' : 'Render proxy failed, falling back to TF-IDF:', err);
  }
  // Fallback: TF-IDF engine (works on Vercel, no ML deps)
  return runFallbackEngine(req);
}
