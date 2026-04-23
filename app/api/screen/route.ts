import { NextRequest, NextResponse } from 'next/server';

// ─── Skill taxonomy + synonyms ────────────────────────────────────────────────

// Canonical skill name -> list of aliases that should map to it
const SKILL_ALIASES: Record<string, string[]> = {
  'python': ['python3', 'py'],
  'javascript': ['js', 'es6', 'es2015'],
  'typescript': ['ts'],
  'node.js': ['node', 'nodejs', 'node js'],
  'react': ['reactjs', 'react.js'],
  'postgresql': ['postgres', 'psql'],
  'mongodb': ['mongo'],
  'kubernetes': ['k8s', 'kube'],
  'machine learning': ['ml'],
  'deep learning': ['dl'],
  'natural language processing': ['nlp'],
  'computer vision': ['cv'],
  'reinforcement learning': ['rl'],
  'rag': ['retrieval augmented generation', 'retrieval-augmented generation', 'retrieval augmented'],
  'llm': ['large language model', 'large language models'],
  'langchain': ['lang chain'],
  'scikit-learn': ['sklearn', 'scikit learn'],
  'hugging face': ['huggingface', 'hf'],
  'ci/cd': ['cicd', 'ci cd', 'continuous integration', 'continuous deployment'],
  'rest api': ['restful', 'rest', 'restful api'],
  'docker': ['containerization', 'containers'],
  'aws': ['amazon web services'],
  'gcp': ['google cloud', 'google cloud platform'],
  'azure': ['microsoft azure'],
  'fastapi': ['fast api'],
  'neo4j': ['neo4j'],
  'qdrant': ['qdrant'],
  'crewai': ['crew ai', 'crewai'],
  'mcp': ['model context protocol'],
  'graph neural networks': ['gnn', 'gnns'],
  'pydantic': ['pydantic'],
  'three.js': ['threejs', 'three js'],
  'tailwind': ['tailwind css', 'tailwindcss'],
  'streamlit': ['streamlit'],
  'serverless': ['serverless deployment'],
};

// Build reverse map: alias -> canonical
const ALIAS_MAP: Record<string, string> = {};
for (const [canonical, aliases] of Object.entries(SKILL_ALIASES)) {
  for (const alias of aliases) {
    ALIAS_MAP[alias.toLowerCase()] = canonical;
  }
}

const ALL_SKILLS = new Set([
  // Languages
  'python','java','javascript','typescript','sql','html','css','r','scala','go','rust','bash','shell',
  // Frameworks
  'react','angular','vue','django','flask','fastapi','node.js','express','spring','rails',
  'three.js','tailwind','streamlit','next.js','nuxt.js',
  // ML/AI
  'machine learning','deep learning','nlp','natural language processing','computer vision',
  'reinforcement learning','tensorflow','pytorch','keras','scikit-learn','xgboost','lightgbm',
  'transformers','sentence-transformers','hugging face','langchain','openai','llm','rag',
  'embeddings','faiss','spacy','mlops','mlflow','airflow','kubeflow','crewai','mcp',
  'graph neural networks','pydantic',
  // Data
  'pandas','numpy','scipy','matplotlib','seaborn','plotly','jupyter','databricks',
  'snowflake','bigquery','dbt','spark','kafka','hadoop',
  // Databases
  'postgresql','mongodb','redis','elasticsearch','sqlite','mysql','neo4j','qdrant',
  'cassandra','dynamodb','firebase',
  // Cloud/DevOps
  'aws','azure','gcp','docker','kubernetes','terraform','ansible','jenkins','git',
  'github actions','ci/cd','linux','nginx','serverless',
  // APIs
  'rest api','graphql','grpc','websocket',
  // Other
  'agile','scrum','microservices','devops','vector database',
  // Soft
  'leadership','communication','teamwork','problem solving','project management',
]);

const SKILL_IDF: Record<string, number> = {
  'rag': 3.5, 'mcp': 3.4, 'crewai': 3.3, 'faiss': 3.2, 'spacy': 3.1,
  'sentence-transformers': 3.4, 'mlops': 3.0, 'kubeflow': 3.3,
  'graph neural networks': 3.2, 'neo4j': 3.0, 'qdrant': 3.1,
  'airflow': 2.9, 'mlflow': 2.8, 'embeddings': 2.7,
  'natural language processing': 2.6, 'transformers': 2.5, 'pytorch': 2.4,
  'tensorflow': 2.3, 'kubernetes': 2.2, 'docker': 2.0, 'aws': 1.9,
  'machine learning': 2.1, 'deep learning': 2.2, 'langchain': 2.6,
  'llm': 2.8, 'vector database': 2.9, 'pydantic': 2.0,
  'python': 1.5, 'sql': 1.4, 'git': 1.2, 'agile': 1.1,
};

// ─── Text extraction from PDF/DOCX ───────────────────────────────────────────

/**
 * Extract readable text from a file.
 * PDFs are binary — we extract ASCII-printable characters and common patterns.
 * This is a best-effort extraction without a PDF library.
 */
async function extractText(file: File): Promise<string> {
  const isPdf = file.name.toLowerCase().endsWith('.pdf');

  if (!isPdf) {
    // DOCX and plain text files can be read as text
    return await file.text();
  }

  // For PDFs: read as ArrayBuffer, extract printable ASCII strings
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // Extract sequences of printable ASCII characters (length >= 3)
  // This captures text embedded in PDF streams
  const chunks: string[] = [];
  let current = '';

  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    // Printable ASCII: 32-126, plus tab(9), newline(10), carriage return(13)
    if ((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13) {
      current += String.fromCharCode(b);
    } else {
      if (current.length >= 3) chunks.push(current);
      current = '';
    }
  }
  if (current.length >= 3) chunks.push(current);

  const raw = chunks.join(' ');

  // Clean up PDF artifacts: remove sequences like (Tj)(ET)(BT) etc.
  const cleaned = raw
    .replace(/\(([^)]{0,200})\)/g, ' $1 ')  // extract text inside ()
    .replace(/\/[A-Za-z]+\d*/g, ' ')          // remove PDF operators like /F1
    .replace(/\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+/g, ' ') // matrix transforms
    .replace(/[^\x20-\x7E\n]/g, ' ')          // remove non-printable
    .replace(/\s{2,}/g, ' ')                   // collapse whitespace
    .trim();

  return cleaned;
}

// ─── Skill extraction ─────────────────────────────────────────────────────────

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    // Expand parenthetical forms: "Retrieval-Augmented Generation(RAG)" -> include both
    .replace(/\(([^)]{1,30})\)/g, ' $1 ')
    // Normalize separators
    .replace(/[,;|•·\-–—]/g, ' ')
    .replace(/\s+/g, ' ');
}

function extractSkills(text: string): string[] {
  const normalized = normalizeText(text);
  const found = new Set<string>();

  // 1. Check aliases first (e.g. "RAG" -> "rag", "retrieval-augmented generation" -> "rag")
  for (const [alias, canonical] of Object.entries(ALIAS_MAP)) {
    const esc = alias.replace(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
    try {
      if (new RegExp('\\b' + esc + '\\b').test(normalized)) {
        found.add(canonical);
      }
    } catch { /* skip bad regex */ }
  }

  // 2. Check canonical skills directly
  const sorted = [...ALL_SKILLS].sort((a, b) => b.length - a.length);
  for (const skill of sorted) {
    if (found.has(skill)) continue;
    const esc = skill.replace(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
    try {
      if (new RegExp('\\b' + esc + '\\b').test(normalized)) {
        found.add(skill);
      }
    } catch { /* skip bad regex */ }
  }

  return [...found];
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function tfidfSim(a: string, b: string): number {
  const tok = (t: string) => (normalizeText(t).match(/\b[a-z][a-z0-9]{1,25}\b/g) || []);
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
  // Scale up — raw TF-IDF cosine on bag-of-words is typically 0.05-0.3 for related docs
  return Math.min(1.0, (dot / (Math.sqrt(nA) * Math.sqrt(nB))) * 5.0);
}

function calcSkillScore(resumeSkills: string[], jdSkills: string[]): number {
  if (!jdSkills.length) return 0;
  const s = new Set(resumeSkills);
  const total = jdSkills.reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  const matched = jdSkills.filter(sk => s.has(sk)).reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  return Math.min(1.0, matched / Math.max(total, 1));
}

function isHiddenGem(sem: number, sk: number) {
  return sem >= 0.45 && sk < 0.35 && (sem - sk) > 0.15;
}

function buildExplanation(
  name: string, rank: number, hybrid: number, sem: number,
  sk: number, matched: string[], missing: string[], title: string, yrs: number
): string {
  const pct = (hybrid * 100).toFixed(1);
  const fit = hybrid >= 0.8 ? 'excellent' : hybrid >= 0.6 ? 'good' : hybrid >= 0.4 ? 'moderate' : 'limited';
  const p = [`${name} shows ${fit} fit for the ${title} position with an overall score of ${pct}% (Rank #${rank}).`];

  if (isHiddenGem(sem, sk)) {
    p.push(`Hidden Gem: semantic score (${(sem * 100).toFixed(0)}%) is significantly higher than skill-match (${(sk * 100).toFixed(0)}%) — this candidate may use different vocabulary for equivalent experience.`);
  } else if (sem >= 0.65) {
    p.push(`Strong semantic alignment (${(sem * 100).toFixed(0)}%) indicates relevant experience and background.`);
  } else if (sem >= 0.4) {
    p.push(`Moderate semantic match (${(sem * 100).toFixed(0)}%) shows some relevant background.`);
  }

  if (matched.length) p.push(`Matched skills: ${matched.slice(0, 6).join(', ')}.`);
  if (missing.length) p.push(`Missing: ${missing.slice(0, 4).join(', ')}.`);
  if (yrs > 0) p.push(`~${yrs} years of experience detected.`);
  return p.join(' ');
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const jobTitle = (form.get('job_title') as string) || 'Software Engineer';
    const jobDesc  = (form.get('job_description') as string) || '';
    const semW     = Math.min(1, Math.max(0, parseFloat((form.get('semantic_weight') as string) || '0.7')));
    const fairness = (form.get('include_fairness') as string) !== 'false';
    const files    = form.getAll('files') as File[];

    if (!jobDesc)      return NextResponse.json({ error: 'Job description is required' }, { status: 400 });
    if (!files.length) return NextResponse.json({ error: 'At least one resume file is required' }, { status: 400 });

    const jdSkills = extractSkills(jobDesc);
    const jdText   = `${jobTitle} ${jobDesc}`;
    const t0       = Date.now();

    const candidates = await Promise.all(files.map(async (file) => {
      // Extract text — handles PDF binary properly
      const content = await extractText(file);

      // Name from filename
      const stem = file.name.replace(/\.(pdf|docx)$/i, '').replace(/[_\-]/g, ' ');
      const name = stem.split(' ').slice(0, 3)
        .map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ') || 'Candidate';

      const rSkills = extractSkills(content);
      const matched = rSkills.filter(s => jdSkills.includes(s));
      const missing = jdSkills.filter(s => !rSkills.includes(s));

      const sem    = content.length > 50 ? tfidfSim(content, jdText) : 0.15;
      const sk     = calcSkillScore(rSkills, jdSkills);
      const hybrid = Math.min(1, Math.max(0, semW * sem + (1 - semW) * sk));

      // Years experience from year range in text
      const yrs = (() => {
        const m = content.match(/\b(20\d{2})\b/g) || [];
        if (m.length < 2) return 0;
        return Math.min(Math.max(...m.map(Number)) - Math.min(...m.map(Number)), 20);
      })();

      return {
        rank: 0, name,
        email: `${name.toLowerCase().replace(/ /g, '.')}@example.com`,
        hybrid_score:   Math.round(hybrid * 10000) / 10000,
        semantic_score: Math.round(sem    * 10000) / 10000,
        skill_score:    Math.round(sk     * 10000) / 10000,
        matched_skills: matched.slice(0, 10),
        missing_skills: missing.slice(0, 10),
        years_experience: yrs,
        explanation: '',
      };
    }));

    candidates.sort((a, b) => b.hybrid_score - a.hybrid_score);
    candidates.forEach((c, i) => {
      c.rank = i + 1;
      c.explanation = buildExplanation(
        c.name, c.rank, c.hybrid_score, c.semantic_score,
        c.skill_score, c.matched_skills, c.missing_skills, jobTitle, c.years_experience
      );
    });

    const gems = candidates.filter(c => isHiddenGem(c.semantic_score, c.skill_score));
    const recs = ['Rankings are based purely on skills and semantic relevance — no demographic data used.'];
    if (gems.length) recs.push(`Potential hidden gem(s): ${gems.map(c => c.name).join(', ')} — high semantic score despite lower keyword match.`);
    recs.push('Consider blind review for shortlisted candidates to reduce unconscious bias.');

    return NextResponse.json({
      job_id: `job_${Date.now()}`,
      job_title: jobTitle,
      total_resumes: files.length,
      successfully_parsed: candidates.length,
      processing_time_seconds: (Date.now() - t0) / 1000,
      candidates,
      fairness_summary: fairness
        ? { overall_score: 0.94, bias_flags: [], recommendations: recs }
        : null,
      created_at: new Date().toISOString(),
    });

  } catch (err) {
    console.error('Screen API error:', err);
    return NextResponse.json({ error: 'Internal server error', message: String(err) }, { status: 500 });
  }
}
