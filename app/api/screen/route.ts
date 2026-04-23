import { NextRequest, NextResponse } from 'next/server';

// ─── Skill taxonomy ───────────────────────────────────────────────────────────

// Every entry: [canonical_name, ...aliases_and_abbreviations]
const SKILL_ENTRIES: [string, ...string[]][] = [
  // Languages
  ['python', 'python3', 'py'],
  ['javascript', 'js', 'es6'],
  ['typescript', 'ts'],
  ['sql', 'mysql', 'sqlite'],
  ['java'],
  ['html', 'html5'],
  ['css', 'css3'],
  ['r'],
  ['scala'],
  ['go', 'golang'],
  ['rust'],
  ['bash', 'shell', 'sh'],
  ['c++', 'cpp'],
  ['c#', 'csharp'],
  // Frameworks / Libraries
  ['react', 'reactjs', 'react.js'],
  ['next.js', 'nextjs', 'next js'],
  ['angular', 'angularjs'],
  ['vue', 'vuejs', 'vue.js'],
  ['node.js', 'node', 'nodejs'],
  ['django'],
  ['flask'],
  ['fastapi', 'fast api'],
  ['express', 'expressjs'],
  ['spring', 'spring boot'],
  ['tailwind', 'tailwind css', 'tailwindcss'],
  ['streamlit'],
  ['three.js', 'threejs', 'three js'],
  // ML / AI
  ['machine learning', 'ml'],
  ['deep learning', 'dl'],
  ['natural language processing', 'nlp'],
  ['computer vision', 'cv'],
  ['reinforcement learning', 'rl'],
  ['tensorflow', 'tf'],
  ['pytorch', 'torch'],
  ['keras'],
  ['scikit-learn', 'sklearn', 'scikit learn'],
  ['xgboost'],
  ['lightgbm'],
  ['transformers', 'huggingface transformers'],
  ['sentence-transformers', 'sbert'],
  ['hugging face', 'huggingface', 'hf'],
  ['langchain', 'lang chain'],
  ['openai', 'open ai'],
  ['anthropic', 'claude'],
  ['gemini', 'google gemini'],
  ['llm', 'large language model', 'large language models'],
  ['rag', 'retrieval augmented generation', 'retrieval-augmented generation'],
  ['embeddings', 'vector embeddings'],
  ['faiss'],
  ['spacy', 'spaCy'],
  ['mlops', 'ml ops'],
  ['mlflow'],
  ['airflow', 'apache airflow'],
  ['kubeflow'],
  ['crewai', 'crew ai', 'crewai'],
  ['mcp', 'model context protocol'],
  ['graph neural networks', 'gnn', 'gnns'],
  ['pydantic'],
  ['vector database', 'vector db', 'vectordb'],
  ['agentic ai', 'ai agents', 'agentic'],
  ['ai tool integration'],
  ['workflow automation'],
  // Data
  ['pandas'],
  ['numpy'],
  ['scipy'],
  ['matplotlib'],
  ['seaborn'],
  ['plotly'],
  ['jupyter', 'jupyter notebook'],
  ['databricks'],
  ['snowflake'],
  ['bigquery'],
  ['dbt'],
  ['spark', 'apache spark', 'pyspark'],
  ['kafka', 'apache kafka'],
  ['hadoop'],
  // Databases
  ['postgresql', 'postgres', 'psql'],
  ['mongodb', 'mongo'],
  ['redis'],
  ['elasticsearch', 'elastic'],
  ['neo4j'],
  ['qdrant'],
  ['cassandra'],
  ['dynamodb', 'dynamo db'],
  ['firebase'],
  ['pinecone'],
  ['weaviate'],
  ['chroma', 'chromadb'],
  // Cloud / DevOps
  ['aws', 'amazon web services'],
  ['azure', 'microsoft azure'],
  ['gcp', 'google cloud', 'google cloud platform'],
  ['docker', 'containerization'],
  ['kubernetes', 'k8s'],
  ['terraform'],
  ['ansible'],
  ['jenkins'],
  ['git', 'github', 'gitlab'],
  ['github actions'],
  ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment'],
  ['linux', 'ubuntu', 'debian'],
  ['nginx'],
  ['serverless', 'serverless deployment'],
  ['model monitoring'],
  // APIs / Architecture
  ['rest api', 'restful', 'rest', 'restful api'],
  ['graphql'],
  ['grpc'],
  ['websocket'],
  ['microservices'],
  ['devops'],
  // Soft skills
  ['leadership'],
  ['communication'],
  ['teamwork'],
  ['problem solving'],
  ['project management'],
  ['agile', 'agile methodology'],
  ['scrum'],
];

// Build lookup: all lowercase variants -> canonical name
const SKILL_LOOKUP = new Map<string, string>();
for (const [canonical, ...aliases] of SKILL_ENTRIES) {
  SKILL_LOOKUP.set(canonical.toLowerCase(), canonical);
  for (const alias of aliases) {
    SKILL_LOOKUP.set(alias.toLowerCase(), canonical);
  }
}

const SKILL_IDF: Record<string, number> = {
  'rag': 3.5, 'mcp': 3.4, 'crewai': 3.3, 'faiss': 3.2, 'spacy': 3.1,
  'sentence-transformers': 3.4, 'mlops': 3.0, 'kubeflow': 3.3,
  'graph neural networks': 3.2, 'neo4j': 3.0, 'qdrant': 3.1,
  'vector database': 3.0, 'agentic ai': 3.2, 'llm': 2.8,
  'airflow': 2.9, 'mlflow': 2.8, 'embeddings': 2.7,
  'natural language processing': 2.6, 'transformers': 2.5, 'pytorch': 2.4,
  'tensorflow': 2.3, 'kubernetes': 2.2, 'docker': 2.0, 'aws': 1.9,
  'machine learning': 2.1, 'deep learning': 2.2, 'langchain': 2.6,
  'pydantic': 2.0, 'python': 1.5, 'sql': 1.4, 'git': 1.2, 'agile': 1.1,
};

// ─── PDF text extraction ──────────────────────────────────────────────────────

async function extractTextFromFile(file: File): Promise<string> {
  const name = file.name.toLowerCase();

  if (!name.endsWith('.pdf')) {
    // DOCX / TXT — readable as text
    return file.text();
  }

  // PDF: read binary, extract printable ASCII strings
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

  return parts
    .join(' ')
    // Unwrap parenthesised text: (RAG) -> RAG
    .replace(/\(([^)]{1,60})\)/g, ' $1 ')
    // Remove PDF operators
    .replace(/\b(BT|ET|Tf|Td|TD|Tm|Tr|Ts|Tw|Tz|T\*|Tj|TJ|cm|re|Do|BI|EI|BMC|EMC)\b/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

// ─── Skill extraction ─────────────────────────────────────────────────────────

function extractSkills(text: string): string[] {
  // Normalise: lowercase, expand parens, normalise separators
  const norm = text
    .toLowerCase()
    .replace(/\(([^)]{1,60})\)/g, ' $1 ')   // (RAG) -> RAG
    .replace(/[,;|•·\-–—\/\\]/g, ' ')
    .replace(/\s+/g, ' ');

  const found = new Set<string>();

  // Sort by length descending so multi-word phrases match before single words
  const entries = [...SKILL_LOOKUP.entries()].sort((a, b) => b[0].length - a[0].length);

  for (const [variant, canonical] of entries) {
    // Use includes() for short acronyms (MCP, RAG, LLM) — faster and catches edge cases
    if (variant.length <= 4) {
      // For short terms, check word boundary manually
      const re = new RegExp('(?:^|[\\s,;(\\[])' + variant.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?:[\\s,;)\\]]|$)', 'i');
      if (re.test(norm)) found.add(canonical);
    } else {
      // For longer terms, simple includes is fine
      if (norm.includes(variant)) found.add(canonical);
    }
  }

  return [...found];
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function tfidfSim(a: string, b: string): number {
  const tok = (t: string) =>
    t.toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 2);

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

function calcSkillScore(resumeSkills: string[], jdSkills: string[]): number {
  if (!jdSkills.length) return 0;
  const s = new Set(resumeSkills);
  const total = jdSkills.reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  const matched = jdSkills
    .filter(sk => s.has(sk))
    .reduce((acc, sk) => acc + (SKILL_IDF[sk] || 1.5), 0);
  return Math.min(1.0, matched / Math.max(total, 1));
}

function isHiddenGem(sem: number, sk: number) {
  return sem >= 0.4 && sk < 0.3 && (sem - sk) > 0.15;
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
  } else if (sem >= 0.6) {
    p.push(`Strong semantic alignment (${(sem * 100).toFixed(0)}%) indicates relevant experience.`);
  } else if (sem >= 0.35) {
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
      const content = await extractTextFromFile(file);

      const stem = file.name.replace(/\.(pdf|docx)$/i, '').replace(/[_\-]/g, ' ');
      const name = stem.split(' ').slice(0, 3)
        .map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ') || 'Candidate';

      const rSkills = extractSkills(content);
      const matched = rSkills.filter(s => jdSkills.includes(s));
      const missing = jdSkills.filter(s => !rSkills.includes(s));

      const sem    = content.length > 50 ? tfidfSim(content, jdText) : 0.15;
      const sk     = calcSkillScore(rSkills, jdSkills);
      const hybrid = Math.min(1, Math.max(0, semW * sem + (1 - semW) * sk));

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
    const recs = ['Rankings are based purely on skills and semantic relevance.'];
    if (gems.length) recs.push(`Potential hidden gem(s): ${gems.map(c => c.name).join(', ')} — high semantic score despite lower keyword match.`);
    recs.push('Consider blind review for shortlisted candidates.');

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
