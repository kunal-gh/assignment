import urllib.request
import urllib.error
import json

# Minimal valid PDF with text content
pdf_content = (
    b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
    b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n'
    b'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
    b'/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font '
    b'/Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n'
    b'4 0 obj\n<< /Length 60 >>\nstream\n'
    b'BT /F1 12 Tf 100 700 Td (Python RAG MCP LLM machine learning) Tj ET\n'
    b'endstream\nendobj\nxref\n0 5\n'
    b'0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n'
    b'0000000115 00000 n\n0000000274 00000 n\n'
    b'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n369\n%%EOF'
)

boundary = b'testboundary456'
CRLF = b'\r\n'

body = b'--testboundary456\r\n'
body += b'Content-Disposition: form-data; name="job_title"\r\n\r\n'
body += b'AI Engineer\r\n'
body += b'--testboundary456\r\n'
body += b'Content-Disposition: form-data; name="job_description"\r\n\r\n'
body += b'Python RAG MCP LLM machine learning NLP\r\n'
body += b'--testboundary456\r\n'
body += b'Content-Disposition: form-data; name="files"; filename="test.pdf"\r\n'
body += b'Content-Type: application/pdf\r\n\r\n'
body += pdf_content + b'\r\n'
body += b'--testboundary456--\r\n'

req = urllib.request.Request(
    'https://ai-resume-screener-api-5iq6.onrender.com/screen',
    data=body,
    headers={
        'Content-Type': 'multipart/form-data; boundary=testboundary456',
        'Origin': 'https://ai-resume-screener-sepia.vercel.app',
    },
    method='POST'
)

print("Sending request to Render ML backend...")
try:
    r = urllib.request.urlopen(req, timeout=120)
    data = json.loads(r.read().decode())
    print('SUCCESS! Status:', r.status)
    print('Candidates:', len(data.get('candidates', [])))
    if data.get('candidates'):
        c = data['candidates'][0]
        print('Top candidate:', c['name'])
        print('Score:', c['hybrid_score'])
        print('Matched skills:', c['matched_skills'])
        print('Model used:', data.get('model_used', 'unknown'))
    print('Processing time:', data.get('processing_time_seconds'), 's')
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code)
    print(e.read().decode()[:1000])
except Exception as e:
    print('Error:', type(e).__name__, str(e))
