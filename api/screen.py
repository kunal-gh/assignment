"""
Vercel Serverless Function for resume screening.
Handles file uploads and processing in serverless environment.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.job import JobDescription
from src.parsers.resume_parser import ResumeParser
from src.ranking.ranking_engine import RankingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handler(request):
    """
    Vercel serverless function handler.
    
    Handles POST requests with multipart/form-data containing:
    - files: List of resume files (PDF/DOCX)
    - job_title: Job title string
    - job_description: Job description text
    - semantic_weight: Float (0-1)
    - include_fairness: Boolean
    - embedding_model: Model name string
    """
    
    # CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }
    
    # Handle OPTIONS request (CORS preflight)
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    # Only allow POST
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Parse form data
        form_data = request.form
        files = request.files.getlist('files')
        
        job_title = form_data.get('job_title', 'Software Engineer')
        job_description = form_data.get('job_description', '')
        semantic_weight = float(form_data.get('semantic_weight', 0.7))
        include_fairness = form_data.get('include_fairness', 'true').lower() == 'true'
        embedding_model = form_data.get('embedding_model', 'all-MiniLM-L6-v2')
        
        # Validate inputs
        if not job_description:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'Job description is required'})
            }
        
        if not files:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'At least one resume file is required'})
            }
        
        # Save uploaded files to temp directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file in files:
            if not file.filename.lower().endswith(('.pdf', '.docx')):
                continue
            
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            file_paths.append(temp_path)
        
        if not file_paths:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'No valid resume files (PDF/DOCX only)'})
            }
        
        # Initialize components
        logger.info(f"Processing {len(file_paths)} resumes...")
        
        parser = ResumeParser()
        generator = EmbeddingGenerator(model_name=embedding_model)
        engine = RankingEngine(
            semantic_weight=semantic_weight,
            skill_weight=1.0 - semantic_weight,
            embedding_generator=generator
        )
        
        # Create job description
        job_desc = JobDescription(
            title=job_title,
            description=job_description
        )
        
        # Parse resumes
        resumes = parser.batch_parse(file_paths)
        
        # Process batch
        results = engine.process_batch(
            resumes=resumes,
            job_desc=job_desc,
            include_fairness=include_fairness
        )
        
        # Build response
        candidates_out = []
        for c in results.ranked_candidates:
            resume = c.resume
            name = resume.contact_info.name if resume.contact_info else "Unknown"
            email = resume.contact_info.email if resume.contact_info else None
            
            # Get skill analysis
            from src.ranking.skill_matcher import SkillMatcher
            matcher = SkillMatcher()
            analysis = matcher.analyze_skill_match(
                resume.skills,
                job_desc.required_skills,
                job_desc.preferred_skills,
            )
            
            candidates_out.append({
                'rank': c.rank,
                'name': name,
                'email': email,
                'hybrid_score': round(c.hybrid_score, 4),
                'semantic_score': round(c.semantic_score, 4),
                'skill_score': round(c.skill_score, 4),
                'matched_skills': analysis.get('matched_required', [])[:10],
                'missing_skills': analysis.get('missing_required', [])[:10],
                'years_experience': resume.get_years_of_experience(),
                'explanation': c.explanation,
            })
        
        # Fairness summary
        fairness_summary = None
        if results.fairness_report:
            fr = results.fairness_report
            fairness_summary = {
                'overall_score': fr.get_overall_fairness_score(),
                'bias_flags': fr.bias_flags,
                'recommendations': fr.recommendations[:3],
            }
        
        response_data = {
            'job_id': results.job_id,
            'job_title': job_title,
            'total_resumes': results.total_resumes,
            'successfully_parsed': results.successfully_parsed,
            'processing_time_seconds': round(results.processing_time, 3),
            'candidates': candidates_out,
            'fairness_summary': fairness_summary,
            'created_at': results.processing_time,
        }
        
        # Cleanup temp files
        for path in file_paths:
            try:
                os.unlink(path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
