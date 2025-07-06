# -*- coding: utf-8 -*-

# !pip install -q pytesseract pdf2image sentence-transformers transformers faiss-cpu poppler-utils PyPDF2
import re
import numpy as np
from google.colab import files
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ImprovedResumeAnalyzer:
    def __init__(self):
        print("🚀 Initializing Improved Resume Analyzer...")

        # Load embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load text generation model (optional, with fallback)
        self.generator = None
        try:
            print("📥 Loading text generation model...")
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Using base for reliability
                max_length=512,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Text generation model loaded!")
        except Exception as e:
            print(f"⚠️ Text generation model not available: {e}")
            print("📝 Will use rule-based feedback generation")

        # Define comprehensive skill categories
        self.skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php',
                'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r','c','swift'
            ],
            'web_technologies': [
                'react', 'angular', 'vue', 'node.js', 'express', 'django',
                'flask', 'spring', 'laravel', 'html', 'css', 'bootstrap'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'oracle', 'sqlite', 'cassandra', 'dynamodb','nosql'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'git', 'ci/cd', 'devops'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data science', 'analytics',
                'statistics', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
                'pytorch', 'tableau', 'power bi','excel','seaborn','matplotlib'
            ],
            'soft_skills': [
                'leadership', 'communication', 'project management', 'agile',
                'scrum', 'teamwork', 'problem solving', 'analytical thinking'
            ]
        }

        print("✅ Analyzer initialized successfully!")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with multiple fallback methods"""
        methods_tried = []

        # Method 1: PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            if len(text.strip()) > 100:  # Reasonable amount of text
                return text, "PyPDF2"
            methods_tried.append("PyPDF2 (insufficient text)")
        except Exception as e:
            methods_tried.append(f"PyPDF2 failed: {str(e)[:50]}")

        # Method 2: pdfplumber (if available)
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if len(text.strip()) > 100:
                return text, "pdfplumber"
            methods_tried.append("pdfplumber (insufficient text)")
        except ImportError:
            methods_tried.append("pdfplumber not available")
        except Exception as e:
            methods_tried.append(f"pdfplumber failed: {str(e)[:50]}")

        # Method 3: OCR with pytesseract
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(pdf_path)
            text = ""
            for i, img in enumerate(images):
                page_text = pytesseract.image_to_string(img)
                text += f"\n--- Page {i+1} ---\n{page_text}"

            if len(text.strip()) > 50:
                return text, "OCR"
            methods_tried.append("OCR (insufficient text)")
        except Exception as e:
            methods_tried.append(f"OCR failed: {str(e)[:50]}")

        print(f"❌ All extraction methods failed: {methods_tried}")
        return "", "failed"

    def clean_and_normalize_text(self, text):
        """Advanced text cleaning and normalization"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\+\#$$$$]', ' ', text)

        # Normalize common variations
        text = re.sub(r'\bC\+\+\b', 'cpp', text, flags=re.IGNORECASE)
        text = re.sub(r'\bC#\b', 'csharp', text, flags=re.IGNORECASE)
        text = re.sub(r'\bNode\.js\b', 'nodejs', text, flags=re.IGNORECASE)

        return text

    def calculate_cosine_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        # Generate embeddings
        embeddings = self.embed_model.encode([text1, text2])

        # Calculate cosine similarity
        embedding1, embedding2 = embeddings[0], embeddings[1]

        # Cosine similarity formula
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        # Convert to percentage (0-100)
        similarity_percentage = max(0, min(100, cosine_sim * 100))

        return similarity_percentage, cosine_sim

    def extract_skills_advanced(self, text):
        """Advanced skill extraction with categorization"""
        text_lower = text.lower()
        found_skills = {}

        for category, skills in self.skill_categories.items():
            found_skills[category] = []
            for skill in skills:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill)

        return found_skills

    def generate_comprehensive_feedback(self, resume_text, jd_text, similarity_score, resume_skills, jd_skills):
        """Generate comprehensive feedback"""

        # Find matches and gaps
        all_resume_skills = set()
        all_jd_skills = set()

        for category in resume_skills:
            all_resume_skills.update(resume_skills[category])
        for category in jd_skills:
            all_jd_skills.update(jd_skills[category])

        matching_skills = all_resume_skills.intersection(all_jd_skills)
        missing_skills = all_jd_skills - all_resume_skills

        # Generate feedback based on similarity score
        if similarity_score >= 80:
            overall_assessment = "EXCELLENT MATCH! 🎉"
            priority = "Focus on fine-tuning and formatting"
        elif similarity_score >= 65:
            overall_assessment = "GOOD MATCH! 👍"
            priority = "Address missing skills and add more specific examples"
        elif similarity_score >= 50:
            overall_assessment = "MODERATE MATCH 📊"
            priority = "Significant improvements needed in skill alignment"
        else:
            overall_assessment = "LOW MATCH ⚠️"
            priority = "Major restructuring recommended"

        feedback = f"""
🎯 RESUME ANALYSIS RESULTS
{'='*60}

📊 SIMILARITY SCORE: {similarity_score:.1f}%
🎯 OVERALL ASSESSMENT: {overall_assessment}
🔍 PRIORITY: {priority}

✅ MATCHING SKILLS ({len(matching_skills)} found):
{self._format_skills_list(matching_skills)}

❌ MISSING CRITICAL SKILLS ({len(missing_skills)} identified):
{self._format_skills_list(missing_skills)}

📈 SKILL BREAKDOWN BY CATEGORY:
{self._format_skill_breakdown(resume_skills, jd_skills)}

🛠️ SPECIFIC IMPROVEMENT RECOMMENDATIONS:

1. CONTENT IMPROVEMENTS:
   • Add quantifiable achievements (e.g., "Increased efficiency by 25%")
   • Include specific project examples that demonstrate required skills
   • Highlight leadership and collaboration experiences

2. KEYWORD OPTIMIZATION:
   • Naturally incorporate missing keywords: {', '.join(list(missing_skills)[:5])}
   • Use industry-standard terminology from the job description
   • Add relevant certifications or training

3. STRUCTURE & FORMATTING:
   • Create a professional summary that mirrors the job requirements
   • Use bullet points with strong action verbs
   • Ensure consistent formatting and clear section headers

4. SKILL DEMONSTRATION:
   • Provide context for each technical skill mentioned
   • Show progression and growth in your career
   • Include relevant side projects or contributions

{'='*60}
"""
        return feedback

    def _format_skills_list(self, skills):
        """Format skills list for display"""
        if not skills:
            return "   • None identified"

        skills_list = sorted(list(skills))
        if len(skills_list) <= 10:
            return '\n'.join([f"   • {skill.title()}" for skill in skills_list])
        else:
            displayed = skills_list[:8]
            remaining = len(skills_list) - 8
            result = '\n'.join([f"   • {skill.title()}" for skill in displayed])
            result += f"\n   • ... and {remaining} more"
            return result

    def _format_skill_breakdown(self, resume_skills, jd_skills):
        """Format skill breakdown by category"""
        breakdown = ""
        for category in self.skill_categories.keys():
            resume_count = len(resume_skills.get(category, []))
            jd_count = len(jd_skills.get(category, []))

            if jd_count > 0:  # Only show categories that are relevant to the job
                match_rate = (resume_count / jd_count * 100) if jd_count > 0 else 0
                status = "✅" if match_rate >= 70 else "⚠️" if match_rate >= 40 else "❌"

                breakdown += f"   {status} {category.replace('_', ' ').title()}: "
                breakdown += f"{resume_count}/{jd_count} skills ({match_rate:.0f}%)\n"

        return breakdown.strip() if breakdown else "   • No specific skill categories identified"

    def analyze_resume(self):
        """Main analysis function"""
        print("🎯 IMPROVED RESUME FEEDBACK ASSISTANT")
        print("="*60)

        # File uploads
        print("\n📤 Upload your Resume (PDF format):")
        uploaded_resume = files.upload()
        if not uploaded_resume:
            print("❌ No resume uploaded!")
            return None

        resume_path = list(uploaded_resume.keys())[0]

        print("\n📤 Upload Job Description (TXT format):")
        uploaded_jd = files.upload()
        if not uploaded_jd:
            print("❌ No job description uploaded!")
            return None

        jd_path = list(uploaded_jd.keys())[0]

        # Extract and process texts
        print("\n🔍 Extracting text from resume...")
        resume_text, resume_method = self.extract_text_from_pdf(resume_path)

        if not resume_text:
            print("❌ Failed to extract text from resume!")
            return None

        resume_text = self.clean_and_normalize_text(resume_text)
        print(f"✅ Resume text extracted using {resume_method} ({len(resume_text)} characters)")

        print("\n📖 Reading job description...")
        try:
            with open(jd_path, 'r', encoding='utf-8') as f:
                jd_text = self.clean_and_normalize_text(f.read())
            print(f"✅ Job description loaded ({len(jd_text)} characters)")
        except Exception as e:
            print(f"❌ Failed to read job description: {str(e)}")
            return None

        # Calculate similarity using cosine similarity (most reliable method)
        print("\n🧮 Calculating similarity score...")
        similarity_score, raw_cosine = self.calculate_cosine_similarity(resume_text, jd_text)
        print(f"📊 Cosine Similarity: {raw_cosine:.4f}")
        print(f"📊 Similarity Score: {similarity_score:.1f}%")

        # Extract skills
        print("\n🔍 Analyzing skills...")
        resume_skills = self.extract_skills_advanced(resume_text)
        jd_skills = self.extract_skills_advanced(jd_text)

        # Generate comprehensive feedback
        print("\n📝 Generating comprehensive feedback...")
        feedback = self.generate_comprehensive_feedback(
            resume_text, jd_text, similarity_score, resume_skills, jd_skills
        )

        # Display results
        print(feedback)

        return {
            'similarity_score': similarity_score,
            'raw_cosine_similarity': raw_cosine,
            'feedback': feedback,
            'resume_skills': resume_skills,
            'jd_skills': jd_skills,
            'extraction_method': resume_method
        }

# Initialize and run the improved analyzer
if __name__ == "__main__":
    analyzer = ImprovedResumeAnalyzer()
    results = analyzer.analyze_resume()
