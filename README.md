# 📝 Resume Feedback Assistant

This project analyzes a resume against a job description using natural language processing and skill-matching techniques. It gives specific feedback on matching skills, missing qualifications, formatting tips, and overall alignment.

## 🔍 Features

- PDF Resume text extraction (PyPDF2, pdfplumber, OCR)
- Job description matching via SentenceTransformer embeddings
- Cosine similarity scoring
- Skill matching in categories (programming, data science, cloud, etc.)
- Text generation feedback using Flan-T5 (optional)
- Clean, modular class-based implementation

## 🚀 How to Use

### 1. Colab Notebook (Recommended)
[Open in Google Colab](https://colab.research.google.com)

### 2. Local Python Script

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python resume_feedback_assistant.py
```

## 📦 Requirements

- sentence-transformers
- transformers
- torch
- pytesseract
- pdf2image
- poppler-utils
- nltk
- PyPDF2

## 📁 Sample Data

Include your own `resume.pdf` and `job_description.txt`, or use the provided samples(examples).

## 🙌 Credits

Developed by Vinayak Seth  
CSE - Data Science @ Manipal University Jaipur  

