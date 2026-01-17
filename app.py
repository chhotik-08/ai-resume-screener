import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import subprocess
import sys

# Check if model is installed; if not, download it
try:
    import en_core_web_lg
except ImportError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])

# Helper function to extract text
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.lower() # Convert to lowercase for easier matching

# Simple Skill Extractor logic
def extract_keywords(text):
    # You can expand this list with more skills
    skill_bank = [
        "python", "java", "c++", "javascript", "react", "sql", "aws", 
        "machine learning", "data analysis", "project management", 
        "communication", "docker", "kubernetes", "excel", "tableau"
    ]
    found_skills = [skill for skill in skill_bank if skill in text]
    return set(found_skills)

# --- UI Setup ---
st.set_page_config(page_title="AI Resume Insights", layout="wide")
st.title("📄 Smart AI Resume Screener")

# Sidebar
st.sidebar.header("Configuration")
job_description = st.sidebar.text_area("Paste Job Description:", height=250).lower()

# File Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze & Rank"):
    if job_description and uploaded_files:
        resumes_data = []
        jd_keywords = extract_keywords(job_description)

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_keywords = extract_keywords(text)
            
            # Find intersection (matching keywords)
            matched = jd_keywords.intersection(resume_keywords)
            resumes_data.append({
                "name": file.name,
                "text": text,
                "matched_skills": list(matched)
            })

        # TF-IDF Ranking Logic
        all_texts = [job_description] + [r["text"] for r in resumes_data]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        # Combine scores with metadata and sort
        for i, r in enumerate(resumes_data):
            r["score"] = scores[i]

        sorted_resumes = sorted(resumes_data, key=lambda x: x["score"], reverse=True)

        # --- Display Results ---
        st.write("### 🏆 Candidate Rankings & Skill Match")
        
        for i, res in enumerate(sorted_resumes):
            with st.expander(f"Rank {i+1}: {res['name']} ({round(res['score']*100, 2)}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Match Percentage**")
                    st.progress(res['score'])
                
                with col2:
                    st.write("**Matched Keywords:**")
                    if res['matched_skills']:
                        # Displaying skills as tags
                        st.success(", ".join(res['matched_skills']))
                    else:
                        st.warning("No specific skill keywords matched.")

    else:

        st.error("Please upload resumes and provide a job description.")
