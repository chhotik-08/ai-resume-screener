import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os
import subprocess

# --- 1. Robust AI Model Loader ---
@st.cache_resource
def load_nlp_model():
    model_name = "en_core_web_lg"
    try:
        # Check if model exists
        return spacy.load(model_name)
    except OSError:
        # If not found, download it automatically
        st.info(f"Downloading AI Model ({model_name}). Please wait...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

# Initialize spaCy
nlp = load_nlp_model()

# --- 2. Helper Functions ---
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_keywords(text):
    # Rule-based skill extraction using spaCy
    doc = nlp(text)
    # Common tech labels usually fall under 'ORG' or 'PRODUCT' in standard NER, 
    # but for this basic version, we will look for tokens in a custom bank.
    skill_bank = ["python", "java", "sql", "aws", "docker", "machine learning", "excel", "tableau", "react", "javascript"]
    found_skills = [token.text.lower() for token in doc if token.text.lower() in skill_bank]
    return set(found_skills)

# --- 3. UI Setup ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 Smart AI Resume Screening & Ranking")

st.sidebar.header("Job Details")
job_description = st.sidebar.text_area("Paste the Job Description here:", height=300).lower()

uploaded_files = st.file_uploader("Upload Resumes (PDF format)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze Resumes"):
    if job_description and uploaded_files:
        resumes_data = []
        jd_keywords = extract_keywords(job_description)

        with st.status("Analyzing Resumes..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resume_keywords = extract_keywords(text)
                
                # Compare keywords
                matched = jd_keywords.intersection(resume_keywords)
                resumes_data.append({
                    "name": file.name,
                    "text": text,
                    "matched_skills": list(matched)
                })

            # AI Vectorization and Similarity Scoring
            all_texts = [job_description] + [r["text"] for r in resumes_data]
            cv = TfidfVectorizer()
            matrix = cv.fit_transform(all_texts)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

            for i, r in enumerate(resumes_data):
                r["score"] = scores[i]

        # Sorting results by score
        sorted_resumes = sorted(resumes_data, key=lambda x: x["score"], reverse=True)

        st.success("✅ Analysis Complete!")
        
        # Displaying results
        for i, res in enumerate(sorted_resumes):
            score_percent = round(res['score'] * 100, 2)
            with st.expander(f"Rank {i+1}: {res['name']} — {score_percent}% Match"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Score", f"{score_percent}%")
                with col2:
                    st.write("**Matched Keywords:**")
                    st.write(", ".join(res['matched_skills']) if res['matched_skills'] else "No common keywords found.")
    else:
        st.warning("Please provide both a Job Description and at least one Resume.")
