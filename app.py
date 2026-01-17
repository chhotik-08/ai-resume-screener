import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def clean_and_extract_skills(text):
    # 1. Basic Cleaning: Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    
    # 2. Tokenize and remove stop words
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    # 3. Simple Keyword Matching (Skill Bank)
    skill_bank = ["python", "java", "sql", "aws", "docker", "machine learning", "react", "excel"]
    found_skills = [skill for skill in skill_bank if skill in filtered_words]
    
    return " ".join(filtered_words), set(found_skills)

# --- UI Setup ---
st.title("📄 Lightweight AI Resume Screener")

jd_input = st.text_area("Paste Job Description:")
uploaded_files = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True)

if st.button("Rank Now"):
    if jd_input and uploaded_files:
        clean_jd, jd_skills = clean_and_extract_skills(jd_input)
        
        resumes_list = []
        for file in uploaded_files:
            # Extract PDF Text
            reader = PdfReader(file)
            raw_text = " ".join([page.extract_text() for page in reader.pages])
            
            # Clean and get skills
            clean_res, res_skills = clean_and_extract_skills(raw_text)
            matched_skills = jd_skills.intersection(res_skills)
            
            resumes_list.append({
                "name": file.name,
                "clean_text": clean_res,
                "matched": matched_skills
            })

        # Calculate Scores
        all_texts = [clean_jd] + [r["clean_text"] for r in resumes_list]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Display Results
        for i, res in enumerate(resumes_list):
            st.write(f"### {i+1}. {res['name']}")
            st.write(f"**Match Score:** {round(scores[i]*100, 2)}%")
            st.write(f"**Matching Skills:** {', '.join(res['matched'])}")
            st.divider()
