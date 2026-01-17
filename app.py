import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_and_extract_skills(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    # Expanded Skill Bank
    skill_bank = ["python", "java", "sql", "aws", "docker", "machine learning", "react", "excel", "tableau", "c++", "javascript"]
    found_skills = [skill for skill in skill_bank if skill in filtered_words]
    
    return " ".join(filtered_words), set(found_skills)

# --- UI Setup ---
st.set_page_config(page_title="Resume Ranker Pro", layout="wide")
st.title("🏆 AI Resume Ranker & Reporter")

st.sidebar.header("Input Section")
jd_input = st.sidebar.text_area("Paste Job Description:", height=300)
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Generate Ranking Report"):
    if jd_input and uploaded_files:
        clean_jd, jd_skills = clean_and_extract_skills(jd_input)
        
        report_data = []
        
        for file in uploaded_files:
            reader = PdfReader(file)
            raw_text = " ".join([page.extract_text() for page in reader.pages])
            
            clean_res, res_skills = clean_and_extract_skills(raw_text)
            matched_skills = jd_skills.intersection(res_skills)
            
            # Temporary storage for scoring
            report_data.append({
                "Candidate Name": file.name,
                "Clean Text": clean_res,
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None"
            })

        # Calculate Similarity Scores
        all_texts = [clean_jd] + [r["Clean Text"] for r in report_data]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        # Build Final DataFrame
        for i, r in enumerate(report_data):
            r["Match Score (%)"] = round(scores[i] * 100, 2)
        
        # Create DataFrame and remove hidden text column for display
        df = pd.DataFrame(report_data)
        display_df = df.drop(columns=["Clean Text"]).sort_values(by="Match Score (%)", ascending=False)

        # --- Display Results ---
        st.write("### 📊 Ranking Summary")
        st.dataframe(display_df, use_container_width=True)

        # --- Download Button ---
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='resume_ranking_report.csv',
            mime='text/csv',
        )
    else:
        st.error("Please provide both Job Description and Resumes.")
