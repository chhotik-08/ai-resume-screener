import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# --- 1. Custom Aesthetic CSS ---
st.set_page_config(page_title="ResumeAI Pro", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #8e9eab 100%);
    }

    /* Main Container with Border and Shadow */
    .main-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    /* Styling Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Logic Implementation ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# --- 3. Header ---
st.title("🎯 ResumeAI Screening Pro")
st.markdown("##### *Smart Talent Acquisition through AI-driven Analytics*")
st.divider()

# --- 4. Two-Column Layout ---
# Using the with statement to wrap our layout in a styled div via markdown
st.markdown('<div class="main-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Input Specifications")
    st.info("Paste the job requirements and upload the candidate pool below.")
    
    jd_input = st.text_area("Job Description", placeholder="Enter requirements...", height=200)
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    analyze_btn = st.button("🚀 Analyze & Rank Candidates")

with col2:
    st.subheader("📊 Analytics Dashboard")
    
    if analyze_btn and jd_input and uploaded_files:
        with st.spinner('Calculating Match Scores...'):
            clean_jd = clean_text(jd_input)
            resumes_data = []
            
            for file in uploaded_files:
                reader = PdfReader(file)
                raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
                resumes_data.append({"Name": file.name, "Clean": clean_text(raw_text)})

            # Scoring Engine
            texts = [clean_jd] + [r["Clean"] for r in resumes_data]
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(texts)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

            # Prepare Results Table
            results_df = pd.DataFrame({
                "Rank": range(1, len(scores) + 1),
                "Candidate": [r["Name"] for r in resumes_data],
                "Match Score (%)": [round(s * 100, 1) for s in scores]
            }).sort_values(by="Match Score (%)", ascending=False)
            
            # Resetting rank after sorting
            results_df["Rank"] = range(1, len(results_df) + 1)

            st.balloons()
            st.dataframe(
                results_df.style.background_gradient(cmap='Blues', subset=["Match Score (%)"]),
                use_container_width=True,
                hide_index=True
            )
            
            # Download Section
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Recruitment Report", data=csv, file_name='ranking_report.csv')
    
    elif not analyze_btn:
        st.write("Results will appear here after analysis.")
        # Placeholder image/icon for empty state
        st.image("https://cdn-icons-png.flaticon.com/512/5087/5087579.png", width=150)

st.markdown('</div>', unsafe_allow_html=True)

# --- 5. Footer ---
st.divider()
f_col1, f_col2, f_col3 = st.columns(3)
with f_col2:
    st.markdown("<p style='text-align: center;'>Built by [Your Name] | Powered by Python & NLP</p>", unsafe_allow_html=True)
