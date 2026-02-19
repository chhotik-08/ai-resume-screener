import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# --- 1. THEME & INTERFACE CONFIG ---
st.set_page_config(page_title="TalentFlow AI", page_icon="⚡", layout="wide")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 5px;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Button Polish */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        width: 100%;
        border: none;
        padding: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return " ".join([w for w in text.split() if w not in stop_words])

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='color: #2563EB;'>⚡ TalentFlow</h2>", unsafe_allow_html=True)
    st.write("---")
    st.subheader("📁 Upload Center")
    jd_input = st.text_area("Job Description", placeholder="Paste JD requirements...", height=200)
    uploaded_files = st.file_uploader("Candidate Resumes (PDF)", type="pdf", accept_multiple_files=True)
    st.write("---")
    analyze_btn = st.button("Run AI Screening")

# --- 4. MAIN CONTENT AREA ---
st.markdown("<div class='main-header'>Recruitment Dashboard</div>", unsafe_allow_html=True)
st.write("Real-time AI analysis of candidate compatibility.")

if analyze_btn and jd_input and uploaded_files:
    with st.spinner('AI Engine processing documents...'):
        clean_jd = clean_text(jd_input)
        resumes_data = []
        
        for file in uploaded_files:
            reader = PdfReader(file)
            raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
            resumes_data.append({"Name": file.name, "Clean": clean_text(raw_text)})

        # Vectorization
        texts = [clean_jd] + [r["Clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        # Data Prep
        results_df = pd.DataFrame({
            "Candidate": [r["Name"] for r in resumes_data],
            "Score": [round(s * 100, 1) for s in scores]
        }).sort_values(by="Score", ascending=False)

        # --- Front-End Layout ---
        col1, col2, col3 = st.columns(3)
        
        # Metric Card 1: Total Candidates
        with col1:
            st.markdown(f"<div class='metric-card'><h3>{len(uploaded_files)}</h3><p>Resumes Analyzed</p></div>", unsafe_allow_html=True)
        
        # Metric Card 2: Top Scorer
        with col2:
            top_name = results_df.iloc[0]['Candidate']
            st.markdown(f"<div class='metric-card'><h3>{results_df.iloc[0]['Score']}%</h3><p>Top Match: {top_name}</p></div>", unsafe_allow_html=True)
            
        # Metric Card 3: Download Button
        with col3:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.download_button("📥 Export CSV", data=csv, file_name='talent_report.csv')
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("---")
        
        # Main Dashboard split
        res_col, chart_col = st.columns([1.5, 1])
        
        with res_col:
            st.subheader("Detailed Ranking")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
        with chart_col:
            st.subheader("Score Distribution")
            st.bar_chart(results_df.set_index("Candidate"), color="#2563EB")

else:
    st.image("https://cdn-icons-png.flaticon.com/512/5087/5087579.png", width=100)
    st.warning("👈 Please provide a Job Description and Resumes in the sidebar to begin.")
