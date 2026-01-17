import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# --- Custom Styling ---
st.set_page_config(page_title="ResumeAI Pro", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .css-10trblm {
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Logic ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# --- Website Layout ---
st.title("🎯 ResumeAI Pro")
st.markdown("##### *The intelligent way to find your next top hire.*")
st.divider()

# Create Tabs for a cleaner look
tab1, tab2 = st.tabs(["🚀 Screener", "📖 How it Works"])

with tab1:
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("Configuration")
        jd_input = st.text_area("Target Job Description", placeholder="Paste the JD here...", height=250)
        uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
        analyze_btn = st.button("Start AI Analysis")

    with col2:
        st.subheader("Analysis Dashboard")
        if analyze_btn and jd_input and uploaded_files:
            with st.spinner('AI is processing resumes...'):
                clean_jd = clean_text(jd_input)
                resumes_data = []
                
                for file in uploaded_files:
                    reader = PdfReader(file)
                    raw_text = " ".join([page.extract_text() for page in reader.pages])
                    resumes_data.append({"Name": file.name, "Clean": clean_text(raw_text)})

                # Scoring
                texts = [clean_jd] + [r["Clean"] for r in resumes_data]
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(texts)
                scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

                # Results
                results_df = pd.DataFrame({
                    "Candidate Name": [r["Name"] for r in resumes_data],
                    "Match Score": [round(s * 100, 1) for s in scores]
                }).sort_values(by="Match Score", ascending=False)

                st.balloons()
                st.success("Analysis Complete!")
                
                # Use a stylized table
                st.dataframe(results_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Report to CSV", data=csv, file_name='hr_report.csv')
        else:
            st.info("Waiting for input... Upload resumes and paste a JD on the left to begin.")

with tab2:
    st.write("### How the AI Works")
    st.write("""
    1. **Text Extraction:** We use PyPDF2 to pull raw data from your PDF files.
    2. **Natural Language Processing:** NLTK filters out 'stop words' to focus on core skills.
    3. **Vectorization:** TF-IDF converts words into mathematical vectors.
    4. **Cosine Similarity:** We measure the 'angle' between your JD and the Resume to find the perfect match.
    """)
