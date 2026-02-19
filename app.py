import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# --- 1. THEMES & STYLING ---
st.set_page_config(page_title="TalentMatch AI", page_icon="🏢", layout="wide")

# Custom CSS for a Professional UI
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Modern Background Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(239, 246, 249) 0%, rgb(206, 239, 253) 90%);
    }

    /* Glassmorphism Card Effect */
    .result-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: 0.3s all ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1063/1063376.png", width=80)
    st.title("TalentMatch AI")
    st.markdown("---")
    st.subheader("🛠️ Configuration")
    
    jd_input = st.text_area("Job Requirements", placeholder="Paste the Job Description here...", height=250)
    uploaded_files = st.file_uploader("Candidate Resumes", type="pdf", accept_multiple_files=True)
    
    st.markdown("---")
    analyze_btn = st.button("Generate Rankings")

# --- 4. MAIN DASHBOARD ---
# Title Area
st.markdown("<h1>Recruitment Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Upload resumes to compare candidate profiles against job requirements using AI-powered semantic analysis.</p>", unsafe_allow_html=True)

if analyze_btn and jd_input and uploaded_files:
    # Wrap results in a card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    with st.spinner('🔍 Analyzing candidate profiles...'):
        clean_jd = clean_text(jd_input)
        resumes_data = []
        
        for file in uploaded_files:
            reader = PdfReader(file)
            raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
            resumes_data.append({"Name": file.name, "Clean": clean_text(raw_text)})

        # NLP Similarity Engine
        texts = [clean_jd] + [r["Clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        # Build DataFrame
        results_df = pd.DataFrame({
            "Candidate Name": [r["Name"] for r in resumes_data],
            "Match Score (%)": [round(s * 100, 1) for s in scores]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Show Results
        st.success(f"Successfully analyzed {len(uploaded_files)} resumes.")
        
        # Split display into two columns
        col_res, col_chart = st.columns([2, 1])
        
        with col_res:
            st.markdown("### 🏆 Top Candidates")
            st.dataframe(
                results_df.style.background_gradient(cmap='Blues', subset=["Match Score (%)"]),
                use_container_width=True,
                hide_index=True
            )
        
        with col_chart:
            st.markdown("### 📊 Distribution")
            st.bar_chart(results_df.set_index("Candidate Name"))

        # Download Report
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Detailed Report",
            data=csv,
            file_name='talent_match_report.csv',
            mime='text/csv'
        )
    st.markdown('</div>', unsafe_allow_html=True)

elif not analyze_btn:
    # Landing Page Visual
    st.markdown("""
        <div style="text-align: center; padding: 100px; color: #94a3b8;">
            <img src="https://cdn-icons-png.flaticon.com/512/2618/2618576.png" width="120" style="opacity: 0.5;">
            <h3>Your analysis results will appear here</h3>
            <p>Use the sidebar to upload documents and start the engine.</p>
        </div>
    """, unsafe_allow_html=True)
