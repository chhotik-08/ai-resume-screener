import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# --- 1. THEMES & EMERALD STYLING ---
st.set_page_config(page_title="Emerald Resume AI", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Emerald Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #059669 100%);
    }

    /* Glass Container for Main Content */
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-text {
        color: white !important;
    }

    /* Emerald Buttons */
    .stButton>button {
        background: #10b981;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px;
        font-weight: 700;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background: #059669;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }

    /* Styling Table/Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
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

# --- 3. SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("<h1 style='color: white;'>🌲 EmeraldAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a7f3d0;'>Next-Gen Talent Acquisition</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### ⚙️ Setup")
    jd_input = st.text_area("Job Description", placeholder="Paste requirements here...", height=200)
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    analyze_btn = st.button("Run Emerald Analysis")

# --- 4. MAIN INTERFACE ---
st.markdown("<h1 style='color: white; font-size: 3rem;'>Candidate Ranking Dashboard</h1>", unsafe_allow_html=True)

if analyze_btn and jd_input and uploaded_files:
    # Main White Card Start
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    with st.spinner('Emerald Engine is evaluating candidates...'):
        clean_jd = clean_text(jd_input)
        resumes_data = []
        
        for file in uploaded_files:
            reader = PdfReader(file)
            raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
            resumes_data.append({"Name": file.name, "Clean": clean_text(raw_text)})

        # Semantic Similarity
        texts = [clean_jd] + [r["Clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        results_df = pd.DataFrame({
            "Candidate Name": [r["Name"] for r in resumes_data],
            "Match Score (%)": [round(s * 100, 1) for s in scores]
        }).sort_values(by="Match Score (%)", ascending=False)

        # Dashboard Visuals
        col_metric, col_chart = st.columns([1, 2])
        
        with col_metric:
            st.metric("Total Resumes", len(uploaded_files))
            st.success("✅ Analysis Complete")
            
            # Top Candidate Highlight
            top_name = results_df.iloc[0]["Candidate Name"]
            st.markdown(f"**Top Pick:** {top_name}")

        with col_chart:
            st.bar_chart(results_df.set_index("Candidate Name"), color="#10b981")

        st.divider()
        st.markdown("### 🏆 Detailed Rankings")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Report", data=csv, file_name='emerald_ranking.csv')

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Aesthetic Empty State
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 60px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2); text-align: center;'>
            <h2 style='color: white;'>Ready to find the best talent?</h2>
            <p style='color: #d1fae5;'>Configure the job description and upload resumes in the sidebar to begin.</p>
        </div>
    """, unsafe_allow_html=True)
