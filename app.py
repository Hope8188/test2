import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Gazette - Data Magazine", layout="wide")

import dotenv
dotenv.load_dotenv(".env.local")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# --- Constants ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
SITE_URL = "http://localhost:8501" 
SITE_NAME = "Gazette"

# --- Models & API Logic ---
def get_editorial_copy(data_summary):
    prompt = f"""
    You are a world-class business magazine editor. 
    Transform the following raw data summary into a high-end editorial for a digital magazine.
    Data Summary: {data_summary}

    Structure:
    1. A dramatic headline for the 'Cover Story'.
    2. An 'Executive Insights' narrative (approx 200 words) using premium business English.
    3. Three 'Key Growth Pillars' bullets with a short descriptive sentence for each.
    4. A final 'Editor's Outlook' summarizing future direction.
    
    Maintain a polished, analytical, yet captivating tone. Do not use markdown headers, just plain text with labels like [HEADLINE], [INSIGHTS], etc.
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-flash-1.5", # High speed & cost efficiency
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[HEADLINE] Generation Failed\n[INSIGHTS] Error connecting to OpenRouter: {str(e)}"

# --- UI Layout ---

# Title section
st.markdown('<div class="magazine-header">GAZETTE</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.5rem; color: #6366f1; letter-spacing: 2px;'>DATA TRANSFORMED INTO STORY</p>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("## Configuration")
    uploaded_file = st.file_uploader("Drop your data (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully.")
        st.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        
    st.markdown("---")
    st.markdown("### Export")
    if st.button("Print to PDF"):
        st.info("Use Ctrl+P (Command+P) to print this layout to a high-quality PDF.")

# Main content
if uploaded_file:
    # 1. Summarize Data 
    summary_stats = df.describe().to_string()
    column_names = list(df.columns)
    
    # 2. Hero Section 
    st.markdown('<div class="magazine-card">', unsafe_allow_html=True)
    
    if st.button("✨ Draft New Edition (Generate Insights)"):
        with st.spinner("Analyzing data and writing editorial..."):
            st.session_state.editorial = get_editorial_copy(f"Columns: {column_names} | Statistics: {summary_stats}")
            
    if 'editorial' in st.session_state:
        editorial = st.session_state.editorial
        
        # Parsing LLM output safely
        parts = editorial.split('[')
        content = {}
        for p in parts:
            if ']' in p:
                try:
                    tag, val = p.split(']', 1)
                    content[tag.strip()] = val.strip()
                except ValueError:
                    pass

        # Beautiful Editorial Layout
        st.markdown(f"<h1 style='font-size:3rem; margin-bottom: 0;'>{content.get('HEADLINE', 'The Quarterly Signal')}</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin-top:1rem; margin-bottom: 2rem;'/>", unsafe_allow_html=True)
        
        st.markdown(f'<div class="editorial-text">{content.get("INSIGHTS", "Loading insights...")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. Data Visuals in Magazine Style
        st.markdown("### <br>Market Visualization", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                num_cols = df.select_dtypes(include=['number']).columns
                fig = px.area(df, x=df.index, y=num_cols[0], 
                             title=f"Evolution of {num_cols[0]}",
                             template="plotly_dark",
                             color_discrete_sequence=['#818cf8'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Needs numeric columns for charts.")

        with col2:
            for col_name in df.select_dtypes(include=['number']).columns[:3]:
                avg = df[col_name].mean()
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #6366f1;">
                    <p style="margin:0; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Average {col_name}</p>
                    <h3 style="margin:0; font-size: 2rem; color: #e0e7ff;">{avg:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)

        # 4. Detailed Data Grid
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Explore Raw Dataset Source"):
            st.dataframe(df, use_container_width=True)
    else:
        st.markdown("<p style='color: #94a3b8; font-style: italic;'>Click 'Draft New Edition' to generate your magazine.</p></div>", unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div class="magazine-card" style="text-align: center; padding: 4rem 2rem;">
        <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">No Edition Loaded.</h2>
        <p style="color: #94a3b8; font-size: 1.2rem;">Please upload a CSV file in the sidebar to generate the latest magazine edition tailored to your data.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer for Export
st.sidebar.markdown("---")
if st.sidebar.button("Print to PDF"):
    st.sidebar.info("Use Ctrl+P (Command+P) to print this layout to a high-quality PDF.")
