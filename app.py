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
    You are a world-class magazine editor and senior data analyst. 
    Analyze the following raw data summary and transform it into a captivating, high-end editorial for a digital magazine.
    Data Summary: {data_summary}

    Context: I do not know what this data is about. It could be sales, healthcare, engineering metrics, or anything else.
    Your mission is to look at the columns and statistics, deduce the core theme, and tell the most compelling 'story' hidden in these numbers.

    Structure:
    1. A dramatic, relevant headline for the 'Cover Story'.
    2. An 'Executive Insights' narrative (approx 200 words) using premium, authoritative language.
    3. Three 'Key Discoveries' bullets with a short descriptive sentence for each.
    4. A final 'Editor's Outlook' summarizing future direction.
    
    Maintain a polished, analytical, yet captivating tone. Do not use markdown headers, just plain text with labels like [HEADLINE], [INSIGHTS], etc.
    If the data is completely unrecognizable, narrate the structure of the data itself creatively.
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-1.5-flash", # Fixed 404 string
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
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;'>The Data Brief</p>", unsafe_allow_html=True)

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
    if st.button("Download as PDF", key="export_pdf_btn"):
        # Inject JavaScript to trigger the native browser print/save-as-pdf dialog instantly
        print_js = """
        <script>
            window.print();
        </script>
        """
        st.components.v1.html(print_js, height=0, width=0)

# Main content
if uploaded_file:
    # 1. Summarize Data 
    summary_stats = df.describe().to_string()
    column_names = list(df.columns)
    
    # 2. Hero Section 
    st.markdown('<div class="magazine-card">', unsafe_allow_html=True)
    
    if st.button("Generate Edition"):
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
        st.markdown(f"<h1 style='font-size:2.5rem; margin-bottom: 0px; color: #ffffff; font-weight: 700;'>{content.get('HEADLINE', 'The Quarterly Signal')}</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 0; height: 1px; background: #334155; margin-top:1.5rem; margin-bottom: 2rem;'/>", unsafe_allow_html=True)
        
        st.markdown(f'<div class="editorial-text">{content.get("INSIGHTS", "Loading insights...")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. Data Visuals in Magazine Style
        st.markdown("<h3 style='margin-top: 2rem; color: #f8fafc;'>Market Visualization</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                num_cols = df.select_dtypes(include=['number']).columns
                fig = px.area(df, x=df.index, y=num_cols[0], 
                             title=f"{num_cols[0]} Trajectory",
                             template="plotly_dark",
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#cbd5e1"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualizations expect at least 2 numeric columns.")

        with col2:
            for col_name in df.select_dtypes(include=['number']).columns[:3]:
                avg = df[col_name].mean()
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;">
                    <p style="margin:0; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; font-weight: 600;">Avg {col_name}</p>
                    <h3 style="margin:0; margin-top: 0.5rem; font-size: 1.8rem; color: #ffffff;">{avg:,.2f}</h3>
                </div>
                """, unsafe_allow_html=True)

        # 4. Detailed Data Grid
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Dataset Reference Matrix"):
            st.dataframe(df, use_container_width=True)
    else:
        st.markdown("<p style='color: #64748b; font-style: italic;'>Awaiting compilation command.</p></div>", unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div class="magazine-card" style="text-align: left; padding: 3rem 2rem;">
        <h2 style="font-size: 2rem; margin-bottom: 1rem; color: #ffffff;">Workspace Idle</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; line-height: 1.6;">Initialize the engine by passing a structural CSV dataset via the configuration panel.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer for Export
st.sidebar.markdown("---")
if st.sidebar.button("Print to PDF"):
    st.sidebar.info("Use Ctrl+P (Command+P) to print this layout to a high-quality PDF.")
