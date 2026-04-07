import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import os
import io
from fpdf import FPDF

# --- Configuration & Styling ---
st.set_page_config(page_title="Gazette - Data Magazine", layout="wide")

import dotenv
dotenv.load_dotenv(".env.local")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# --- Constants ---
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SITE_URL = "http://localhost:8501"
SITE_NAME = "Gazette"
LARGE_FILE_THRESHOLD_MB = 10
SAMPLE_ROWS = 500

# --- Data Loading with Chunked Support ---
def load_data(uploaded_file):
    """Load CSV smartly - sample large files, full load for small ones."""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > LARGE_FILE_THRESHOLD_MB:
        # Chunked read: read in pieces, sample from each chunk
        chunks = []
        chunk_size = 10000
        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
            chunks.append(chunk.sample(min(SAMPLE_ROWS // 10, len(chunk)), random_state=42))
        df = pd.concat(chunks).reset_index(drop=True)
        is_sampled = True
    else:
        df = pd.read_csv(uploaded_file)
        is_sampled = False

    return df, file_size_mb, is_sampled


# --- LLM Prompt Logic ---
def build_data_summary(df):
    """Build a lean, structured summary to send to the LLM."""
    numeric_summary = df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "No numeric columns."
    cat_summaries = {}
    for col in df.select_dtypes(exclude=['number']).columns[:5]:
        top_vals = df[col].value_counts().head(5).to_dict()
        cat_summaries[col] = {"unique": df[col].nunique(), "top_5": top_vals}

    return {
        "columns": list(df.columns),
        "row_count": len(df),
        "numeric_stats": numeric_summary,
        "categorical_breakdown": cat_summaries
    }


def get_editorial_copy(data_summary_dict):
    import time

    prompt = f"""You are an expert data analyst writing for a business magazine. Analyze the dataset and output ONLY the tagged fields below.

DATASET SUMMARY:
{json.dumps(data_summary_dict, indent=2)}

STRICT OUTPUT FORMAT - Use these exact tags:
[HEADLINE] < punchy 5-8 word title capturing the dataset's main story >
[INSIGHTS] < 2-3 sentences summarizing key patterns. Mention: row count, column types, notable distributions >
[DISCOVERY_1] < specific fact with a number, e.g., "Column X has Y unique values" >
[DISCOVERY_2] < specific fact with a number, e.g., "The top category represents Z% of data" >
[DISCOVERY_3] < specific fact with a number, e.g., "Numeric column ranges from A to B" >
[OUTLOOK] < one forward-looking suggestion for how this data could be used >

RULES:
1. Start immediately with [HEADLINE], no introduction
2. Every DISCOVERY must include a specific number
3. INSIGHTS max 50 words, no fluff
4. OUTLOOK max 15 words
5. End exactly after [OUTLOOK]
6. No markdown, no bullet points, no emojis"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }

    # Fallback models in order of preference (valid free models on OpenRouter)
    models = [
        "google/gemma-2-9b-it:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b:free"
    ]

    last_error = None

    for model in models:
        for attempt in range(5):  # 5 retries per model
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = 3 + (attempt * 2)  # 3s, 5s, 7s, 9s, 11s
                    time.sleep(wait_time)
                    last_error = f"429 from {model}"
                else:
                    response.raise_for_status()
            except Exception as e:
                last_error = str(e)
                if attempt < 4:
                    time.sleep(2)
                continue

    # Try Google AI Studio as final fallback
    if GEMINI_API_KEY:
        try:
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
            gemini_payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            gemini_response = requests.post(gemini_url, json=gemini_payload, timeout=30)
            if gemini_response.status_code == 200:
                result = gemini_response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"Gemini: {gemini_response.status_code}"
        except Exception as e:
            last_error = f"Gemini error: {str(e)}"

    return f"[HEADLINE] Generation Failed\n[INSIGHTS] All models rate-limited. Last error: {last_error}"


# --- PDF Export ---
def build_pdf_report(headline, insights, discoveries, outlook, df, is_sampled):
    def clean(text):
        try:
            return str(text).encode("latin-1", "replace").decode("latin-1")
        except Exception:
            return "Content conversion error"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 20, "GAZETTE", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(59, 130, 246)
    pdf.multi_cell(180, 10, clean(headline), align="L")
    pdf.ln(5)

    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "EXECUTIVE INSIGHTS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(180, 6, clean(insights))
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "STATISTICAL DISCOVERIES", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 41, 59)
    for d in discoveries:
        if d:
            pdf.multi_cell(180, 7, f"- {clean(d)}")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "STRATEGIC OUTLOOK", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(180, 6, clean(outlook))
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(148, 163, 184)
    sampled_note = " (sampled for performance)" if is_sampled else ""
    pdf.cell(0, 6, f"Infrastructure: {len(df)} rows x {len(df.columns)} columns {sampled_note}", new_x="LMARGIN", new_y="NEXT", align="R")
    
    # fpdf2: output() returns bytes/bytearray directly.
    return pdf.output()


# --- UI Layout ---
st.markdown('<div class="magazine-header">GAZETTE</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;'>The Data Brief</p>", unsafe_allow_html=True)

# Main content area
if 'df' not in st.session_state:
    # Show upload in main area for mobile accessibility
    st.markdown('<div class="magazine-card">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 1.5rem; margin-bottom: 1rem; color: #ffffff;'>Upload Your Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 1.5rem;'>Drop a CSV file to begin analysis.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop your data (.csv)", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file:
        df, file_size_mb, is_sampled = load_data(uploaded_file)
        st.session_state.df = df
        st.session_state.is_sampled = is_sampled
        st.success(f"Data uploaded successfully. Loaded {len(df)} rows.")
        if is_sampled:
            st.warning(f"Large file ({file_size_mb:.1f} MB). Analyzing a {len(df)}-row sample.")
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    df = st.session_state.df
    is_sampled = st.session_state.is_sampled
    
    st.markdown('<div class="magazine-card">', unsafe_allow_html=True)

    if st.button("Generate Edition", key="generate_btn"):
        with st.spinner("Analyzing data..."):
            summary_dict = build_data_summary(df)
            st.session_state.editorial = get_editorial_copy(summary_dict)

    if 'editorial' in st.session_state:
        editorial = st.session_state.editorial

        # Robust parsing for tags
        parts = editorial.split('[')
        content = {}
        for p in parts:
            if ']' in p:
                try:
                    tag, val = p.split(']', 1)
                    content[tag.strip()] = val.strip()
                except ValueError:
                    pass
        
        # If no tags were found, use fallback or try a search
        headline = content.get('HEADLINE', 'Data Report')
        insights = content.get('INSIGHTS', '')
        discoveries = [
            content.get('DISCOVERY_1', ''),
            content.get('DISCOVERY_2', ''),
            content.get('DISCOVERY_3', ''),
        ]
        outlook = content.get('OUTLOOK', '')

        # If it was truly "chatty" and didn't use tags, try to scrape it
        if not insights and len(editorial) > 100:
             # Fallback: assume first paragraph is insights if tags are missing
             insights = editorial.strip()[:500] + "..." if len(editorial) > 500 else editorial.strip()

        # Cache reports
        pdf_output = build_pdf_report(headline, insights, discoveries, outlook, df, is_sampled)
        # Convert bytearray to bytes for Streamlit compatibility
        if isinstance(pdf_output, bytearray):
            pdf_output = bytes(pdf_output)
        st.session_state.pdf_report = pdf_output
        
        # Build markdown report
        md_content = f"""# {headline}

## Executive Insights
{insights}

## Statistical Discoveries
"""
        for i, d in enumerate(discoveries, 1):
            if d:
                md_content += f"{i}. {d}\n"
        
        md_content += f"""
## Strategic Outlook
{outlook}

---
*Generated by Gazette | {len(df)} rows × {len(df.columns)} columns*
"""
        st.session_state.markdown_report = md_content

        st.markdown(f"<h1 style='font-size:2.2rem; margin-bottom: 0px; color: #ffffff; font-weight: 700;'>{headline}</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 0; height: 1px; background: #334155; margin-top:1.5rem; margin-bottom: 2rem;'/>", unsafe_allow_html=True)
        st.markdown(f'<div class="editorial-text">{insights}</div>', unsafe_allow_html=True)

        if any(discoveries):
            st.markdown("<h3 style='margin-top: 2rem; color: #f8fafc;'>Key Discoveries</h3>", unsafe_allow_html=True)
            for d in discoveries:
                if d:
                    st.markdown(f"<p style='color: #cbd5e1; padding: 0.6rem 0; border-left: 3px solid #3b82f6; padding-left: 1rem; margin: 0.5rem 0;'>{d}</p>", unsafe_allow_html=True)

        if outlook:
            st.markdown(f"<p style='margin-top:1.5rem; color:#64748b; font-style:italic;'>{outlook}</p>", unsafe_allow_html=True)
        
        # Download button centered below content
        if 'pdf_report' in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="Download Report (.pdf)",
                    data=st.session_state.pdf_report,
                    file_name="gazette_report.pdf",
                    mime="application/pdf",
                    key="pdf_download_btn",
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Visualizations
        st.markdown("<h3 style='margin-top: 2rem; color: #f8fafc;'>Data Visualization</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                fig = px.area(df.head(200), x=df.head(200).index, y=numeric_cols[0],
                             title=f"{numeric_cols[0]} Distribution",
                             template="plotly_dark",
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#cbd5e1"))
                st.plotly_chart(fig, width='stretch')
            else:
                cat_col = df.columns[0]
                counts = df[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, 'Count']
                fig = px.bar(counts.head(15), x=cat_col, y='Count',
                             title=f"Top Values — {cat_col}",
                             template="plotly_dark",
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#cbd5e1"))
                st.plotly_chart(fig, width='stretch')

        with col2:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col_name in numeric_cols[:3]:
                    avg = df[col_name].mean()
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;">
                        <p style="margin:0; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Avg {col_name}</p>
                        <h3 style="margin:0; margin-top: 0.4rem; font-size: 1.6rem; color: #ffffff;">{avg:,.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                for col_name in df.columns[:3]:
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #3b82f6;">
                        <p style="margin:0; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; font-weight: 600;">Unique {col_name}</p>
                        <h3 style="margin:0; margin-top: 0.4rem; font-size: 1.6rem; color: #ffffff;">{df[col_name].nunique()}</h3>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Dataset View"):
            st.dataframe(df, width='stretch')
    else:
        st.markdown("<p style='color: #64748b; font-style: italic;'>Click 'Generate Edition' to analyze your data.</p></div>", unsafe_allow_html=True)
