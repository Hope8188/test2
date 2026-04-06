import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import os
import io

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
    prompt = f"""You are a senior data analyst writing a brief for an internal intelligence report.

Dataset snapshot:
{json.dumps(data_summary_dict, indent=2)}

Deliver a concise, fact-driven report. No metaphors, no dramatic language. Stick to what the data actually shows.

Output format (use these exact labels, no markdown):
[HEADLINE] One sharp sentence summarizing the dataset's core fact.
[INSIGHTS] Max 80 words. What the data is, what stands out numerically, what patterns exist. Be specific.
[DISCOVERY_1] One specific statistical finding.
[DISCOVERY_2] A second concrete finding.
[DISCOVERY_3] A third finding or anomaly.
[OUTLOOK] One sentence on what this data set enables or implies for further analysis.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen/qwen3.6-plus:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[HEADLINE] Generation Failed\n[INSIGHTS] Error connecting to OpenRouter: {str(e)}"


# --- Markdown Export ---
def build_markdown_report(headline, insights, discoveries, outlook, df, is_sampled):
    lines = [
        f"# {headline}",
        "",
        "## Executive Insights",
        insights,
        "",
        "## Key Discoveries",
    ]
    for d in discoveries:
        if d:
            lines.append(f"- {d}")
    lines += [
        "",
        "## Outlook",
        outlook,
        "",
        "---",
        "",
        f"**Dataset:** {len(df)} rows × {len(df.columns)} columns" + (" *(sampled)*" if is_sampled else ""),
        "",
        "### Data Preview",
        "",
        df.head(20).to_markdown(index=False)
    ]
    return "\n".join(lines)


# --- UI Layout ---
st.markdown('<div class="magazine-header">GAZETTE</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;'>The Data Brief</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Configuration")
    uploaded_file = st.file_uploader("Drop your data (.csv)", type=["csv"])

    if uploaded_file:
        df, file_size_mb, is_sampled = load_data(uploaded_file)
        st.session_state.df = df
        st.session_state.is_sampled = is_sampled
        st.success("Data uploaded successfully.")
        if is_sampled:
            st.warning(f"Large file ({file_size_mb:.1f} MB). Analyzing a {len(df)}-row sample.")
        else:
            st.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    st.markdown("---")
    st.markdown("### Export")

    # Markdown download (always available after generation)
    if 'markdown_report' in st.session_state:
        st.download_button(
            label="Download Report (.md)",
            data=st.session_state.markdown_report,
            file_name="gazette_report.md",
            mime="text/markdown",
            key="md_download_btn"
        )

    # Print hint
    if st.button("Print to PDF", key="print_pdf_btn"):
        st.sidebar.info("Press Ctrl+P / Cmd+P — the report is print-optimized.")


# --- Main Content ---
if 'df' in st.session_state:
    df = st.session_state.df
    is_sampled = st.session_state.is_sampled

    st.markdown('<div class="magazine-card">', unsafe_allow_html=True)

    if st.button("Generate Edition"):
        with st.spinner("Analyzing data..."):
            summary_dict = build_data_summary(df)
            st.session_state.editorial = get_editorial_copy(summary_dict)

    if 'editorial' in st.session_state:
        editorial = st.session_state.editorial

        # Parse tagged output
        parts = editorial.split('[')
        content = {}
        for p in parts:
            if ']' in p:
                try:
                    tag, val = p.split(']', 1)
                    content[tag.strip()] = val.strip()
                except ValueError:
                    pass

        headline = content.get('HEADLINE', 'Data Report')
        insights = content.get('INSIGHTS', '')
        discoveries = [
            content.get('DISCOVERY_1', ''),
            content.get('DISCOVERY_2', ''),
            content.get('DISCOVERY_3', ''),
        ]
        outlook = content.get('OUTLOOK', '')

        # Build and cache markdown report
        st.session_state.markdown_report = build_markdown_report(
            headline, insights, discoveries, outlook, df, is_sampled
        )

        # Layout
        st.markdown(f"<h1 style='font-size:2.2rem; margin-bottom: 0px; color: #ffffff; font-weight: 700;'>{headline}</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 0; height: 1px; background: #334155; margin-top:1.5rem; margin-bottom: 2rem;'/>", unsafe_allow_html=True)
        st.markdown(f'<div class="editorial-text">{insights}</div>', unsafe_allow_html=True)

        # Discoveries
        if any(discoveries):
            st.markdown("<h3 style='margin-top: 2rem; color: #f8fafc;'>Key Discoveries</h3>", unsafe_allow_html=True)
            for d in discoveries:
                if d:
                    st.markdown(f"<p style='color: #cbd5e1; padding: 0.6rem 0; border-left: 3px solid #3b82f6; padding-left: 1rem; margin: 0.5rem 0;'>{d}</p>", unsafe_allow_html=True)

        if outlook:
            st.markdown(f"<p style='margin-top:1.5rem; color:#64748b; font-style:italic;'>{outlook}</p>", unsafe_allow_html=True)

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
                st.plotly_chart(fig, use_container_width=True)
            else:
                cat_col = df.columns[0]
                counts = df[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, 'Count']
                fig = px.bar(counts.head(15), x=cat_col, y='Count',
                             title=f"Top Values — {cat_col}",
                             template="plotly_dark",
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#cbd5e1"))
                st.plotly_chart(fig, use_container_width=True)

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
            st.dataframe(df, use_container_width=True)
    else:
        st.markdown("<p style='color: #64748b; font-style: italic;'>Upload a CSV and click Generate Edition.</p></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="magazine-card" style="text-align: left; padding: 3rem 2rem;">
        <h2 style="font-size: 2rem; margin-bottom: 1rem; color: #ffffff;">Workspace Idle</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; line-height: 1.6;">Upload a CSV via the configuration panel to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
