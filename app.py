import os
import re
import duckdb
import chromadb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM

# ─── PATHS ───────────────────────────────────────────────────────
import os

BASE = os.getcwd()   # current project folder (works in cloud)

CHROMA_PATH = os.path.join(BASE, "data", "chroma_db")
DB_PATH = os.path.join(BASE, "data", "bri_research.db")
ENV_PATH = os.path.join(BASE, ".env")
# ─── PAGE CONFIG ─────────────────────────────────────────────────
def ensure_chromadb():
    chroma_local = os.path.join(BASE, "data", "chroma_db")
    if os.path.exists(os.path.join(chroma_local, "chroma.sqlite3")):
        return  # already present, skip download
    st.info("First run: downloading document index (~5GB). This takes 5–10 minutes and only happens once.", icon="⏳")
    try:
        from huggingface_hub import snapshot_download
        import shutil
        token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
        downloaded = snapshot_download(
            repo_id="sanoop-ssk/bri-monitor-chromadb",
            repo_type="dataset",
            local_dir=chroma_local,
            token=token
        )
        st.success("Document index ready.", icon="✅")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to download document index: {e}")
        st.stop()

ensure_chromadb()
st.set_page_config(
    page_title="BRI Finance Research Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── SESSION STATE INIT ──────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─── THEME COLORS ────────────────────────────────────────────────
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg": "#0E1117", "card": "#1E2530", "text": "#FAFAFA",
            "muted": "#8B9BB4", "border": "#2E3A4E", "accent": "#4A90D9",
            "navy": "#2E6DA4", "light_bg": "#161C27"
        }
    else:
        return {
            "bg": "#FFFFFF", "card": "#F8FAFD", "text": "#1A1A2E",
            "muted": "#6B7280", "border": "#E2E8F0", "accent": "#2E6DA4",
            "navy": "#1B3A6B", "light_bg": "#F0F6FF"
        }

T = get_theme()

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {T['bg']}; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    
    .hero-section {{
        background: linear-gradient(135deg, #1B3A6B 0%, #2E6DA4 60%, #4A90D9 100%);
        padding: 3.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }}
    .hero-section::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1.2;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #A8D4F5;
        margin: 0.8rem 0 0 0;
        max-width: 620px;
        line-height: 1.6;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }}
    .stDeployButton {{ display: none; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stToolbar"] {{ display: none; }}
    .stat-card {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-left: 4px solid {T['accent']};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }}
    .stat-number {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {T['navy']};
        margin: 0;
    }}
    .stat-label {{
        font-size: 0.82rem;
        color: {T['muted']};
        margin: 0.2rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .stat-sub {{
        font-size: 0.75rem;
        color: {T['accent']};
        margin: 0.1rem 0 0 0;
    }}
    
    .section-card {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }}
    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {T['navy']};
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {T['accent']};
    }}
    
    .cta-button {{
        display: inline-block;
        background: white;
        color: #1B3A6B;
        padding: 0.85rem 2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        text-decoration: none;
        margin-top: 1.5rem;
        cursor: pointer;
        border: none;
        transition: all 0.2s;
    }}
    
    .chat-header {{
        background: linear-gradient(135deg, #1B3A6B 0%, #2E6DA4 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .chat-title {{
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
    }}
    .chat-sub {{
        color: #A8D4F5;
        font-size: 0.8rem;
        margin: 0.2rem 0 0 0;
    }}
    
    .sample-query-btn {{
        background: {T['light_bg']};
        border: 1px solid {T['border']};
        border-radius: 8px;
        padding: 0.7rem 1rem;
        text-align: left;
        cursor: pointer;
        width: 100%;
        color: {T['text']};
        font-size: 0.85rem;
        transition: all 0.2s;
    }}
    
    .response-container {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }}
    
    .query-badge {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    .badge-doc {{ background: #E8F5E9; color: #2E7D32; border: 1px solid #81C784; }}
    .badge-data {{ background: #FFF3E0; color: #E65100; border: 1px solid #FFB74D; }}
    .badge-hybrid {{ background: #EDE7F6; color: #4527A0; border: 1px solid #9575CD; }}
    
    .source-note {{
        font-size: 0.78rem;
        color: {T['muted']};
        border-top: 1px solid {T['border']};
        padding-top: 0.6rem;
        margin-top: 0.8rem;
        font-style: italic;
    }}
    
    .followup-section {{
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid {T['border']};
    }}
    .followup-label {{
        font-size: 0.78rem;
        color: {T['muted']};
        margin-bottom: 0.4rem;
    }}
    
    .limitation-notice {{
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        padding: 0.6rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #92400E;
        margin-top: 1rem;
    }}

    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ─── INITIALIZE RESOURCES ────────────────────────────────────────
@st.cache_resource
def initialize():
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAILLM(model="gpt-4o-mini", api_key=api_key, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection("bri_documents")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    con = duckdb.connect(DB_PATH, read_only=True)
    client = OpenAI(api_key=api_key)
    return index, con, client

# ─── SYSTEM PROMPT ───────────────────────────────────────────────
BRI_SYSTEM_PROMPT = """
You are a specialized AI research assistant for the study of Chinese overseas 
infrastructure finance and the Belt and Road Initiative (BRI). Built by a PhD 
researcher in International Relations specializing in Chinese infrastructure 
finance, South Asian political economy, and Indo-Pacific dynamics.

KNOWLEDGE SOURCES:
1. Document library — Chinese government white papers, official speeches, 
   academic literature, World Bank/IMF/UN reports, think tank research
2. AidData dataset — 4,861 Chinese infrastructure projects, 183 countries, 
   2000-2023, approximately $1.2 trillion in commitments

SCOPE: All Chinese official overseas infrastructure finance — not only 
formally designated BRI projects. BRI designation is contested; AidData's 
broader coverage is more analytically defensible.

RESPONSE STYLE — ADAPTIVE, NOT FORMULAIC:
- Match your response structure to the question complexity
- Simple factual queries: answer directly and concisely in 2-3 paragraphs
- Complex analytical queries: use light headers to organize naturally
- Only use the full four-part structure for genuinely multi-dimensional 
  hybrid questions requiring both data and document synthesis
- Write like a knowledgeable research colleague, not a report generator
- Avoid repeating the same section headers in every response

STRICT RULES:
1. Every interpretive claim must be attributed to a named source
2. Never generate author-year citations not in retrieved passages  
3. Distinguish Chinese official framing from independent scholarship
4. Flag contested empirical claims as contested
5. Always note AidData figures = commitments, not disbursements
6. No predictive forecasting — offer trend interpretation instead
7. Never presuppose strategic intent without explicit source support

When you do use structure for complex hybrid responses, use these sections:
**What the data shows** / **Official framing** / 
**Scholarly perspectives** / **Open questions**
"""

# ─── AGENT FUNCTIONS ─────────────────────────────────────────────
def query_documents(question, index, client):
    retriever = index.as_retriever(similarity_top_k=8)
    nodes = retriever.retrieve(question)
    context_parts = []
    sources = []
    for i, node in enumerate(nodes):
        filename = node.metadata.get('file_name', 'Unknown')
        # Clean filename for display
        clean_name = filename.replace('.pdf', '').replace('_', ' ').replace(' _ english.scio.gov.cn', '')
        text = node.node.text[:800]
        # Label by name, not number
        context_parts.append(f"[Source: {clean_name}]\n{text}")
        if filename not in sources:
            sources.append(filename)
    context = "\n\n".join(context_parts)
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1,
        messages=[
            {"role": "system", "content": BRI_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
Answer this research question using ONLY the provided source passages.
When citing sources, use the document name provided in brackets, not numbers.
Never generate citations not present in the sources.
Do not use markdown code formatting or backticks in your response.

QUESTION: {question}
SOURCE PASSAGES:
{context}
"""}
        ]
    )
    return {"answer": response.choices[0].message.content,
            "sources": sources, "type": "document"}


def query_data(question, con, client):
    sql_prompt = f"""
You are a SQL expert for a Chinese infrastructure finance database.
Convert the research question into a DuckDB SQL query.

TABLES:
1. bri_projects_core — developing countries (4,498 rows)
2. bri_projects_south_asia — South Asia subset (463 rows)
3. bri_projects_full — all countries (4,861 rows)

KEY COLUMNS: Country_of_Activity, Region_of_Activity, Commitment_Year,
Completion_Year, Title, Flow_Type, Sector_Name, Funding_Agencies_Parent,
Amount_Nominal_USD, Amount_Constant_USD_2023, Collateralized,
Level_of_Public_Liability, WB_Income_Group_Host_Country, Tranche_Count

EXACT REGION VALUES: 'Africa', 'America', 'Asia', 'Europe',
'Middle East', 'Oceania', 'Multi-Region'
USE 'America' for Latin America queries — not 'Latin America'

EXACT SECTOR VALUES (uppercase only, never use LIKE):
'ENERGY', 'TRANSPORT AND STORAGE', 'INDUSTRY, MINING, CONSTRUCTION',
'EDUCATION', 'COMMUNICATIONS', 'HEALTH', 'WATER SUPPLY AND SANITATION',
'GOVERNMENT AND CIVIL SOCIETY', 'AGRICULTURE, FORESTRY, FISHING',
'OTHER SOCIAL INFRASTRUCTURE AND SERVICES', 'OTHER MULTISECTOR'

FLOW TYPES: 'Loan', 'Grant', 'Vague TBD'

TOP FUNDING AGENCIES:
'Export-Import Bank of China (China Eximbank)'
'PRC Central Government', 'China Development Bank (CDB)'
'Industrial and Commercial Bank of China (ICBC)', 'Bank of China (BOC)'

RULES:
- Amount_Constant_USD_2023 for comparisons
- ROUND(x/1e9, 2) for billions
- WHERE Amount_Nominal_USD IS NOT NULL for financial queries
- Never filter by Funding_Agencies_Parent unless specifically asked
- Limit 15 rows max unless asked otherwise
- South Asia queries → bri_projects_south_asia
- Global queries → bri_projects_core

QUESTION: {question}
Return ONLY the SQL query, no explanation, no markdown.
"""
    sql_response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": sql_prompt}]
    )
    sql_query = sql_response.choices[0].message.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    try:
        result_df = con.execute(sql_query).df()
        interpretation = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.1,
            messages=[
                {"role": "system", "content": BRI_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
Interpret these AidData results concisely and clearly.
Figures are commitments not disbursements.
Source: AidData Global Chinese Development Finance Dataset (2000-2023).
Match response length to complexity — avoid padding.

QUESTION: {question}
RESULTS:
{result_df.to_string(index=False)}
"""}
            ]
        )
        return {"answer": interpretation.choices[0].message.content,
                "sql": sql_query, "data": result_df, "type": "data"}
    except Exception as e:
        return {"answer": f"Data query error: {str(e)}",
                "sql": sql_query, "data": None, "type": "data_error"}


def bri_agent(question, index, con, client):
    routing_prompt = f"""
Classify this research question:
- "documents" — qualitative analysis from papers/policy docs
- "data" — quantitative analysis from structured dataset
- "both" — needs both sources combined

QUESTION: {question}
Reply with ONE word: documents, data, or both
"""
    routing = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": routing_prompt}]
    )
    route = routing.choices[0].message.content.strip().lower()
    if route not in ["documents", "data", "both"]:
        route = "both"

    if route == "documents":
        result = query_documents(question, index, client)
    elif route == "data":
        result = query_data(question, con, client)
    else:
        doc_result = query_documents(question, index, client)
        data_result = query_data(question, con, client)
        combined = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.1,
            messages=[
                {"role": "system", "content": BRI_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
Synthesise these findings into a coherent scholarly response.
Adapt the structure to the complexity — use headers only when needed.
Connect the quantitative evidence with the qualitative analysis naturally.

QUESTION: {question}
DOCUMENT ANALYSIS: {doc_result['answer']}
DATA ANALYSIS: {data_result['answer']}
"""}
            ]
        )
        result = {
            "answer": combined.choices[0].message.content,
            "sources": doc_result.get("sources", []),
            "sql": data_result.get("sql", ""),
            "data": data_result.get("data", None),
            "type": "hybrid"
        }

    # Generate follow-up suggestions
    followup_prompt = f"""
You are helping a researcher studying Chinese overseas infrastructure finance and BRI.
Based on their question, suggest 3 specific follow-up research questions.

Rules:
- Each question must be answerable using either AidData financing data OR BRI policy documents
- Keep each under 12 words
- Make them specific and research-relevant, not generic
- Vary the type: one data query, one document query, one comparative
- Return as a Python list of 3 strings ONLY — no other text

Question asked: {question}
"""
    try:
        followup_resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.7,
            messages=[{"role": "user", "content": followup_prompt}]
        )
        import ast
        followups_text = followup_resp.choices[0].message.content.strip()
        followups_text = followups_text.replace("```python", "").replace("```", "").strip()
        followups = ast.literal_eval(followups_text)
        if not isinstance(followups, list):
            followups = []
    except:
        followups = []

    result["route"] = route
    result["followups"] = followups
    return result


# ─── CHART HELPER ────────────────────────────────────────────────
def should_show_chart(result):
    """Only show chart when data is meaningful and adds value."""
    if result.get("data") is None:
        return False
    df = result["data"]
    if df.empty or len(df) < 2:
        return False
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    return len(num_cols) > 0 and len(cat_cols) > 0


def create_chart(df):
    if df is None or df.empty or len(df) < 2:
        return None
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not num_cols:
        return None

    # Find best numeric column (prefer financing/amount/billion columns)
    priority_terms = ['billion', 'usd', 'financing', 'investment', 'amount', 'total']
    num_col = num_cols[0]
    for col in num_cols:
        if any(t in col.lower() for t in priority_terms):
            num_col = col
            break

    # Find best categorical column (prefer country/region, avoid year for bar charts)
    cat_col = None
    year_col = None
    
    for col in df.columns:
        if 'year' in col.lower() or 'Year' in col:
            year_col = col
        elif col in cat_cols and cat_col is None:
            cat_col = col

    unit_label = num_col.replace('_', ' ')
    if any(x in num_col.lower() for x in priority_terms):
        unit_label = f"{unit_label} (USD Bn)"

    is_dark = st.session_state.dark_mode
    bg = 'rgba(0,0,0,0)'
    fc = '#FFFFFF' if is_dark else '#1A1A2E'
    gc = 'rgba(255,255,255,0.1)' if is_dark else 'rgba(0,0,0,0.08)'
    
    colors = ["#1B3A6B","#2E6DA4","#4A90D9","#6BAED6","#9ECAE1",
              "#2166AC","#4393C3","#74ADD1","#1D6996","#0570B0"]

    df_chart = df.head(12).copy()

    try:
        # Time series: use year column on x-axis
        if year_col and len(df[year_col].unique()) > 3:
            fig = px.line(
                df_chart, x=year_col, y=num_col,
                markers=True, color_discrete_sequence=["#2E6DA4"],
                labels={num_col: unit_label, year_col: 'Year'}
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))

        # Categorical with one numeric: horizontal bar
        elif cat_col:
            df_chart = df_chart.sort_values(num_col, ascending=True)
            fig = px.bar(
                df_chart, x=num_col, y=cat_col,
                orientation='h', color=cat_col,
                color_discrete_sequence=colors,
                labels={num_col: unit_label, cat_col: ""}
            )
            fig.update_layout(showlegend=False)
        else:
            return None

        fig.update_layout(
            plot_bgcolor=bg, paper_bgcolor=bg,
            font=dict(family="Arial", size=12, color=fc),
            margin=dict(l=20, r=120, t=40, b=20),
            height=max(300, len(df_chart) * 36),
            yaxis=dict(tickfont=dict(size=11), gridcolor=gc),
            xaxis=dict(tickfont=dict(size=11), gridcolor=gc),
            annotations=[dict(
                text="BRI Finance Research Assistant",
                x=0.99, y=0.01, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=9, color="rgba(100,100,100,0.4)"),
                xanchor="right", yanchor="bottom"
            )]
        )
        return fig
    except:
        return None


# ─── HOMEPAGE ────────────────────────────────────────────────────
def show_homepage():
    index, con, client = initialize()

    # Dark mode toggle
    col_title, col_toggle = st.columns([10, 1])
    with col_toggle:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️",
                     help="Toggle dark/light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">🔬 PhD Research Tool · Beta Version</div>
        <h1 class="hero-title">BRI Infrastructure Finance<br>Research Assistant</h1>
        <p class="hero-subtitle">
            An AI-powered research platform integrating China's overseas infrastructure 
            finance data with policy documents, academic literature, and official sources. 
            Built for rigorous, evidence-grounded analysis of Chinese development finance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("4,861", "Infrastructure Projects", "AidData 2000–2023"),
        ("$1.2 Trillion", "Total Commitments", "Nominal USD"),
        ("183", "Countries Covered", "Global scope"),
        ("42", "Curated Documents", "PDFs indexed"),
    ]
    for col, (num, label, sub) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{num}</p>
                <p class="stat-label">{label}</p>
                <p class="stat-sub">{sub}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview chart
    try:
        trend_data = con.execute("""
            SELECT Commitment_Year as Year,
                   COUNT(*) as Projects,
                   ROUND(SUM(Amount_Constant_USD_2023)/1e9, 1) as Financing_Billions
            FROM bri_projects_core
            WHERE Commitment_Year >= 2000
            AND Amount_Nominal_USD IS NOT NULL
            GROUP BY Commitment_Year
            ORDER BY Commitment_Year
        """).df()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trend_data['Year'], y=trend_data['Financing_Billions'],
            name='Financing (USD Bn)', marker_color='#2E6DA4',
            opacity=0.8, yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=trend_data['Year'], y=trend_data['Projects'],
            name='Projects', line=dict(color='#F59E0B', width=2.5),
            mode='lines+markers', marker=dict(size=5), yaxis='y2'
        ))
        fig.update_layout(
            title=dict(
                text="Chinese Infrastructure Finance Commitments (2000–2023)",
                font=dict(size=14, color=T['navy'])
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=11, color=T['text']),
            yaxis=dict(title="Financing (USD Billions)",
                      gridcolor=T['border'], titlefont=dict(color='#2E6DA4')),
            yaxis2=dict(title="Number of Projects", overlaying='y',
                       side='right', titlefont=dict(color='#F59E0B')),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
            margin=dict(l=20, r=20, t=60, b=20),
            height=380,
            annotations=[dict(
                text="Source: AidData Global Chinese Development Finance Dataset v1.0",
                x=0.99, y=-0.08, xref="paper", yref="paper",
                showarrow=False, font=dict(size=9, color=T['muted']),
                xanchor="right"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

    # Two column info section
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"""
        <div class="section-card">
            <p class="section-title">📊 Data Sources</p>
            <p style="color:{T['text']}; font-size:0.9rem; margin-bottom:0.8rem;">
                <strong>Primary dataset:</strong> AidData Global Chinese Development Finance Dataset v1.0 — 
                the most comprehensive public record of Chinese official overseas finance, covering 
                33,580 loans and grants across 217 countries (2000–2023).
            </p>
            <p style="color:{T['text']}; font-size:0.9rem; margin-bottom:0.8rem;">
                The tool uses an infrastructure-filtered subset of 4,861 project-level records 
                aggregated from tranche-level entries, representing approximately $1.2 trillion 
                in financing commitments.
            </p>
            <p style="color:{T['muted']}; font-size:0.82rem;">
                ⚠️ All figures represent <em>financing commitments</em>, not disbursements.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="section-card">
            <p class="section-title">📚 Document Library</p>
            <p style="color:{T['text']}; font-size:0.9rem; margin-bottom:0.5rem;">
                42 curated documents across five categories:
            </p>
            <ul style="color:{T['text']}; font-size:0.88rem; margin:0; padding-left:1.2rem;">
                <li>Chinese government white papers and policy documents</li>
                <li>Xi Jinping BRI Forum speeches (2017, 2019, 2023)</li>
                <li>Peer-reviewed academic literature</li>
                <li>World Bank, IMF, and UN institutional reports</li>
                <li>Independent research and think tank analysis</li>
            </ul>
            <p style="color:{T['muted']}; font-size:0.82rem; margin-top:0.8rem;">
                Documents are semantically indexed for research-grade retrieval.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Capabilities section
    st.markdown(f"""
    <div class="section-card" style="margin-bottom:1.5rem;">
        <p class="section-title">🔍 What You Can Ask</p>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem;">
            <div>
                <p style="color:{T['navy']}; font-weight:700; font-size:0.9rem; margin:0 0 0.4rem 0;">
                    📊 Data Queries
                </p>
                <p style="color:{T['text']}; font-size:0.85rem; margin:0;">
                    Country comparisons, sector analysis, financing trends, 
                    regional breakdowns, debt exposure patterns
                </p>
            </div>
            <div>
                <p style="color:{T['navy']}; font-weight:700; font-size:0.9rem; margin:0 0 0.4rem 0;">
                    📄 Policy Discourse
                </p>
                <p style="color:{T['text']}; font-size:0.85rem; margin:0;">
                    Chinese official framing, white paper analysis, 
                    speech content, policy evolution over time
                </p>
            </div>
            <div>
                <p style="color:{T['navy']}; font-weight:700; font-size:0.9rem; margin:0 0 0.4rem 0;">
                    🔗 Hybrid Analysis
                </p>
                <p style="color:{T['text']}; font-size:0.85rem; margin:0;">
                    Connect financing data with policy narratives, 
                    compare official claims with empirical evidence
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    col_cta1, col_cta2, col_cta3 = st.columns([2, 2, 2])
    with col_cta2:
        if st.button("🔬 Launch Research Assistant →",
                     use_container_width=True,
                     type="primary"):
            st.session_state.page = "chat"
            st.rerun()

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; color:{T['muted']}; font-size:0.8rem; 
                padding-top:1rem; border-top:1px solid {T['border']};">
        Built by a PhD researcher in International Relations · 
        Data: AidData GCDF v1.0 · AI: OpenAI GPT-4o-mini · 
        Beta Version — outputs should be verified against primary sources
    </div>
    """, unsafe_allow_html=True)


# ─── CHAT PAGE ───────────────────────────────────────────────────
def show_chat():
    index, con, client = initialize()

    # Header row
    col_back, col_title, col_toggle = st.columns([1, 8, 1])
    with col_back:
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()
    with col_title:
        st.markdown("""
        <div class="chat-header">
            <div>
                <p class="chat-title">🌐 BRI Infrastructure Finance Research Assistant</p>
                <p class="chat-sub">Ask questions about Chinese overseas infrastructure finance · Data + Documents + AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_toggle:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # Opening screen — sample queries if no messages yet
    if not st.session_state.messages:
        st.markdown(f"""
        <p style="text-align:center; color:{T['muted']}; font-size:0.9rem; 
                  margin:1.5rem 0 1rem 0;">
            Start with a sample question or type your own below
        </p>
        """, unsafe_allow_html=True)

        sample_queries = [
            "Which South Asian countries received the most Chinese energy financing after 2013?",
            "How has Chinese infrastructure financing changed before and after 2017?",
            "What does China's Guiding Principles document say about BRI debt sustainability?",
            "Compare Chinese transport financing in Africa vs Latin America"
        ]

        col1, col2 = st.columns(2)
        for i, query in enumerate(sample_queries):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"💬 {query}", key=f"sample_{i}",
                             use_container_width=True):
                    st.session_state.pending_query = query
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                route = message.get("route", "")
                badge_class = {"documents": "badge-doc",
                               "data": "badge-data",
                               "both": "badge-hybrid"}.get(route, "badge-doc")
                badge_label = {"documents": "📄 Document Query",
                               "data": "📊 Data Query",
                               "both": "🔗 Hybrid Query"}.get(route, "")
                if badge_label:
                    st.markdown(f'<span class="query-badge {badge_class}">{badge_label}</span>',
                               unsafe_allow_html=True)

                clean_answer = message["content"].replace("```", "").replace("`", "'")
                st.markdown(clean_answer)

                # Conditional data display
                if message.get("data") is not None and not message["data"].empty:
                    with st.expander("📊 Data Table & Chart", expanded=True):
                        df = message["data"]
                        # Add unit to column names
                        rename_map = {}
                        for col in df.columns:
                            if any(x in col.lower() for x in ['billion', 'usd', 'financing', 'investment', 'amount']):
                                rename_map[col] = f"{col} (USD Bn)"
                        st.dataframe(df.rename(columns=rename_map),
                                    use_container_width=True)
                        fig = create_chart(df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                # Source note — clean, not raw filenames
                if message.get("sources"):
                    source_count = len(message["sources"])
                    st.markdown(f"""
                    <div class="source-note">
                        📚 Response draws on {source_count} indexed source(s) 
                        from the document library including policy documents, 
                        academic literature, and institutional reports.
                    </div>
                    """, unsafe_allow_html=True)

                # Limitation notice
                st.markdown("""
                <div class="limitation-notice">
                    ⚠️ Financing figures represent commitments, not disbursements. 
                    Verify key findings against primary sources before publication.
                </div>
                """, unsafe_allow_html=True)

                # Follow-up suggestions
                if message.get("followups"):
                    st.markdown(f"""
                    <div class="followup-section">
                        <p class="followup-label">💡 You might also ask:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    fu_cols = st.columns(len(message["followups"]))
                    for j, (fcol, fq) in enumerate(zip(fu_cols, message["followups"])):
                        with fcol:
                            if st.button(fq, key=f"fu_{id(message)}_{j}",
                                        use_container_width=True):
                                st.session_state.pending_query = fq
                                st.rerun()
            else:
                st.markdown(message["content"])

    # Process pending query
    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and data..."):
                result = bri_agent(prompt, index, con, client)

            route = result.get("route", "")
            badge_class = {"documents": "badge-doc", "data": "badge-data",
                          "both": "badge-hybrid"}.get(route, "badge-doc")
            badge_label = {"documents": "📄 Document Query",
                          "data": "📊 Data Query",
                          "both": "🔗 Hybrid Query"}.get(route, "")
            if badge_label:
                st.markdown(f'<span class="query-badge {badge_class}">{badge_label}</span>',
                           unsafe_allow_html=True)

            clean_answer = message["content"].replace("```", "").replace("`", "'")
            st.markdown(clean_answer)

            if result.get("data") is not None and not result["data"].empty:
                with st.expander("📊 Data Table & Chart", expanded=True):
                    df = result["data"]
                    rename_map = {}
                    for col in df.columns:
                        if any(x in col.lower() for x in ['billion', 'usd', 'financing', 'investment', 'amount']):
                            rename_map[col] = f"{col} (USD Bn)"
                    st.dataframe(df.rename(columns=rename_map),
                                use_container_width=True)
                    fig = create_chart(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            if result.get("sources"):
                source_count = len(result["sources"])
                st.markdown(f"""
                <div class="source-note">
                    📚 Response draws on {source_count} indexed source(s) 
                    from the document library.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="limitation-notice">
                ⚠️ Financing figures represent commitments, not disbursements. 
                Verify key findings against primary sources before publication.
            </div>
            """, unsafe_allow_html=True)

            if result.get("followups"):
                st.markdown(f"""
                <div class="followup-section">
                    <p class="followup-label">💡 You might also ask:</p>
                </div>
                """, unsafe_allow_html=True)
                fu_cols = st.columns(len(result["followups"]))
                for j, (fcol, fq) in enumerate(zip(fu_cols, result["followups"])):
                    with fcol:
                        if st.button(fq, key=f"new_fu_{j}",
                                    use_container_width=True):
                            st.session_state.pending_query = fq
                            st.rerun()

        msg = {
            "role": "assistant",
            "content": result["answer"],
            "route": route,
            "sources": result.get("sources", []),
            "data": result.get("data"),
            "followups": result.get("followups", [])
        }
        st.session_state.messages.append(msg)
        st.rerun()

    # Chat input
    if prompt := st.chat_input(
        "Ask a research question about Chinese infrastructure finance..."
    ):
        st.session_state.pending_query = prompt
        st.rerun()


# ─── ROUTER ──────────────────────────────────────────────────────
if st.session_state.page == "home":
    show_homepage()
else:
    show_chat()