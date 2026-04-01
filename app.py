"""
The BRI Monitor — Chinese Development Finance Research Platform
app.py · 6-page single-file application

Pages: Home · Chat Assistant · Data Explorer · Documents · About · Contact

Setup:
  1. Copy to C:/Users/sanoo/bri-research-assistant/app.py
  2. Ensure .env contains OPENAI_API_KEY
  3. Run: streamlit run app.py
"""

import os, re, ast
import requests as http_req
import duckdb, chromadb
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM

# ── Configuration ─────────────────────────────────────────────────
load_dotenv()
BASE        = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE, "data", "chroma_db"))
DB_PATH     = os.getenv("DB_PATH",     os.path.join(BASE, "data", "bri_research.db"))
load_dotenv(dotenv_path=os.path.join(BASE, ".env"))

FORMSPREE = "https://formspree.io/f/xreoqrdl"

# ── Auto-download chroma_db from Hugging Face on first cloud run ──
def ensure_chromadb():
    chroma_local = os.path.join(BASE, "data", "chroma_db")
    if os.path.exists(os.path.join(chroma_local, "chroma.sqlite3")):
        return  # already present locally, skip download
    st.info("First run: downloading document index. This takes 5-10 minutes and only happens once.")
    try:
        from huggingface_hub import snapshot_download
        token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
        snapshot_download(
            repo_id="sanoop-ssk/bri-monitor-chromadb",
            repo_type="dataset",
            local_dir=chroma_local,
            token=token
        )
        st.success("Document index ready.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to download document index: {e}")
        st.stop()

st.set_page_config(
    page_title="The BRI Monitor",
    page_icon="🌐", layout="wide",
    initial_sidebar_state="collapsed"
)

ensure_chromadb()

for k, v in [("page","home"),("messages",[]),("dark_mode",True),("pending_query",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Flat colour palette ───────────────────────────────────────────
L = {"bg":"#EEF2F7","surface":"#FFFFFF","navy":"#1B3A6B","blue":"#1D5FA8",
     "text":"#0A1628","muted":"#3D4F66","border":"#B8C8DC","accent":"#1D5FA8",
     "tag":"#D4E5F7"}
D = {"bg":"#0D1117","surface":"#161B22","navy":"#58A6FF","blue":"#388BFD",
     "text":"#E6EDF3","muted":"#8B949E","border":"#30363D","accent":"#388BFD",
     "tag":"#1C2A3A"}

def T(): 
    st.session_state.dark_mode = True  # dark mode locked
    return D

CHART_PAL = ["#1B3A6B","#2E6DA4","#4A90D9","#6BAED6","#9ECAE1",
             "#2166AC","#4393C3","#74ADD1","#1D6996","#0570B0"]


# ── CSS ──────────────────────────────────────────────────────────
def inject_css():
    t = T()
    dark = st.session_state.dark_mode
    # Sidebar bg must match app bg
    sidebar_bg = t['bg'] if not dark else "#0D1117"
    st.markdown(f"""<style>
/* ── Base ── */
.stApp{{background:{t['bg']} !important}}
.block-container{{padding-top:.3rem !important;padding-bottom:1rem;max-width:1400px}}
.stDeployButton,#MainMenu,footer,[data-testid="stToolbar"]
  {{display:none !important;visibility:hidden !important}}
header[data-testid="stHeader"]{{background:transparent !important;height:0}}

/* ── GLOBAL body text — explicit selectors for light mode readability ── */
p{{color:{t['text']}}}
li{{color:{t['text']}}}
label{{color:{t['text']}}}
td, th{{color:{t['text']}}}
/* Streamlit markdown containers */
.stMarkdown p,.stMarkdown li,.stMarkdown div,.stMarkdown span
  {{color:{t['text']} !important}}
.stMarkdown strong,.stMarkdown em
  {{color:{t['text']} !important}}
/* Streamlit write / text elements */
[data-testid="stMarkdownContainer"] *:not([style*="background"]):not([style*="#1B3A6B"])
  {{color:{t['text']} !important}}
/* Chat messages */
.stChatMessage p,.stChatMessage li,.stChatMessage span,
.stChatMessage strong,.stChatMessage em
  {{color:{t['text']} !important}}
/* Headings outside dark containers */
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4
  {{color:{t['navy']} !important}}
/* Form labels and widget labels */
.stTextInput label,.stTextArea label,.stSelectbox label,.stMultiSelect label,
.stSlider label,.stRadio label,.stCheckbox label,[data-testid="stWidgetLabel"] p
  {{color:{t['text']} !important;font-weight:600 !important}}
/* Native text & caption */
[data-testid="stText"],[data-testid="stMarkdownContainer"] p
  {{color:{t['text']} !important}}
/* Tab bar */
button[data-baseweb="tab"]
  {{color:{t['muted']} !important}}
button[data-baseweb="tab"][aria-selected="true"]
  {{color:{t['navy']} !important;font-weight:700 !important}}
/* Dataframe cells */
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th,
[data-testid="StyledFullScreenFrame"] td,[data-testid="StyledFullScreenFrame"] th
  {{color:{t['text']} !important}}
/* Expander */
[data-testid="stExpander"] summary p,[data-testid="stExpander"] div p
  {{color:{t['text']} !important}}
/* Select box and multiselect options */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="popover"] li
  {{color:{t['text']} !important}}
/* Metric values */
[data-testid="stMetricValue"],[data-testid="stMetricLabel"]
  {{color:{t['navy']} !important}}
/* Alert boxes */
[data-testid="stAlert"] p,[data-testid="stAlert"] div
  {{color:{t['text']} !important}}
/* st.info, st.success, st.warning text */
div[data-testid="stNotification"] p
  {{color:{t['text']} !important}}
/* Radio button labels */
[data-testid="stRadio"] label,[data-testid="stRadio"] p
  {{color:{t['text']} !important}}
/* Slider tick marks */
[data-testid="stSlider"] p
  {{color:{t['muted']} !important}}
/* Caption text */
[data-testid="stCaptionContainer"] p
  {{color:{t['muted']} !important}}
/* Chat input placeholder */
[data-testid="stChatInput"] textarea
  {{color:{t['text']} !important;background:{t['surface']} !important}}
/* Sidebar */
section[data-testid="stSidebar"]
  {{background:{sidebar_bg} !important}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div
  {{color:{t['text']} !important}}

/* ── Sticky navbar: top-level only — avoids inner blocks ── */
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > div:first-child {{
  position:sticky !important;
  top:0 !important;
  z-index:9999 !important;
  background:{t['bg']} !important;
  box-shadow:0 2px 10px rgba(0,0,0,.22) !important;
}}

/* ── ALL primary buttons: teal action colour ── */
button[kind="primary"] {{
  background:#1A7A8A !important;
  border-color:#1A7A8A !important;
  color:#FFFFFF !important;
  font-weight:700 !important;
}}
button[kind="primary"]:hover {{background:#155F6D !important}}

/* ── Responsive: allow natural wrapping on narrow screens ── */
@media (max-width:768px) {{
  .block-container{{padding-left:.5rem !important;padding-right:.5rem !important}}
  .mr{{flex-direction:column}}
}}

/* ── Multiselect tags — navy palette ── */
span[data-baseweb="tag"]
  {{background-color:{t['tag']} !important;border:1px solid {t['border']} !important}}
span[data-baseweb="tag"] span
  {{color:{t['navy']} !important;font-weight:600}}
span[data-baseweb="tag"] button,
span[data-baseweb="tag"] [role="presentation"]
  {{color:{t['navy']} !important}}

/* ── Hero containers — dark-bg divs ALWAYS show white text ── */
/* Force white text on ALL dark hero containers regardless of theme */
div[style*="background:#0D1B2E"],
div[style*="background:#0D1B2E"] p,
div[style*="background:#0D1B2E"] div,
div[style*="background:#0D1B2E"] span
  {{color:#FFFFFF !important}}
div[style*="background:#1B3A6B"],
div[style*="background:#1B3A6B"] p,
div[style*="background:#1B3A6B"] div,
div[style*="background:#1B3A6B"] span
  {{color:#FFFFFF !important}}

/* ── Stat cards ── */
.stat-card{{background:{t['surface']};border:1px solid {t['border']};
  border-left:3px solid {t['blue']};border-radius:8px;padding:.85rem 1rem;
  text-align:center;transition:transform .15s}}
.sn{{color:{t['navy']} !important;font-size:1.5rem;font-weight:800;margin:0}}
.sl{{color:{t['muted']} !important;font-size:.71rem;text-transform:uppercase;
     letter-spacing:.05em;margin:.15rem 0 0}}
.ss{{color:{t['blue']} !important;font-size:.68rem;margin:.05rem 0 0}}

/* ── Info cards ── */
.ic{{background:{t['surface']};border:1px solid {t['border']};
     border-radius:8px;padding:1.1rem;height:100%}}
.ic .ct{{color:{t['navy']} !important;font-size:.86rem;font-weight:700;
         margin:0 0 .6rem;padding-bottom:.3rem;border-bottom:2px solid {t['blue']}}}
.ic p,.ic li{{color:{t['text']} !important;font-size:.83rem;line-height:1.6}}

/* ── Section heading ── */
.sec{{color:{t['navy']} !important;font-size:.88rem;font-weight:700;
  border-bottom:2px solid {t['blue']};padding-bottom:.28rem;margin-bottom:.7rem}}

/* ── Chat badges ── */
.qb{{display:inline-block;padding:.15rem .55rem;border-radius:4px;
     font-size:.69rem;font-weight:600;margin-bottom:.5rem}}
.qb-doc  {{background:#1A3D2B;color:#6EE7A0 !important;border:1px solid #2D6A4F}}
.qb-data {{background:#3D2500;color:#FFBA57 !important;border:1px solid #8B5000}}
.qb-hyb  {{background:#2A1A4A;color:#C4A0FF !important;border:1px solid #6B46C1}}
.qb-adj  {{background:#0D2A4A;color:#7CC3F5 !important;border:1px solid #1E5A9A}}
.qb-ot   {{background:#3D2800;color:#FFD080 !important;border:1px solid #8B6000}}

/* ── Response body ── */
.rb p,.rb li,.rb span,.rb strong,.rb em,.rb h2,.rb h3,.rb h4
  {{color:{t['text']} !important;line-height:1.72}}

/* ── Chat footnotes ── */
.fn{{font-size:.72rem;color:{t['muted']} !important;
     border-top:1px solid {t['border']};padding-top:.4rem;
     margin-top:.6rem;font-style:italic}}
.ful{{font-size:.71rem;color:{t['muted']} !important;margin-bottom:.28rem}}

/* ── Data coverage badge ── */
.cov{{display:inline-block;background:{t['tag']};border:1px solid {t['border']};
  color:{t['muted']} !important;font-size:.7rem;padding:.18rem .55rem;
  border-radius:4px;margin-bottom:.5rem}}

/* ── Data Explorer summary cards ── */
.df{{color:{t['navy']} !important;font-weight:700;font-size:.82rem;
     border-bottom:1px solid {t['border']};padding-bottom:.25rem;margin-bottom:.5rem}}
.mc{{background:{t['surface']};border:1px solid {t['border']};
     border-left:3px solid {t['blue']};border-radius:6px;
     padding:.75rem .9rem;flex:1;min-width:0}}
.mv{{color:{t['navy']} !important;font-size:1.35rem;font-weight:800;margin:0}}
.ml{{color:{t['muted']} !important;font-size:.69rem;text-transform:uppercase;
     letter-spacing:.04em;margin:.1rem 0 0}}
.ms{{color:{t['blue']} !important;font-size:.67rem;margin:.05rem 0 0}}
.mr{{display:flex;gap:.65rem;margin-bottom:.85rem;flex-wrap:wrap}}

/* ── Documents ── */
.doc-cat{{color:{t['navy']} !important;font-weight:700;font-size:.9rem;
          border-bottom:2px solid {t['blue']};padding-bottom:.3rem;margin:.9rem 0 .55rem}}
.doc-item{{background:{t['surface']};border:1px solid {t['border']};
           border-radius:5px;padding:.55rem .85rem;margin-bottom:.3rem}}
.doc-item p{{color:{t['text']} !important;font-size:.82rem;margin:0;line-height:1.5}}
.doc-item .dy{{color:{t['muted']} !important;font-size:.7rem}}

/* ── Methodology blocks ── */
.mb{{background:{t['surface']};border:1px solid {t['border']};
     border-radius:6px;padding:1rem 1.3rem;margin-bottom:.75rem}}
.mt{{color:{t['navy']} !important;font-weight:700;font-size:.88rem;
     margin:0 0 .5rem;border-bottom:2px solid {t['blue']};padding-bottom:.28rem}}

/* ── Info box ── */
.ib{{background:{t['surface']};border:1px solid {t['border']};
     border-left:3px solid {t['accent']};border-radius:5px;
     padding:.5rem .8rem;font-size:.8rem;color:{t['text']} !important;
     margin-bottom:.7rem;line-height:1.6}}
.ib strong{{color:{t['navy']} !important}}

/* ── Navbar wrapper (no visual effect, just logical) ── */
.nb{{margin:0;padding:0}}

/* ── Footer ── */
.pf{{text-align:center;color:{t['muted']} !important;font-size:.69rem;
     padding:.5rem;border-top:1px solid {t['border']};margin-top:.5rem}}

/* ── Floating theme toggle + scroll-to-top ── */
.float-btns{{
  position:fixed;bottom:1.2rem;right:1.2rem;z-index:99999;
  display:flex;flex-direction:column;gap:.4rem;align-items:flex-end
}}
.float-btn{{
  background:{t['navy']};color:#FFFFFF;border:1px solid rgba(255,255,255,.25);
  border-radius:50%;width:2.2rem;height:2.2rem;cursor:pointer;
  font-size:1rem;display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 8px rgba(0,0,0,.3);transition:transform .15s
}}
.float-btn:hover{{transform:scale(1.1)}}

/* ── Hide dataframe toolbar (download, search, fullscreen) ── */
[data-testid="stElementToolbar"]
  {{display:none !important;visibility:hidden !important}}
[data-testid="StyledFullScreenFrame"] > div:last-child
  {{display:none !important}}

/* ── Responsive layout ── */
.block-container{{
  padding-left:clamp(.5rem,3vw,2rem) !important;
  padding-right:clamp(.5rem,3vw,2rem) !important;
  max-width:min(1400px,98vw) !important;
}}
@media (max-width:900px) {{
  .mr{{flex-direction:column !important}}
  .stat-card{{padding:.6rem .7rem !important}}
  .sn{{font-size:1.2rem !important}}
  div[data-testid="stHorizontalBlock"] button
    {{font-size:.68rem !important;padding:.2rem .3rem !important}}
}}
@media (max-width:640px) {{
  div[data-testid="stHorizontalBlock"]{{flex-wrap:wrap !important;gap:.2rem !important}}
  .ic{{padding:.7rem !important}}
}}

/* ── Stronger light-mode specificity overrides ── */
.stMarkdown p,.stMarkdown li,.stMarkdown label
  {{color:{t['text']} !important}}
[data-testid="stMarkdownContainer"] p
  {{color:{t['text']} !important}}
[data-testid="stForm"] label,[data-testid="stForm"] p
  {{color:{t['text']} !important}}
.stChatMessage [data-testid="stMarkdownContainer"] p
  {{color:{t['text']} !important}}
html body .stApp [data-testid="stWidgetLabel"] p
  {{color:{t['text']} !important;font-weight:600 !important}}
html body .stApp [data-testid="stRadio"] > label > div > p
  {{color:{t['text']} !important}}
html body .stApp [data-baseweb="select"] [class*="singleValue"]
  {{color:{t['text']} !important}}
html body .stApp [data-baseweb="select"] [class*="placeholder"]
  {{color:{t['muted']} !important}}
html body .stApp thead th,html body .stApp tbody td
  {{color:{t['text']} !important}}
html body .stApp [data-testid="stChatInput"] textarea
  {{color:{t['text']} !important;background:{t['surface']} !important;
    border-color:{t['border']} !important}}
</style>""", unsafe_allow_html=True)

# ── Initialize ────────────────────────────────────────────────────
@st.cache_resource
def initialize():
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm         = OpenAILLM(model="gpt-4o-mini", api_key=api_key, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
    coll    = chroma.get_or_create_collection("bri_documents")
    index   = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=coll))
    con     = duckdb.connect(DB_PATH, read_only=True)
    client  = OpenAI(api_key=api_key)
    return index, con, client

@st.cache_resource
def get_con():
    return duckdb.connect(DB_PATH, read_only=True)


# ── System prompt ─────────────────────────────────────────────────
BRI_SYSTEM_PROMPT = """
You are a specialised research assistant for Chinese overseas infrastructure
finance and the Belt and Road Initiative (BRI), built as part of doctoral
research in International Relations.

KNOWLEDGE SOURCES:
1. Document library — Chinese government white papers, official speeches,
   World Bank/IMF/UN reports, academic literature, think tank research.
2. AidData (2025) China's Global Loans and Grants Dataset v1.0 — 4,861 infrastructure projects, 183 countries, 2000–2023.

STRICT EPISTEMIC RULES (non-negotiable):
1. ONLY include claims directly supported by a retrieved source passage
   or an AidData query result shown to you. Do NOT draw on general training
   knowledge to fill gaps. If sources are insufficient, say so explicitly.
2. Never state contested empirical claims as descriptive fact. The following
   MUST always be attributed to a specific named source and flagged:
   — Claims about which governance/corruption profiles China targets
   — Claims about strategic motivations behind financing decisions
   — Claims comparing Chinese and Western financing models
   Frame these as: "According to [source], scholars argue that..."
3. Distinguish Chinese official framing from independent scholarly analysis.
4. AidData figures = financing commitments, not disbursements. Always note.
5. CAUSAL CLAIMS PROHIBITED unless from a cited quantitative study.
   Use "associated with", "coincides with" — never "caused" or "led to".
6. No forecasting. Trend interpretation only.
7. NEVER use backtick or code formatting for numbers, figures, or any text.
8. Do NOT insert parenthetical source citations (Source 2) or (Author, year)
   — attribute inline: "According to the World Bank (2019) Belt and Road Economics..."

SOURCE ATTRIBUTION RULES:
- Always cite sources by their full institutional name as given in the
  [Source: ...] tag. For example:
  Correct: "According to the Nedopil / Green Finance & Development Center BRI Investment Reports..."
  Correct: "The AidData (2021) Banking on the Belt and Road report notes..."
  Never use: filename-style references like "Nedopil-2026" or "nedopil_2026"
- If multiple passages from the same source support a claim, cite it once.

WHEN DOCUMENT SOURCES ARE INSUFFICIENT:
Say clearly: "The indexed document sources do not contain sufficient
information to address [specific aspect]. The available sources address..."
Do not supplement with general knowledge.

RESPONSE STYLE:
- Write like a knowledgeable research colleague: clear, grounded, concise.
- Lead with the most important finding, then supporting detail.
- Simple questions: 2-3 paragraphs, no headers.
- Complex hybrid: use headers only when genuinely needed.
- Adapt length to question complexity — do not pad.

FOR ALL RESPONSES: End with a brief "Analytical note:" paragraph stating
(a) what this analysis establishes from the sources, (b) what it cannot
establish, and (c) what further verification is recommended.
"""

# ── Constants ─────────────────────────────────────────────────────
REGION_MAP = {"America": "Latin America & Caribbean"}
KNOWN_TABLES = {"bri_projects_core","bri_projects_south_asia","bri_projects_full"}
OFF_TOPIC_MSG = ("This tool is designed for research on Chinese overseas "
    "infrastructure finance and the Belt and Road Initiative. Your question "
    "appears to be outside that scope. Please try a question about BRI "
    "financing patterns, Chinese development finance, or related topics.")

# Keywords that force hybrid routing regardless of LLM routing decision
HYBRID_KEYWORDS = {
    "trend","trends","over time","change","recent","latest","post-2023",
    "sector","sectors","green","shift","decline","decrease","increase",
    "growth","compare","comparison","versus","vs","contrast","why",
    "explain","context","policy","official","framing",
    "narrative","2024","2025","2026","after 2023",
    "from 2023","since 2023","from 2022","since 2022","current","now",
    "today","this year","last year","recently","nowadays","latest data",
    "current state","state of","what is happening","ongoing"
}

# Phrases that force documents-only routing (question is about what sources say)
DOCUMENT_FORCE_PHRASES = {
    "white paper","white papers","what do.*documents","which documents",
    "what does.*say","what does.*state","what does.*outline",
    "according to.*policy","bri forum speech","xi jinping","communiqué",
    "what are.*strategies in bri","key financing strategies",
    "guiding principles","debt sustainability framework",
    "what does.*report say","document analysis","indexed sources"
}

def should_force_documents(q: str) -> bool:
    """Force documents-only route when question is clearly about source content."""
    q_lower = q.lower()
    import re
    for phrase in DOCUMENT_FORCE_PHRASES:
        if re.search(phrase, q_lower):
            return True
    return False

SAMPLE_QUERIES = [
    "Which 10 countries received the most Chinese infrastructure financing from 2013 to 2023?",
    "How has Chinese energy sector financing changed by year from 2010 to 2023?",
    "Compare total BRI financing across regions: Africa, South Asia, and Southeast Asia",
    "What do BRI debt sustainability documents say about high-risk borrower countries?"
]

# ── Source display name mapping ───────────────────────────────────
# Maps filename substrings (lowercase) → proper academic citation name
SOURCE_NAMES = {
    "nedopil": "Nedopil / Green Finance & Development Center, BRI Investment Reports",
    "banking_on_the_belt": "AidData (2021), Banking on the Belt and Road",
    "banking on the belt": "AidData (2021), Banking on the Belt and Road",
    "how_china_lends": "AidData / SAIS-CARI / Peterson Institute (2021), How China Lends",
    "how china lends": "AidData / SAIS-CARI / Peterson Institute (2021), How China Lends",
    "belt_and_road_economics": "World Bank (2019), Belt and Road Economics",
    "belt and road economics": "World Bank (2019), Belt and Road Economics",
    "guiding_principles": "Ministry of Finance PRC (2017), Guiding Principles on Financing BRI",
    "guiding principles": "Ministry of Finance PRC (2017), Guiding Principles on Financing BRI",
    "debt_sustainability": "Ministry of Finance PRC (2019), BRI Debt Sustainability Framework",
    "debt sustainability": "Ministry of Finance PRC (2019), BRI Debt Sustainability Framework",
    "green_investment": "Green Investment Principles for the Belt and Road (2019)",
    "green investment": "Green Investment Principles for the Belt and Road (2019)",
    "vision_and_actions": "NDRC / MFA / MOFCOM (2015), Vision and Actions on Jointly Building the Silk Road",
    "vision and actions": "NDRC / MFA / MOFCOM (2015), Vision and Actions on Jointly Building the Silk Road",
    "bri_forum": "Belt and Road Forum for International Cooperation, Joint Communiqué",
    "brf": "Belt and Road Forum for International Cooperation, Joint Communiqué",
    "forum": "Belt and Road Forum for International Cooperation",
    "xi_jinping": "Xi Jinping, BRI Forum Keynote Speech",
    "xi jinping": "Xi Jinping, BRI Forum Keynote Speech",
    "chasing_china": "AidData / GeoEcon (2023), Chasing China's Belt and Road",
    "chasing china": "AidData / GeoEcon (2023), Chasing China's Belt and Road",
    "sdg": "United Nations, Sustainable Development Goals Progress Report",
    "wb_belt": "World Bank, Belt and Road Initiative Report",
    "imf": "International Monetary Fund, BRI Assessment",
    "green_belt": "Ministry of Ecology and Environment PRC (2017), Guidance on Promoting Green Belt and Road",
    "green belt": "Ministry of Ecology and Environment PRC (2017), Guidance on Promoting Green Belt and Road",
}

def get_source_display_name(filename: str) -> str:
    """Map a PDF filename to a proper academic citation name."""
    fn_lower = filename.lower().replace(".pdf","")
    for key, display in SOURCE_NAMES.items():
        if key in fn_lower:
            return display
    # Fallback: clean filename without extension, title-cased
    return fn_lower.replace("_"," ").replace("-"," ").title()


# ── Helpers ───────────────────────────────────────────────────────
def clean_resp(text: str) -> str:
    """Strip backticks, code formatting, parenthetical citations, and markdown links."""
    text = text.replace("```", "")
    # Remove ALL inline backtick spans
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    # Remove any remaining stray backticks
    text = text.replace('`', '')
    # Strip markdown hyperlinks — convert [text](url) to just text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Strip bare URLs that GPT sometimes inserts
    text = re.sub(r'https?://\S+', '', text)
    # Strip parenthetical doc citations
    text = re.sub(
        r'\([A-Z][^\)]{4,80}(?:\.pdf|\.gov\.cn|WB|IMF|ADB|UN|20\d\d)[^\)]*\)', '', text)
    text = re.sub(r'\(Source\s*\d+\)', '', text)
    # Clean double spaces
    return re.sub(r'  +', ' ', text).strip()

def validate_sql(sql: str) -> tuple:
    for op in ["DELETE","UPDATE","DROP","INSERT","ALTER","CREATE","TRUNCATE"]:
        if op in sql.upper(): return False, f"Disallowed operation: {op}"
    if not any(t in sql for t in KNOWN_TABLES): return False, "No known table referenced."
    return True, ""

def apply_regions(df):
    if df is None or df.empty: return df
    df = df.copy()
    if "Region_of_Activity" in df.columns:
        df["Region_of_Activity"] = df["Region_of_Activity"].replace(REGION_MAP)
    # Also fix any column named Region
    if "Region" in df.columns:
        df["Region"] = df["Region"].replace(REGION_MAP)
    return df

def round_df(df):
    """Round all float columns to 2 decimal places."""
    if df is None or df.empty: return df
    df = df.copy()
    for col in df.select_dtypes(include=["float64","float32"]).columns:
        df[col] = df[col].round(2)
    return df

def show_df(df):
    """Display a clean dataframe with 1-based index, drop rows that are entirely null."""
    if df is None or df.empty: return
    df = round_df(apply_regions(df))
    # Drop rows where all numeric columns are NaN/None (e.g. World Bank: None comparison rows)
    num_cols = df.select_dtypes(include=["float64","int64","float32"]).columns
    if len(num_cols) > 0:
        df = df.dropna(subset=num_cols, how="all")
    if df.empty: return
    ren = {c: f"{c} (USD Bn)" for c in df.columns
           if any(x in c.lower() for x in ["billion","usd","financing","investment","amount","total"])
           and "(USD" not in c}
    out = df.rename(columns=ren).reset_index(drop=True).copy()
    out.index = range(1, len(out)+1)
    st.dataframe(out, use_container_width=True)

def history_ctx(msgs, n=3):
    pairs, i = [], len(msgs)-1
    while i >= 0 and len(pairs) < n:
        if msgs[i]["role"]=="assistant" and i>0 and msgs[i-1]["role"]=="user":
            pairs.append(f"User: {msgs[i-1]['content']}\n"
                         f"Assistant (summary): {msgs[i]['content'][:200]}...")
            i -= 2
        else: i -= 1
    return "\n\n".join(reversed(pairs))

def classify_topic(q, client):
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role":"user","content":
            f"Classify as 'core' (BRI/Chinese infra finance/policy), "
            f"'adjacent' (IR theory/dev economics/geopolitics/multilateral), "
            f"or 'off_topic' (unrelated to IR/development/finance).\n"
            f"Reply ONE word.\nQuestion: {q}"}])
    v = r.choices[0].message.content.strip().lower()
    return v if v in ["core","adjacent","off_topic"] else "core"

def should_force_hybrid(q: str) -> bool:
    """Force hybrid routing for trend/comparative/recent questions."""
    q_lower = q.lower()
    return any(kw in q_lower for kw in HYBRID_KEYWORDS)

def should_show_chart(route: str, df) -> bool:
    """Show charts only for pure data queries with aggregated results."""
    if route != "data": return False
    if df is None or df.empty or len(df) < 2: return False
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    if not num_cols: return False
    # Suppress chart if result looks like a project listing
    proj_cols = [c for c in df.columns
                 if any(x in c.lower() for x in ["title","project","name","description","display"])]
    if proj_cols and len(df) > 4: return False
    return True


# ── Chart helpers ─────────────────────────────────────────────────
def make_chat_chart(df):
    """Build chart for chat data responses — full time series, no head(12) cap."""
    if df is None or df.empty or len(df) < 2: return None
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not num_cols: return None

    prio    = ["billion","usd","financing","investment","amount","total"]
    num_col = next((c for c in num_cols if any(p in c.lower() for p in prio)), num_cols[0])
    year_col= next((c for c in df.columns if "year" in c.lower()), None)
    cat_col = next((c for c in cat_cols), None)

    unit = num_col.replace("_"," ")
    if any(x in num_col.lower() for x in prio) and "(USD" not in num_col:
        unit = f"{unit} (USD Bn)"

    dark = st.session_state.dark_mode
    fc   = "#E6EDF3" if dark else "#0F1923"
    gc   = "rgba(255,255,255,.07)" if dark else "rgba(0,0,0,.06)"

    # Time series: use ALL rows, not head(12)
    df_plot = df.copy() if (year_col and df[year_col].nunique() > 3) else df.head(15).copy()

    try:
        if year_col and df[year_col].nunique() > 3:
            fig = px.line(df_plot, x=year_col, y=num_col, markers=True,
                          color_discrete_sequence=["#2E6DA4"],
                          labels={num_col: unit, year_col: "Year"})
            fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
        elif cat_col:
            df_plot = df_plot.sort_values(num_col, ascending=True)
            fig = px.bar(df_plot, x=num_col, y=cat_col, orientation="h",
                         color=cat_col, color_discrete_sequence=CHART_PAL,
                         labels={num_col: unit, cat_col: ""})
            fig.update_layout(showlegend=False)
        else:
            return None

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=11, color=fc),
            margin=dict(l=15, r=100, t=30, b=20),
            height=max(280, len(df_plot)*32),
            yaxis=dict(tickfont=dict(size=10, color=fc), gridcolor=gc),
            xaxis=dict(tickfont=dict(size=10, color=fc), gridcolor=gc),
            annotations=[dict(text="The BRI Monitor · AidData (2025) CGLD v1.0",
                x=.99, y=.01, xref="paper", yref="paper", showarrow=False,
                xanchor="right", yanchor="bottom",
                font=dict(size=7, color="rgba(120,120,120,.35)"))]
        )
        return fig
    except: return None

def de_chart_layout(fig, height=360):
    dark = st.session_state.dark_mode
    fc   = "#E6EDF3" if dark else "#0F1923"
    gc   = "rgba(255,255,255,.07)" if dark else "rgba(0,0,0,.06)"
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=11, color=fc),
        margin=dict(l=10, r=10, t=40, b=35), height=height,
        yaxis=dict(gridcolor=gc, tickfont=dict(size=10, color=fc)),
        xaxis=dict(gridcolor=gc, tickfont=dict(size=10, color=fc)),
        legend=dict(font=dict(size=10, color=fc), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text="The BRI Monitor · AidData (2025) CGLD v1.0",
            x=.99, y=-.11, xref="paper", yref="paper", showarrow=False,
            xanchor="right", font=dict(size=7, color="rgba(120,120,120,.38)"))]
    )
    return fig


# ── Agent ─────────────────────────────────────────────────────────
def q_docs(question, index, client, history="", adjacent=False):
    nodes  = index.as_retriever(similarity_top_k=15).retrieve(question)
    parts, sources = [], []
    for node in nodes:
        fn   = node.metadata.get("file_name","Unknown")
        name = get_source_display_name(fn)
        parts.append(f"[Source: {name}]\n{node.node.text[:1400]}")
        if fn not in sources: sources.append(fn)
    hb  = f"\nCONVERSATION CONTEXT:\n{history}\n" if history else ""
    adj = ("\nNote: adjacent topic — answer only from sources, clearly flag gaps.\n"
           if adjacent else "")
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1,
        messages=[
            {"role":"system","content":BRI_SYSTEM_PROMPT},
            {"role":"user","content":
                f"Answer using ONLY the provided source passages.\n"
                f"Attribute sources inline using their full institutional name as given "
                f"in the [Source: ...] tags. Never use filename-style references. "
                f"No parenthetical citations. No backtick formatting. No general knowledge.\n"
                f"If sources are insufficient, say so clearly.\n"
                f"{adj}{hb}\nQUESTION: {question}\n\nPASSAGES:\n{''.join(parts)}"}
        ])
    return {"answer":r.choices[0].message.content, "sources":sources, "type":"document"}

def q_data(question, con, client, history=""):
    hb = f"\nCONVERSATION CONTEXT:\n{history}\n" if history else ""
    sql_prompt = (
        "You are a SQL expert for a Chinese infrastructure finance DuckDB database.\n"
        f"{hb}\n"
        "TABLES (always default to bri_projects_core unless user specifies otherwise):\n"
        "1. bri_projects_core        — developing countries, 4,498 rows  ← DEFAULT\n"
        "2. bri_projects_south_asia  — South Asia subset, 463 rows\n"
        "3. bri_projects_full        — all countries including developed, 4,861 rows\n\n"
        "KEY COLUMNS (ONLY use real column names from this list — no aliases in SELECT unless aggregating):\n"
        "Country_of_Activity, Region_of_Activity, Commitment_Year, Completion_Year,\n"
        "Display_Title, Title, Flow_Type, Sector_Name, Funding_Agencies_Parent,\n"
        "Amount_Nominal_USD, Amount_Constant_USD_2023, Collateralized,\n"
        "Level_of_Public_Liability, WB_Income_Group_Host_Country, Tranche_Count\n"
        "For project names always use: COALESCE(Display_Title, Title) AS Project_Name\n"
        "NEVER select both Display_Title AND Title in the same query — use COALESCE instead.\n\n"
        "EXACT REGION VALUES (use exactly as written):\n"
        "'Africa','America','Asia','Europe','Middle East','Oceania','Multi-Region'\n"
        "Latin America is stored as 'America' — always use 'America' in SQL.\n\n"
        "EXACT SECTOR VALUES (uppercase, never use LIKE or partial matches):\n"
        "'ENERGY','TRANSPORT AND STORAGE','INDUSTRY, MINING, CONSTRUCTION',\n"
        "'EDUCATION','COMMUNICATIONS','HEALTH','WATER SUPPLY AND SANITATION',\n"
        "'GOVERNMENT AND CIVIL SOCIETY','AGRICULTURE, FORESTRY, FISHING',\n"
        "'OTHER SOCIAL INFRASTRUCTURE AND SERVICES','OTHER MULTISECTOR'\n\n"
        "FLOW TYPES: 'Loan','Grant','Vague TBD'\n\n"
        "CRITICAL SQL RULES:\n"
        "- Use Amount_Constant_USD_2023 for ALL financial comparisons\n"
        "- ALWAYS apply ROUND(SUM(Amount_Constant_USD_2023)/1e9, 2) for financial totals\n"
        "- Name the rounded column clearly e.g. Total_Financing_Billions\n"
        "- WHERE Amount_Nominal_USD IS NOT NULL for financial queries\n"
        "- NEVER create derived columns not in the KEY COLUMNS list\n"
        "- NEVER add a Financing_Model, Financing_Type or similar invented column\n"
        "- South Asia → bri_projects_south_asia\n"
        "- Global/all countries → bri_projects_core (NOT bri_projects_full)\n\n"
        "QUERY TYPE RULES — read carefully before choosing:\n"
        "\n"
        "COUNTRY vs REGION: CRITICAL DISTINCTION\n"
        "- If the question names SPECIFIC COUNTRIES (e.g. Indonesia, Russia, Pakistan):\n"
        "  → ALWAYS use Country_of_Activity, NEVER Region_of_Activity\n"
        "  → WHERE Country_of_Activity IN ('Indonesia', 'Russia') etc.\n"
        "- If the question asks about regions/continents (Africa, Asia, etc.):\n"
        "  → Use Region_of_Activity\n"
        "\n"
        "TREND / YEAR RANGE (e.g. 'from 2010 to 2020', 'by year', 'trends'):\n"
        "  → MUST include Commitment_Year in SELECT and GROUP BY\n"
        "  → SELECT Region_of_Activity, Commitment_Year, ROUND(SUM(...)/1e9,2)\n"
        "  → GROUP BY Region_of_Activity, Commitment_Year (or Country, Commitment_Year)\n"
        "  → ORDER BY Region_of_Activity, Commitment_Year, NO LIMIT\n"
        "  → Show ALL years in range — never drop the year column\n"
        "\n"
        "REGIONAL TOTAL (no year, just overall totals by region):\n"
        "  → SELECT Region_of_Activity, ROUND(SUM(...)/1e9,2) AS Total_Financing_Billions\n"
        "  → GROUP BY Region_of_Activity, ORDER BY Total_Financing_Billions DESC\n"
        "\n"
        "SECTOR BREAKDOWN per country/region:\n"
        "  → SELECT Country_of_Activity (or Region), Sector_Name, ROUND(SUM(...)/1e9,2)\n"
        "  → GROUP BY Country_of_Activity (or Region), Sector_Name\n"
        "  → ORDER BY Country/Region, Total DESC\n"
        "\n"
        "RANKING (top N countries/sectors):\n"
        "  → GROUP BY dimension, ORDER BY value DESC, LIMIT 20\n"
        "\n"
        "PROJECT LISTING (explicit 'list projects', 'which projects'):\n"
        "  → COALESCE(Display_Title,Title) AS Project_Name, Country_of_Activity, Commitment_Year, ROUND(Amount_Constant_USD_2023/1e6,2) AS Amount_M\n"
        "  → ORDER BY Amount DESC, LIMIT 25\n"
        "\n"
        "DEFAULT: prefer aggregation. Never omit Commitment_Year if question mentions years/trends.\n\n"
        f"QUESTION: {question}\nReturn ONLY the SQL — no explanation, no fences."
    )
    sr = client.chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role":"user","content":sql_prompt}])
    sql = sr.choices[0].message.content.strip().replace("```sql","").replace("```","").strip()

    ok, warn = validate_sql(sql)
    if not ok:
        return {"answer":f"Query could not be executed: {warn}",
                "sql":sql,"data":None,"type":"data_error"}
    try:
        df = round_df(apply_regions(get_con().execute(sql).df()))
        ir = client.chat.completions.create(
            model="gpt-4o", temperature=0.1,
            messages=[
                {"role":"system","content":BRI_SYSTEM_PROMPT},
                {"role":"user","content":
                    f"Interpret these AidData results. Figures = commitments not disbursements.\n"
                    f"No backtick formatting. No parenthetical citations. No general knowledge.\n"
                    f"Only interpret what the data shows — do not infer causality.\n"
                    f"{hb}\nQUESTION: {question}\nRESULTS:\n{df.to_string(index=False)}"}
            ])
        answer = ir.choices[0].message.content

        # Pakistan/CPEC scope note — figures often queried vs public $62bn claim
        pk_query = "pakistan" in question.lower() or "cpec" in question.lower()
        pk_data  = (df is not None and not df.empty and
                    any("Pakistan" in str(v) for v in df.values.flat))
        if pk_query or pk_data:
            answer += (
                "\n\n**Data scope note (Pakistan / CPEC):** The figure above reflects "
                "AidData's coverage of Chinese official finance (state-backed loans and grants) "
                "and will differ from the ~$62 billion figure commonly cited for CPEC. That "
                "higher figure includes announced project packages, private investment, and "
                "non-official finance that fall outside AidData's scope. The AidData figure "
                "represents the more conservative, verifiable official-finance component only."
            )
        return {"answer":answer,"sql":sql,"data":df,"type":"data"}
    except Exception as e:
        return {"answer":f"Data query error: {e}","sql":sql,"data":None,"type":"data_error"}

def bri_agent(question, index, con, client):
    # Off-topic guard
    topic = classify_topic(question, client)
    if topic == "off_topic":
        return {"answer":OFF_TOPIC_MSG,"route":"off_topic",
                "sources":[],"data":None,"followups":[]}

    adjacent = (topic == "adjacent")
    history  = history_ctx(st.session_state.messages)

    # Detect chart/visualisation requests and prepend a note
    chart_request = any(kw in question.lower() for kw in
        ["pie chart","bar chart","line chart","chart","graph","visuali","plot"])
    chart_note = (
        "[NOTE: This tool cannot render charts directly in chat. "
        "Provide the underlying data clearly in your response, and the system "
        "will display it as a data table. The user can view charts in the "
        "Data Explorer page. Do NOT refuse the data query — answer with the data.]\n\n"
    ) if chart_request else ""

    # Force documents-only for clearly source-content questions
    if should_force_documents(question):
        route = "documents"
    # Force hybrid for trend/comparative/recent keywords
    elif should_force_hybrid(question):
        route = "both"
    else:
        rr = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role":"user","content":
                f"Classify as 'documents', 'data', or 'both'.\n"
                f"Use context for follow-up questions.\n"
                f"{f'CONTEXT: {history[:300]}' if history else ''}\n"
                f"QUESTION: {question}\nReply ONE word."}])
        route = rr.choices[0].message.content.strip().lower()
        if route not in ["documents","data","both"]: route = "both"

    if route == "documents":
        result = q_docs(question, index, client, history, adjacent)
    elif route == "data":
        result = q_data(question + (f"\n{chart_note}" if chart_note else ""), con, client, history)
    else:
        dr = q_docs(question, index, client, history, adjacent)
        da = q_data(question + (f"\n{chart_note}" if chart_note else ""), con, client, history)
        # Detect if question asks about periods beyond AidData coverage
        beyond_2023 = any(yr in question for yr in ["2024","2025","2026","current","recent","latest","ongoing"])
        post_note = (
            "\nNOTE: The question references a period beyond AidData's 2023 coverage. "
            "You MUST explicitly draw on the document sources (especially Nedopil BRI "
            "Investment Reports 2023/2025/2026) for post-2023 trends. Clearly attribute "
            "each post-2023 claim to its document source. State explicitly when data "
            "coverage ends (2023) and which document provides post-2023 context.\n"
        ) if beyond_2023 else ""
        comb = client.chat.completions.create(
            model="gpt-4o", temperature=0.1,
            messages=[
                {"role":"system","content":BRI_SYSTEM_PROMPT},
                {"role":"user","content":
                    f"Synthesise into a scholarly response using ONLY the analyses below.\n"
                    f"No general knowledge. No backtick formatting. No markdown links. "
                    f"No parenthetical citations. No bare URLs.\n"
                    f"FRAMING RULES:\n"
                    f"- Lead with whichever source best answers the question — not always AidData.\n"
                    f"- If the question asks about documents/policy/speeches, lead with document analysis.\n"
                    f"- If the question asks about data/numbers/trends, lead with AidData findings.\n"
                    f"- Only reference AidData if its results add concrete value (numbers, rankings, trends).\n"
                    f"- If document analysis and data results contradict, state clearly:\n"
                    f"  'The document sources indicate X, while the AidData dataset shows Y.\n"
                    f"  This discrepancy may reflect differences in coverage or definitional scope.'\n"
                    f"{post_note}"
                    f"{f'CONTEXT: {history[:300]}' if history else ''}\n"
                    f"QUESTION: {question}\n"
                    f"DOCUMENT ANALYSIS:\n{dr['answer']}\n\n"
                    f"DATA ANALYSIS:\n{da['answer']}"}
            ])
        raw_answer = comb.choices[0].message.content

        # ── Self-revision quality check ──────────────────────────
        revision_prompt = (
            f"You are a quality reviewer for a BRI research assistant.\n"
            f"ORIGINAL QUESTION: {question}\n\n"
            f"DRAFT RESPONSE:\n{raw_answer}\n\n"
            f"Check the draft for these issues and rewrite ONLY if problems found:\n"
            f"1. Does it actually answer the question asked? (not a tangential answer)\n"
            f"2. Are specific numbers cited correctly from the data provided?\n"
            f"3. Is it free of vague filler ('it is worth noting', 'it is important to understand')?\n"
            f"4. Does it avoid making causal claims not supported by sources?\n"
            f"5. If question names specific countries, does response discuss those countries — not just regions?\n"
            f"If the draft is good, return it unchanged. If there are issues, rewrite to fix them.\n"
            f"Return ONLY the final response text — no preamble, no explanation of changes."
        )
        rev = client.chat.completions.create(
            model="gpt-4o", temperature=0.1,
            messages=[{"role":"user","content":revision_prompt}])
        final_answer = rev.choices[0].message.content

        result = {"answer":final_answer,
                  "sources":dr.get("sources",[]),"data":da.get("data"),
                  "type":"hybrid"}

    # Follow-ups
    try:
        fr = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.4,
            messages=[{"role":"user","content":
                f"Suggest 3 follow-up research questions for a BRI finance research tool.\n"
                f"The tool has: (1) AidData dataset — 4,861 projects, 183 countries, 2000–2023;\n"
                f"(2) 42 indexed policy documents (BRI white papers, World Bank, Nedopil reports).\n"
                f"Rules: Each question must be answerable with the data or documents above.\n"
                f"Suggest ONE data query (country/sector/year numbers), ONE document query\n"
                f"(policy/framework/strategy), ONE comparative (two regions/sectors/periods).\n"
                f"Under 12 words each. No general geopolitics. No questions outside BRI finance scope.\n"
                f"Return ONLY a Python list of 3 strings.\nContext question: {question}"}])
        ft = fr.choices[0].message.content.strip().replace("```python","").replace("```","").strip()
        followups = ast.literal_eval(ft)
        if not isinstance(followups, list): followups = []
    except: followups = []

    result["route"]    = route
    result["followups"]= followups
    result["adjacent"] = adjacent
    return result


# ── Navbar ────────────────────────────────────────────────────────
NAV = [("home","Home"),("chat","Chat"),("data_explorer","Data Explorer"),
       ("documents","Documents"),("about","About"),("contact","Contact")]

def render_navbar():
    # Navbar is ALWAYS dark navy — self-contained CSS so it works on every page
    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
  background:#1B3A6B !important;
  padding:.4rem .8rem !important;
  border-radius:0 0 6px 6px !important;
  margin-bottom:.9rem !important;
  align-items:center !important;
  gap:.3rem !important;
}
div[data-testid="stHorizontalBlock"] > div {
  display:flex !important; align-items:center !important;
}
div[data-testid="stHorizontalBlock"] button {
  height:2.4rem !important; min-height:2.4rem !important;
  padding:.25rem .6rem !important; font-size:.76rem !important;
  font-weight:600 !important; border-radius:5px !important;
  cursor:pointer !important; display:flex !important;
  align-items:center !important; justify-content:center !important;
  width:100% !important; white-space:normal !important;
  word-break:break-word !important; line-height:1.15 !important;
  text-align:center !important;
  border:1px solid rgba(255,255,255,.25) !important;
  background:rgba(255,255,255,.1) !important;
  color:#FFFFFF !important; pointer-events:auto !important;
  position:relative !important; z-index:1 !important;
}
div[data-testid="stHorizontalBlock"] button p { color:#FFFFFF !important }
div[data-testid="stHorizontalBlock"] button:hover {
  background:rgba(255,255,255,.22) !important;
  border-color:rgba(255,255,255,.5) !important;
}
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
  background:rgba(255,255,255,.28) !important;
  border:1.5px solid rgba(255,255,255,.7) !important;
  color:#FFFFFF !important;
  box-shadow:0 0 0 2px rgba(255,255,255,.12) !important;
  font-weight:600 !important;
}
</style>""", unsafe_allow_html=True)
    c0, *nav_cols, cz = st.columns([3]+[1.5]*len(NAV)+[0.5])
    with c0:
        st.markdown("""
<div style="padding:.3rem 0">
  <p style="color:#fff !important;font-size:1.1rem;font-weight:800;margin:0;
            font-family:Arial,sans-serif;letter-spacing:-.01em">🌐 The BRI Monitor</p>
  <p style="color:#A8D4F5 !important;font-size:.62rem;margin:.06rem 0 0;
            font-family:Arial,sans-serif;white-space:nowrap;overflow:hidden;
            text-overflow:ellipsis">
    BRI · Chinese Development Finance · AI</p>
</div>""", unsafe_allow_html=True)
    cur = st.session_state.page
    for col, (pg, label) in zip(nav_cols, NAV):
        with col:
            btn_type = "primary" if cur == pg else "secondary"
            if st.button(label, key=f"nb_{pg}",
                         use_container_width=True, type=btn_type):
                st.session_state.page = pg
                st.rerun()
    with cz:
        # Theme locked to dark mode — toggle removed
        pass


# ── Chat message renderer ─────────────────────────────────────────
def render_msg(message, idx):
    t     = T()
    route = message.get("route","")
    adj   = message.get("adjacent", False)

    badge = {"documents":("qb-doc","📄 Document Query"),
             "data":("qb-data","📊 Data Query"),
             "both":("qb-hyb","🔗 Hybrid Query"),
             "off_topic":("qb-ot","⚠️ Out of Scope")}
    if adj:
        badge = {k:("qb-adj", f"{v[1]} · adjacent topic") for k,v in badge.items()}
    cls, lbl = badge.get(route, ("qb-doc",""))
    if lbl:
        st.markdown(f'<span class="qb {cls}">{lbl}</span>', unsafe_allow_html=True)

    answer = clean_resp(message.get("content", message.get("answer","")))
    st.markdown(f'<div class="rb" style="color:{t["text"]};line-height:1.72">'
                f'{answer}</div>', unsafe_allow_html=True)

    # Data table — only show when genuinely adds value
    data = message.get("data")
    def _table_is_meaningful(df, route):
        if df is None or df.empty or len(df) < 2: return False
        if route not in ("data", "both"): return False
        # Must have at least one numeric column (aggregated result)
        num_cols = df.select_dtypes(include=["float64","int64","float32"]).columns.tolist()
        if not num_cols: return False
        # Suppress if it's a raw string-only funder/agency listing (no numeric values)
        all_string = all(df[c].dtype == object for c in df.columns if c not in ["Sector","Region","Country"])
        if all_string and len(num_cols) == 0: return False
        return True
    show_table = _table_is_meaningful(data, route)
    if show_table:
        with st.expander("📊 Data Table & Chart", expanded=True):
            show_df(data)
            if should_show_chart(route, data):
                fig = make_chat_chart(data)
                if fig: st.plotly_chart(fig, use_container_width=True)

    # Source note
    if message.get("sources") and route != "off_topic":
        st.markdown(f'<div class="fn">📚 Draws on '
                    f'{len(message["sources"])} indexed source(s).</div>',
                    unsafe_allow_html=True)

    # Follow-ups
    if message.get("followups"):
        st.markdown('<p class="ful">💡 You might also ask:</p>', unsafe_allow_html=True)
        fcols = st.columns(len(message["followups"]))
        for j, (fc, fq) in enumerate(zip(fcols, message["followups"])):
            with fc:
                if st.button(fq, key=f"fu_{idx}_{j}", use_container_width=True):
                    st.session_state.pending_query = fq
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════
def show_home():
    t = T(); inject_css()
    render_navbar()
    st.markdown("<div style='margin-bottom:.6rem'></div>", unsafe_allow_html=True)

    # ── Hero + Globe (globe in isolated iframe so JS survives reruns) ──
    hero_left, hero_right = st.columns([1, 2.8])
    with hero_left:
        components.html("""
<!DOCTYPE html><html><body style="margin:0;background:transparent;display:flex;
  align-items:center;justify-content:center;height:200px">
<canvas id="g" width="190" height="190" style="border-radius:50%"></canvas>
<script>
(function(){
  var c=document.getElementById('g'),ctx=c.getContext('2d');
  var W=190,H=190,R=88,cx=W/2,cy=H/2,rot=0;
  var nodes=[
    [23.7,90.4],[31.5,74.3],[1.3,103.8],[39.9,116.4],[3.1,101.7],
    [15.5,32.5],[-1.3,36.8],[41.0,28.9],[59.9,30.3],[48.9,2.3],
    [51.5,-0.1],[55.7,37.6],[-33.9,18.4],[36.7,3.0],[25.2,55.3],
    [13.5,2.1],[6.4,2.3],[-8.8,13.2],[4.4,18.6],[-34.6,-58.4],
    [19.4,-99.1],[40.7,-74.0],[35.7,139.7],[-37.8,144.9]
  ];
  function ll(la,lo,r){
    var ph=(90-la)*Math.PI/180,th=(lo+r)*Math.PI/180;
    return{x:cx+R*Math.sin(ph)*Math.cos(th),y:cy-R*Math.cos(ph),
           z:R*Math.sin(ph)*Math.sin(th)};
  }
  function draw(r){
    ctx.clearRect(0,0,W,H);
    var g=ctx.createRadialGradient(cx,cy,8,cx,cy,R);
    g.addColorStop(0,'rgba(22,60,110,.98)');
    g.addColorStop(1,'rgba(5,12,30,.99)');
    ctx.beginPath();ctx.arc(cx,cy,R,0,2*Math.PI);
    ctx.fillStyle=g;ctx.fill();
    // Grid
    for(var lo=-180;lo<180;lo+=30){
      ctx.beginPath();var f=true;
      for(var la=-90;la<=90;la+=4){
        var p=ll(la,lo,r);
        if(p.z>=0){if(f){ctx.moveTo(p.x,p.y);f=false;}else ctx.lineTo(p.x,p.y);}
        else f=true;
      }
      ctx.strokeStyle='rgba(46,109,164,.2)';ctx.lineWidth=.55;ctx.stroke();
    }
    for(var la2=-60;la2<=60;la2+=30){
      ctx.beginPath();f=true;
      for(var lo2=-180;lo2<=180;lo2+=4){
        var p2=ll(la2,lo2,r);
        if(p2.z>=0){if(f){ctx.moveTo(p2.x,p2.y);f=false;}else ctx.lineTo(p2.x,p2.y);}
        else f=true;
      }
      ctx.strokeStyle='rgba(46,109,164,.13)';ctx.lineWidth=.45;ctx.stroke();
    }
    // Equator
    ctx.beginPath();f=true;
    for(var lo3=-180;lo3<=180;lo3+=3){
      var pe=ll(0,lo3,r);
      if(pe.z>=0){if(f){ctx.moveTo(pe.x,pe.y);f=false;}else ctx.lineTo(pe.x,pe.y);}
      else f=true;
    }
    ctx.strokeStyle='rgba(91,184,245,.4)';ctx.lineWidth=1;ctx.stroke();
    // Nodes
    nodes.forEach(function(n){
      var p=ll(n[0],n[1],r);
      if(p.z>2){
        var a=Math.min(1,(p.z/R)*1.5);
        var ng=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,7);
        ng.addColorStop(0,'rgba(91,184,245,'+a+')');
        ng.addColorStop(1,'rgba(91,184,245,0)');
        ctx.beginPath();ctx.arc(p.x,p.y,7,0,2*Math.PI);
        ctx.fillStyle=ng;ctx.fill();
        ctx.beginPath();ctx.arc(p.x,p.y,2.2,0,2*Math.PI);
        ctx.fillStyle='rgba(160,220,255,'+a+')';ctx.fill();
      }
    });
    // Rim
    ctx.beginPath();ctx.arc(cx,cy,R,0,2*Math.PI);
    ctx.strokeStyle='rgba(46,109,164,.45)';ctx.lineWidth=1.5;ctx.stroke();
  }
  function frame(){rot+=0.13;draw(rot);requestAnimationFrame(frame);}
  frame();
})();
</script></body></html>""", height=200)

    with hero_right:
        st.markdown(f"""
<div style="background:#0D1B2E;border-radius:10px;padding:1.5rem 2rem 1.5rem 2.2rem;
            border:1px solid rgba(46,109,164,.3);position:relative;overflow:hidden;
            background-image:linear-gradient(rgba(46,109,164,.05) 1px,transparent 1px),
            linear-gradient(90deg,rgba(46,109,164,.05) 1px,transparent 1px);
            background-size:36px 36px;min-height:200px;display:flex;
            flex-direction:column;justify-content:center;box-sizing:border-box">
  <p style="color:#5BB8F5;font-size:.68rem;font-weight:700;letter-spacing:.14em;
            text-transform:uppercase;margin:0 0 .45rem;font-family:Arial,sans-serif">
    Chinese Development Finance Research Platform</p>
  <div style="color:#FFFFFF;font-size:1.75rem;font-weight:800;line-height:1.1;
              font-family:Arial,sans-serif;margin:0 0 .45rem;letter-spacing:-.02em">
    The BRI Monitor</div>
  <p style="color:#A8C8E8;font-size:.83rem;margin:0 0 .75rem;max-width:460px;
            line-height:1.6;font-family:Arial,sans-serif">
    Integrating AidData infrastructure finance records with policy documents
    and academic literature for evidence-grounded analysis of Chinese overseas
    development finance.</p>
  <div style="display:flex;gap:.55rem;flex-wrap:wrap">
    {"".join(f'<span style="background:rgba(46,109,164,.25);border:1px solid rgba(91,184,245,.28);color:#5BB8F5;font-size:.69rem;font-weight:600;padding:.22rem .65rem;border-radius:20px;font-family:Arial,sans-serif">{lbl}</span>' for lbl in ["4,861 Projects","183 Countries","$1.2 Trillion","2000–2023","42 Documents"])}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Four stat cards ──
    for col, (num, lbl, sub) in zip(st.columns(4), [
        ("4,861","Infrastructure Projects","AidData 2000–2023"),
        ("$1.2 Trillion","Total Commitments","Nominal USD"),
        ("183","Countries Covered","Global scope"),
        ("42","Curated Documents","PDFs indexed"),
    ]):
        with col:
            st.markdown(f"""
<div class="stat-card">
  <p class="sn">{num}</p><p class="sl">{lbl}</p><p class="ss">{sub}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Single CTA
    _, cb, _ = st.columns([1,4,1])
    with cb:
        if st.button("💬  Open Research Assistant →",
                     use_container_width=True, type="primary", key="cta"):
            st.session_state.page = "chat"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Trend chart
    try:
        con = get_con()
        tr  = con.execute("""
            SELECT Commitment_Year AS Year, COUNT(*) AS Projects,
                   ROUND(SUM(Amount_Constant_USD_2023)/1e9,1) AS Financing_Billions
            FROM bri_projects_core
            WHERE Commitment_Year>=2000 AND Amount_Nominal_USD IS NOT NULL
            GROUP BY Commitment_Year ORDER BY Commitment_Year""").df()
        dark = st.session_state.dark_mode
        fc   = "#E6EDF3" if dark else "#0D1B2A"
        fig  = go.Figure()
        fig.add_trace(go.Bar(x=tr["Year"],y=tr["Financing_Billions"],
            name="Financing (USD Bn)",marker_color="#2E6DA4",opacity=.82,yaxis="y"))
        fig.add_trace(go.Scatter(x=tr["Year"],y=tr["Projects"],
            name="Projects",line=dict(color="#F59E0B",width=2.5),
            mode="lines+markers",marker=dict(size=5),yaxis="y2"))
        fig.update_layout(
            title=dict(text="Chinese Infrastructure Finance Commitments (2000–2023)",
                       font=dict(size=13,color=t["navy"])),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial",size=11,color=fc),
            yaxis=dict(title=dict(text="USD Billions",font=dict(color="#2E6DA4",size=10)),
                       gridcolor=t["border"]),
            yaxis2=dict(title=dict(text="Projects",font=dict(color="#F59E0B",size=10)),
                        overlaying="y",side="right"),
            legend=dict(orientation="h",y=1.08,x=.5,xanchor="center"),
            margin=dict(l=15,r=15,t=52,b=25),height=340,
            annotations=[dict(text="Source: AidData (2025) CGLD v1.0",
                x=.99,y=-.09,xref="paper",yref="paper",showarrow=False,
                xanchor="right",font=dict(size=8,color=t["muted"]))]
        )
        st.plotly_chart(fig, use_container_width=True)
    except: pass

    # Two-column info — wrapped in matching section background
    st.markdown(f"""
<div style="background:{t['surface']};border:1px solid {t['border']};
            border-radius:8px;padding:1rem;margin-bottom:.5rem;display:flex;gap:1rem">
  <div style="flex:1;border-right:1px solid {t['border']};padding-right:1rem">
    <p style="color:{t['navy']} !important;font-size:.86rem;font-weight:700;
              margin:0 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid {t['blue']}">
      📊 Data</p>
    <p style="color:{t['text']} !important;font-size:.83rem;line-height:1.6;margin:0">
      <strong>AidData (2025), China's Global Loans and Grants Dataset, Version 1.0</strong>
      — 33,580 loans and grants across 217 countries (2000–2023). Filtered to
      4,861 infrastructure project-level records (~$1.2 trillion in commitments).</p>
    <p style="color:{t['muted']} !important;font-size:.77rem;margin-top:.5rem">
      ⚠️ All figures = <em>commitments</em>, not disbursements.</p>
  </div>
  <div style="flex:1;padding-left:.5rem">
    <p style="color:{t['navy']} !important;font-size:.86rem;font-weight:700;
              margin:0 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid {t['blue']}">
      📚 Documents</p>
    <p style="color:{t['text']} !important;font-size:.83rem;line-height:1.6;margin:0">
      42 curated sources: Chinese government white papers, BRI Forum speeches,
      World Bank and IMF institutional reports, and peer-reviewed academic literature.
      Semantically indexed for research-grade retrieval.</p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="ic" style="margin-bottom:1rem">
  <p class="ct">🔍 What You Can Do</p>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:.8rem">
    <div><p style="color:{t['navy']} !important;font-weight:700;font-size:.82rem;margin:0 0 .22rem">
      💬 Chat</p>
      <p style="font-size:.79rem;margin:0;line-height:1.5">
      Ask research questions combining data and policy documents</p></div>
    <div><p style="color:{t['navy']} !important;font-weight:700;font-size:.82rem;margin:0 0 .22rem">
      📊 Explore</p>
      <p style="font-size:.79rem;margin:0;line-height:1.5">
      Filter and visualise 4,861 projects by country, sector, year</p></div>
    <div><p style="color:{t['navy']} !important;font-weight:700;font-size:.82rem;margin:0 0 .22rem">
      🗺️ Map</p>
      <p style="font-size:.79rem;margin:0;line-height:1.5">
      Choropleth maps of financing and project counts globally</p></div>
    <div><p style="color:{t['navy']} !important;font-weight:700;font-size:.82rem;margin:0 0 .22rem">
      📄 Documents</p>
      <p style="font-size:.79rem;margin:0;line-height:1.5">
      Browse official sources, speeches, and institutional reports</p></div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="pf">Copyright © The BRI Monitor 2026</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — CHAT
# ══════════════════════════════════════════════════════════════════
def show_chat():
    t = T(); inject_css()
    render_navbar()
    index, con, client = initialize()

    # ── Chat header bar ──
    hc_left, hc_right = st.columns([5, 1])
    with hc_left:
        st.markdown(
            '<span class="cov">📅 AidData (2025) CGLD v1.0 · 2000–2023 · '
            '4,861 projects · 183 countries · 42 documents indexed</span>',
            unsafe_allow_html=True)
    with hc_right:
        if st.button("↺ Clear chat", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

    if not st.session_state.messages:
        st.markdown(f"""
<div style="background:{t['surface']};border:1px solid {t['border']};
  border-radius:8px;padding:1.1rem 1.3rem;margin:.5rem 0 .8rem">
  <p style="color:{t['navy']} !important;font-weight:700;font-size:.83rem;
            margin:0 0 .5rem">Research Assistant — How to use</p>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem">
    <div style="font-size:.78rem;color:{t['text']} !important;line-height:1.5">
      <strong style="color:{t['blue']} !important">📊 Data queries</strong><br>
      Ask about volumes, rankings, trends across 4,861 projects and 183 countries.
    </div>
    <div style="font-size:.78rem;color:{t['text']} !important;line-height:1.5">
      <strong style="color:{t['blue']} !important">📄 Document queries</strong><br>
      Ask about policy framing, white papers, BRI Forum speeches, World Bank reports.
    </div>
    <div style="font-size:.78rem;color:{t['text']} !important;line-height:1.5">
      <strong style="color:{t['blue']} !important">🔗 Hybrid queries</strong><br>
      Combine both — e.g. "What does official policy say vs what the data shows on energy?"
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<p style="color:{t['muted']} !important;font-size:.82rem;
          margin:.2rem 0 .5rem;font-weight:600">Sample questions:</p>""",
          unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, q in enumerate(SAMPLE_QUERIES):
            with (c1 if i%2==0 else c2):
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_query = q; st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"]=="assistant": render_msg(msg, idx)
            else: st.markdown(msg["content"])

    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and data…"):
                result = bri_agent(prompt, index, con, client)
            msg = {"role":"assistant","content":result["answer"],
                   "route":result.get("route",""),"sources":result.get("sources",[]),
                   "data":result.get("data"),"followups":result.get("followups",[]),
                   "adjacent":result.get("adjacent",False)}
            render_msg(msg, len(st.session_state.messages))
        st.session_state.messages.append(msg)
        st.rerun()

    if prompt := st.chat_input(
            "Ask a research question about Chinese infrastructure finance…"):
        st.session_state.pending_query = prompt; st.rerun()

    # Scroll-to-top button injected via st.markdown (always visible in chat)
    st.markdown("""
<style>
#bri-top-btn{
  position:fixed;bottom:5rem;right:1.2rem;z-index:99999;
  width:2.6rem;height:2.6rem;border-radius:50%;
  background:#1B3A6B;color:#FFFFFF;
  border:1px solid rgba(255,255,255,.35);font-size:1.15rem;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  box-shadow:0 3px 12px rgba(0,0,0,.5);opacity:.9;transition:all .2s;
  text-decoration:none;line-height:1;font-weight:700
}
#bri-top-btn:hover{opacity:1;transform:scale(1.08);background:#2E6DA4}
</style>
<a id="bri-top-btn" title="Back to top" href="#"
   onclick="(function(){
     var candidates = [
       window.parent.document.querySelector('.main .block-container'),
       window.parent.document.querySelector('[data-testid=stAppViewContainer]'),
       window.parent.document.querySelector('.main'),
       window.parent.document.documentElement
     ];
     for(var i=0;i<candidates.length;i++){
       if(candidates[i]){candidates[i].scrollTo({top:0,behavior:'smooth'});break;}
     }
     return false;
   })()">↑</a>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="pf">Copyright © The BRI Monitor 2026</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_filter_opts():
    con = get_con()
    countries = sorted(con.execute(
        "SELECT DISTINCT Country_of_Activity FROM bri_projects_core "
        "WHERE Country_of_Activity IS NOT NULL ORDER BY 1"
    ).df()["Country_of_Activity"].tolist())
    regions = sorted(con.execute(
        "SELECT DISTINCT Region_of_Activity FROM bri_projects_core "
        "WHERE Region_of_Activity IS NOT NULL ORDER BY 1"
    ).df()["Region_of_Activity"].tolist())
    sectors = sorted(con.execute(
        "SELECT DISTINCT Sector_Name FROM bri_projects_core "
        "WHERE Sector_Name IS NOT NULL ORDER BY 1"
    ).df()["Sector_Name"].tolist())
    flows = sorted(con.execute(
        "SELECT DISTINCT Flow_Type FROM bri_projects_core "
        "WHERE Flow_Type IS NOT NULL ORDER BY 1"
    ).df()["Flow_Type"].tolist())
    income = sorted(con.execute(
        "SELECT DISTINCT WB_Income_Group_Host_Country FROM bri_projects_core "
        "WHERE WB_Income_Group_Host_Country IS NOT NULL ORDER BY 1"
    ).df()["WB_Income_Group_Host_Country"].tolist())
    yr = con.execute(
        "SELECT MIN(Commitment_Year),MAX(Commitment_Year) FROM bri_projects_core"
    ).fetchone()
    am = con.execute(
        "SELECT MIN(Amount_Constant_USD_2023),MAX(Amount_Constant_USD_2023) "
        "FROM bri_projects_core WHERE Amount_Constant_USD_2023 IS NOT NULL"
    ).fetchone()
    return countries, regions, sectors, flows, income, yr, am

def query_filtered(cou, reg, sec, flo, inc, yr0, yr1, am0, am1):
    con = get_con()
    def esc(v): return f"'{str(v).replace(chr(39),chr(39)+chr(39))}'"
    conds = [
        "Amount_Nominal_USD IS NOT NULL",
        f"Commitment_Year BETWEEN {yr0} AND {yr1}",
        # Include rows where Amount_Constant_USD_2023 is NULL OR within the range
        # This avoids silently excluding valid projects that lack 2023-adjusted values
        f"(Amount_Constant_USD_2023 IS NULL OR Amount_Constant_USD_2023 BETWEEN {am0} AND {am1})",
    ]
    if cou: conds.append(f"Country_of_Activity IN ({','.join(map(esc,cou))})")
    if reg: conds.append(f"Region_of_Activity IN ({','.join(map(esc,reg))})")
    if sec: conds.append(f"Sector_Name IN ({','.join(map(esc,sec))})")
    if flo: conds.append(f"Flow_Type IN ({','.join(map(esc,flo))})")
    if inc: conds.append(f"WB_Income_Group_Host_Country IN ({','.join(map(esc,inc))})")
    sql = (
        # Aggregate tranches: group by project identity + key dimensions
        # Show distinct funders per project; retain tranche count for transparency
        f"SELECT "
        f"COALESCE(Display_Title, Title) AS Project,"
        f"Country_of_Activity AS Country,"
        f"Region_of_Activity AS Region,"
        f"MIN(Commitment_Year) AS Year,"
        f"Sector_Name AS Sector,"
        f"Flow_Type,"
        f"ROUND(SUM(Amount_Nominal_USD)/1e6, 2) AS Amount_Nominal_M,"
        f"ROUND(COALESCE(SUM(Amount_Constant_USD_2023), 0)/1e6, 2) AS Amount_2023_M,"
        f"STRING_AGG(DISTINCT Funding_Agencies_Parent, ' | ' "
        f"  ORDER BY Funding_Agencies_Parent) AS Funders,"
        f"WB_Income_Group_Host_Country AS Income_Group,"
        f"COUNT(*) AS Tranches "
        f"FROM bri_projects_core WHERE {' AND '.join(conds)} "
        f"GROUP BY "
        f"COALESCE(Display_Title, Title), Country_of_Activity, Region_of_Activity,"
        f"Sector_Name, Flow_Type, WB_Income_Group_Host_Country "
        f"ORDER BY SUM(Amount_Nominal_USD) DESC NULLS LAST"
    )
    df = con.execute(sql).df()
    return round_df(apply_regions(df))

def de_summary(df):
    t = T()
    if df.empty: st.warning("No projects match current filters."); return
    fin     = df["Amount_2023_M"].sum()/1000
    top_sec = df.groupby("Sector")["Amount_2023_M"].sum().idxmax() if not df.empty else "—"
    top_cou = df.groupby("Country")["Amount_2023_M"].sum().idxmax() if not df.empty else "—"
    funder_col = "Funders" if "Funders" in df.columns else ("Funder" if "Funder" in df.columns else None)
    loan_p  = df["Flow_Type"].eq("Loan").sum()/len(df)*100 if not df.empty else 0
    yspan   = f"{int(df['Year'].min())}–{int(df['Year'].max())}"
    st.markdown(f"""
<div class="mr">
  <div class="mc"><p class="mv">{fin:,.1f}</p>
    <p class="ml">Total Financing</p><p class="ms">USD Bn (2023-adj)</p></div>
  <div class="mc"><p class="mv">{len(df):,}</p>
    <p class="ml">Projects</p><p class="ms">Filtered selection</p></div>
  <div class="mc"><p class="mv">{top_sec.title()[:18]}</p>
    <p class="ml">Top Sector</p><p class="ms">By volume</p></div>
  <div class="mc"><p class="mv">{top_cou[:15]}</p>
    <p class="ml">Top Recipient</p><p class="ms">By volume</p></div>
  <div class="mc"><p class="mv">{loan_p:.0f}%</p>
    <p class="ml">Loan Share</p><p class="ms">{yspan}</p></div>
</div>""", unsafe_allow_html=True)

def show_data_explorer():
    t = T(); inject_css(); render_navbar()
    st.markdown('<p class="sec" style="font-size:1rem">📊 Data Explorer</p>',
                unsafe_allow_html=True)

    try:
        countries,regions,sectors,flows,income,yr,am = load_filter_opts()
    except Exception as e:
        st.error(f"Database error: {e}"); return

    yr0,yr1  = int(yr[0]),int(yr[1])
    am0m,am1m= round(float(am[0])/1e6,2), round(float(am[1])/1e6,2)
    # Clamp am0m to 0 so the slider min is always 0
    am0m = max(0.0, am0m)

    # ── Quick preset chips — each clears all other filters first ──
    PRESETS = {
        "p_sa":  ("🌏 South Asia",  "de_cou",  ["Pakistan","Myanmar","Sri Lanka","Bangladesh","Nepal","Maldives","Afghanistan","India","Bhutan"]),
        "p_en":  ("⚡ Energy",       "de_sec",  ["ENERGY"]),
        "p_tr":  ("🚂 Transport",    "de_sec",  ["TRANSPORT AND STORAGE"]),
        "p_p17": ("📅 Post-2017",    "de_yr",   (2017, yr1)),
        "p_gr":  ("🎁 Grants",       "de_flo",  ["Grant"]),
        "p_af":  ("🌍 Africa",       "de_reg",  ["Africa"]),
    }
    active_preset = st.session_state.get("de_active_preset", None)

    # Build active filter indicator
    active_labels = []
    if st.session_state.get("de_cou"): active_labels.append(f"{len(st.session_state['de_cou'])} countries")
    if st.session_state.get("de_reg"): active_labels.append(", ".join(st.session_state["de_reg"]))
    if st.session_state.get("de_sec"): active_labels.append(", ".join(st.session_state["de_sec"]))
    if st.session_state.get("de_flo"): active_labels.append(", ".join(st.session_state["de_flo"]))

    # Row: label + clear button
    pc_header, pc_clear = st.columns([5, 1])
    with pc_header:
        status = f"Active: {' · '.join(active_labels)}" if active_labels else "No filters active — showing all projects"
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.35rem">
  <span style="font-size:.75rem;font-weight:700;color:{t['muted']};
    text-transform:uppercase;letter-spacing:.05em">Quick filters:</span>
  <span style="font-size:.74rem;color:{t['blue']};font-style:italic">{status}</span>
</div>""", unsafe_allow_html=True)
    with pc_clear:
        if st.button("✕ Clear all", key="de_clr", use_container_width=True):
            for k in ["de_reg","de_cou","de_sec","de_flo","de_inc","de_yr","de_am","de_active_preset"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    # Preset chip buttons — CSS injected to highlight active one
    pc = st.columns(len(PRESETS))
    for col, (key, (label, state_key, val)) in zip(pc, PRESETS.items()):
        with col:
            is_active = (active_preset == key)
            btn_style = "primary" if is_active else "secondary"
            if st.button(label, key=key, use_container_width=True, type=btn_style):
                # Clear ALL filter state, then apply just this preset cleanly
                for k in ["de_reg","de_cou","de_sec","de_flo","de_inc","de_yr","de_am"]:
                    if k in st.session_state: del st.session_state[k]
                st.session_state[state_key] = val
                st.session_state["de_active_preset"] = key
                st.rerun()

    st.markdown("<div style='margin-bottom:.4rem'></div>", unsafe_allow_html=True)

    # ── Sidebar filters ──
    with st.sidebar:
        st.markdown(f'<p class="df">🔍 Filter Panel</p>', unsafe_allow_html=True)
        st.caption("Leave blank = include all.")
        s_reg = st.multiselect("Region",      regions,   key="de_reg")
        s_cou = st.multiselect("Country",     countries, key="de_cou")
        s_sec = st.multiselect("Sector",      sectors,   key="de_sec")
        s_flo = st.multiselect("Flow Type",   flows,     key="de_flo")
        s_inc = st.multiselect("Income Group",income,    key="de_inc")
        st.markdown("**Year Range**")
        yr_sel = st.slider("", yr0, yr1, (yr0, yr1), key="de_yr",
                           label_visibility="collapsed")
        st.markdown("**Amount (USD M, 2023-adj)**")
        am_sel = st.slider("", am0m, am1m, (am0m, am1m), key="de_am",
                           label_visibility="collapsed")
        if st.button("↺ Reset All", use_container_width=True):
            for k in ["de_reg","de_cou","de_sec","de_flo","de_inc","de_yr","de_am","de_active_preset"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
        st.divider()
        st.markdown(f"""<div style="font-size:.71rem;color:{t['muted']} !important;
          line-height:1.6">Source: AidData (2025) CGLD v1.0<br>
          Infrastructure-filtered, project-level.<br>
          Amounts = commitments, not disbursements.</div>""",
          unsafe_allow_html=True)

    df = query_filtered(s_cou,s_reg,s_sec,s_flo,s_inc,
                        yr_sel[0],yr_sel[1],am_sel[0]*1e6,am_sel[1]*1e6)

    st.markdown('<p class="sec">Summary Statistics</p>', unsafe_allow_html=True)
    de_summary(df)
    if df.empty: return

    tab_map, tab_charts, tab_table = st.tabs(["🗺️ Map","📈 Charts","📋 Table"])

    # MAP
    with tab_map:
        st.markdown('<p class="sec">Geographic Distribution</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="ib">{len(df):,} projects across '
                    f'{df["Country"].nunique()} countries. '
                    f'Darker shading = higher value.</div>', unsafe_allow_html=True)
        metric = st.radio("Map metric:",
                          ["Financing (USD Bn)","Project Count"],
                          horizontal=True, key="de_map")
        agg = (df.groupby("Country")
                 .agg(Projects=("Project","count"),
                      Financing_Bn=("Amount_2023_M",
                                    lambda x: round(x.sum()/1000,2)))
                 .reset_index())
        col = "Financing_Bn" if "Financing" in metric else "Projects"
        col_lbl = "Financing (USD Bn)" if col=="Financing_Bn" else "Project Count"
        try:
            fig = px.choropleth(
                agg, locations="Country", locationmode="country names",
                color=col,
                color_continuous_scale=[[0,"#EBF5FB"],[.15,"#AED6F1"],
                    [.35,"#5DADE2"],[.6,"#2E86C1"],[.8,"#1B4F72"],[1,"#0A1931"]],
                labels={col:col_lbl},
                title=f"Chinese Infrastructure {col_lbl} by Country",
                projection="natural earth")
            fig.update_geos(showcoastlines=True, coastlinecolor="#CCCCCC",
                showland=True, landcolor="#F8F9FA",
                showocean=True, oceancolor="#EBF5FB",
                showlakes=False, showframe=False)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=42,b=8), height=430,
                title=dict(font=dict(size=13,color=t["navy"])),
                coloraxis_colorbar=dict(thickness=12,len=.55,
                                        tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Map unavailable: {e}")

        # ── Country Profile Panel ──
        st.markdown('<p class="sec" style="margin-top:.8rem">Country Profile</p>',
                    unsafe_allow_html=True)
        all_countries = sorted(agg["Country"].tolist())
        sel_country = st.selectbox(
            "Select a country to view its financing profile:",
            ["— select —"] + all_countries, key="de_country_profile",
            label_visibility="visible")
        if sel_country != "— select —":
            cp = df[df["Country"] == sel_country]
            # Also get total project count (including those with undisclosed amounts)
            try:
                total_all = get_con().execute(
                    f"SELECT COUNT(*) FROM bri_projects_core "
                    f"WHERE Country_of_Activity = '{sel_country.replace(chr(39), chr(39)+chr(39))}'"
                ).fetchone()[0]
            except: total_all = None

            if not cp.empty:
                total_bn  = cp["Amount_2023_M"].sum() / 1000
                proj_ct   = len(cp)        # projects with disclosed amounts
                yr_range  = f"{int(cp['Year'].min())}–{int(cp['Year'].max())}"
                top_sec   = cp.groupby("Sector")["Amount_2023_M"].sum().idxmax()
                fdr_col = "Funders" if "Funders" in cp.columns else ("Funder" if "Funder" in cp.columns else None)
                top_fund = (cp[fdr_col].value_counts().idxmax()
                            if fdr_col and cp[fdr_col].notna().any() else "—")
                loan_pct  = cp["Flow_Type"].eq("Loan").sum() / proj_ct * 100
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
<div class="mc"><p class="mv">{total_bn:,.2f}</p>
<p class="ml">Total Financing</p><p class="ms">USD Bn · disclosed only</p></div>""",
                        unsafe_allow_html=True)
                with col_b:
                    disclosed_note = f"of {total_all:,} total" if total_all else yr_range
                    st.markdown(f"""
<div class="mc"><p class="mv">{proj_ct:,}</p>
<p class="ml">Projects (with amounts)</p><p class="ms">{disclosed_note}</p></div>""",
                        unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
<div class="mc"><p class="mv">{loan_pct:.0f}%</p>
<p class="ml">Loan Share</p><p class="ms">vs grant · {yr_range}</p></div>""",
                        unsafe_allow_html=True)
                st.markdown(f"""
<div class="ib" style="margin-top:.5rem">
  <strong>Top sector:</strong> {top_sec.title()} &nbsp;·&nbsp;
  <strong>Primary funder:</strong> {top_fund}
  {f' &nbsp;·&nbsp; <em style="color:{t["muted"]};font-size:.78rem">{total_all - proj_ct:,} projects have undisclosed amounts and are excluded from financing totals.</em>' if total_all and total_all > proj_ct else ''}
</div>""", unsafe_allow_html=True)

                # Sector breakdown — show only sectors with actual financing, 1-based index
                sec_breakdown = (cp.groupby("Sector")["Amount_2023_M"]
                                   .sum().div(1000).round(3)
                                   .sort_values(ascending=False)
                                   .reset_index())
                sec_breakdown.columns = ["Sector", "Financing (USD Bn)"]
                # Filter to sectors with real financing (> 0)
                sec_breakdown = sec_breakdown[sec_breakdown["Financing (USD Bn)"] > 0].copy()
                sec_breakdown.index = range(1, len(sec_breakdown) + 1)
                if not sec_breakdown.empty:
                    st.dataframe(sec_breakdown, use_container_width=True,
                                 height=min(250, 45 + len(sec_breakdown) * 36))

                # Pakistan/CPEC note
                if sel_country == "Pakistan":
                    st.info(
                        "**Pakistan / CPEC scope note:** This figure reflects AidData's "
                        "official-finance coverage only. The frequently cited ~$62 billion "
                        "CPEC figure includes announced packages, private investment, and "
                        "non-official finance outside AidData's scope."
                    )

        st.markdown('<p class="sec" style="margin-top:.8rem">Top 20 Countries — by Financing Volume</p>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="ib" style="font-size:.76rem">Showing projects with disclosed '
                    f'financing amounts only. Some projects have undisclosed amounts and are '
                    f'not reflected in these totals.</div>', unsafe_allow_html=True)
        top20 = (agg.sort_values("Financing_Bn",ascending=False)
                    .head(20).reset_index(drop=True))
        top20.index = range(1,len(top20)+1)
        top20.columns = ["Country","Projects (disclosed)","Financing (USD Bn)"]
        st.dataframe(top20, use_container_width=True)

    # CHARTS
    with tab_charts:
        st.markdown('<p class="sec">Analytical Charts</p>', unsafe_allow_html=True)

        # Year range filter specific to charts
        yr_all = sorted(df["Year"].dropna().unique().astype(int).tolist())
        if len(yr_all) >= 2:
            ch_yr = st.select_slider(
                "Chart year range:",
                options=yr_all,
                value=(yr_all[0], yr_all[-1]),
                key="ch_yr_range"
            )
        else:
            ch_yr = (yr_all[0], yr_all[0]) if yr_all else (2000, 2023)
        df_ch = df[(df["Year"] >= ch_yr[0]) & (df["Year"] <= ch_yr[1])]

        ct = st.radio("Chart type:",["Financing Trend","Sector Distribution",
                                "Regional Comparison"],
                      horizontal=True, key="de_ct")
        if ct=="Financing Trend":
            ts = (df_ch.groupby("Year")
                    .agg(Projects=("Project","count"),
                         Financing_Bn=("Amount_2023_M",
                                       lambda x: round(x.sum()/1000,2)))
                    .reset_index())
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ts["Year"],y=ts["Financing_Bn"],
                name="Financing (USD Bn)",marker_color="#2E6DA4",
                opacity=.82,yaxis="y"))
            fig.add_trace(go.Scatter(x=ts["Year"],y=ts["Projects"],
                name="Projects",line=dict(color="#F59E0B",width=2),
                mode="lines+markers",marker=dict(size=5),yaxis="y2"))
            fig.update_layout(
                title=dict(text=f"Financing Trend ({ch_yr[0]}–{ch_yr[1]})",
                           font=dict(size=13,color=t["navy"])),
                yaxis=dict(title=dict(text="USD Bn",
                                      font=dict(color="#2E6DA4",size=10))),
                yaxis2=dict(title=dict(text="Projects",
                                       font=dict(color="#F59E0B",size=10)),
                            overlaying="y",side="right"),
                legend=dict(orientation="h",y=1.1,x=.5,xanchor="center"))
            st.plotly_chart(de_chart_layout(fig), use_container_width=True)

        elif ct=="Sector Distribution":
            sec = (df_ch.groupby("Sector")
                     .agg(Projects=("Project","count"),
                          Financing_Bn=("Amount_2023_M",
                                        lambda x: round(x.sum()/1000,2)))
                     .sort_values("Financing_Bn",ascending=True).reset_index())
            fig = px.bar(sec,x="Financing_Bn",y="Sector",orientation="h",
                         color="Sector",color_discrete_sequence=CHART_PAL,
                         labels={"Financing_Bn":"Financing (USD Bn)","Sector":""},
                         title=f"Sector Distribution ({ch_yr[0]}–{ch_yr[1]})")
            fig.update_layout(showlegend=False,
                              title=dict(font=dict(size=13,color=t["navy"])))
            st.plotly_chart(de_chart_layout(fig,height=max(300,len(sec)*36)),
                            use_container_width=True)

        else:
            reg = (df_ch.groupby("Region")
                     .agg(Projects=("Project","count"),
                          Financing_Bn=("Amount_2023_M",
                                        lambda x: round(x.sum()/1000,2)))
                     .sort_values("Financing_Bn",ascending=False).reset_index())
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Financing (USD Bn)",
                x=reg["Region"],y=reg["Financing_Bn"],
                marker_color="#2E6DA4",yaxis="y"))
            fig.add_trace(go.Scatter(name="Projects",
                x=reg["Region"],y=reg["Projects"],
                mode="markers",
                marker=dict(color="#F59E0B",size=9,symbol="diamond"),yaxis="y2"))
            fig.update_layout(
                title=dict(text=f"Regional Comparison ({ch_yr[0]}–{ch_yr[1]})",
                           font=dict(size=13,color=t["navy"])),
                xaxis=dict(tickangle=-18),
                yaxis=dict(title=dict(text="USD Bn",font=dict(color="#2E6DA4",size=10))),
                yaxis2=dict(title=dict(text="Projects",font=dict(color="#F59E0B",size=10)),
                            overlaying="y",side="right"),
                legend=dict(orientation="h",y=1.1,x=.5,xanchor="center"))
            st.plotly_chart(de_chart_layout(fig), use_container_width=True)

    # TABLE
    with tab_table:
        st.markdown(f'<p class="sec">Project Data ({len(df):,} projects)</p>',
                    unsafe_allow_html=True)
        all_cols = df.columns.tolist()
        def_cols = ["Country","Region","Year","Sector","Flow_Type",
                    "Amount_2023_M","Funders","Tranches","Project"]
        vis = [c for c in def_cols if c in all_cols]
        with st.expander("Choose columns"):
            sel = st.multiselect("", all_cols, default=vis,
                                 key="de_cols", label_visibility="collapsed")
        show = sel if sel else vis
        out  = df[show].copy()
        out.index = range(1, len(out)+1)
        st.dataframe(out, use_container_width=True, height=460)

    st.markdown("""
<div class="pf">Copyright © The BRI Monitor 2026</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════
def show_documents():
    t = T(); inject_css(); render_navbar()
    st.markdown(f"""
<div style="background:#1B3A6B;padding:1.8rem 2rem;border-radius:6px;margin-bottom:1rem">
  <div style="color:#FFFFFF;font-size:1.5rem;font-weight:800;font-family:Arial,sans-serif;margin:0 0 .4rem">📄 Document Library</div>
  <p style="color:#C8DCF0;font-size:.88rem;font-family:Arial,sans-serif;margin:0">Sources indexed for AI-assisted retrieval in the Research Assistant.
  All AI responses draw exclusively from this corpus.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="ib">The Research Assistant draws only from the indexed corpus described below.
No other documents or general knowledge are used. Individual source links will be
published here alongside the companion methods paper.</div>""",
    unsafe_allow_html=True)

    # ── Corpus Categories ──
    cats = [
        ("🇨🇳", "Official Chinese Government Documents",
         "Chinese government white papers, BRI policy frameworks, Ministry of Finance "
         "guidelines, and debt sustainability frameworks. Includes documents from "
         "NDRC, MFA, MOFCOM, and the Ministry of Ecology and Environment.",
         "2015–2023"),
        ("🎙️", "Official Speeches &amp; Forum Communiqués",
         "Xi Jinping keynote speeches from the Belt and Road Forums for International "
         "Cooperation (2017, 2019, 2023) and associated joint communiqués.",
         "2017, 2019, 2023"),
        ("🏛️", "Multilateral &amp; Institutional Reports",
         "World Bank transport corridor analyses, AidData research reports including "
         "Banking on the Belt and Road and How China Lends, Nedopil/GFDC BRI Investment "
         "Reports (2023, 2025, 2026), and IMF and UN assessments.",
         "2019–2026"),
        ("📚", "Academic &amp; Research Literature",
         "Peer-reviewed articles and research institution reports selected by explicit "
         "criteria: Q1/Q2 journal publications or established institutions (AidData, "
         "Lowy Institute, Chatham House); directly addressing Chinese overseas "
         "infrastructure finance or BRI debt patterns; 50+ citations or recognised BRI "
         "scholars. Full titles documented in companion methods paper (forthcoming).",
         "2018–2024"),
    ]
    for icon, cat, desc, period in cats:
        st.markdown(f'<p class="doc-cat">{icon} {cat}</p>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="doc-item">
  <p>{desc}</p>
  <p class="dy">Coverage period: {period}</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:{t['surface']};border:1px solid {t['border']};
  border-left:3px solid {t['blue']};border-radius:6px;padding:.8rem 1rem;margin-top:.8rem">
  <p style="color:{t['navy']} !important;font-weight:700;font-size:.83rem;margin:0 0 .3rem">
    📋 Full Document List — Coming Soon</p>
  <p style="color:{t['text']} !important;font-size:.81rem;margin:0;line-height:1.6">
  Individual document titles, authors, and verified source URLs are being finalised
  and will be published here. The complete corpus with full metadata will be
  documented in the companion methods paper. To request access ahead of publication,
  please use the Contact page.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="pf">Copyright © The BRI Monitor 2026</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT (combined About + Methodology + Limitations)
# ══════════════════════════════════════════════════════════════════
def show_about():
    t = T(); inject_css(); render_navbar()
    st.markdown(f"""
<div style="background:#1B3A6B;padding:1.8rem 2rem;border-radius:6px;margin-bottom:1rem">
  <div style="color:#FFFFFF;font-size:1.5rem;font-weight:800;font-family:Arial,sans-serif;margin:0 0 .4rem">About The BRI Monitor</div>
  <p style="color:#C8DCF0;font-size:.88rem;font-family:Arial,sans-serif;margin:0">Background, scope, and how to cite this tool.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="mb">
  <p class="mt">About This Project</p>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  The BRI Monitor is an independent research platform developed by Sanoop Sajan Koshy,
  PhD Candidate in International Relations at the Indian Institute of Technology Madras.
  It integrates the AidData (2025), China's Global Loans and Grants Dataset, Version 1.0 (2000–2023) with a curated corpus of 42 policy documents, institutional reports,
  and academic sources to support evidence-grounded analysis of Chinese overseas
  development finance patterns and BRI-related policy frameworks.</p>
  <br>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  The tool is designed for academic and policy research purposes. All AI-generated
  outputs draw exclusively from the indexed dataset and document corpus — not from
  general AI training knowledge — and should be treated as analytical starting points
  requiring verification against primary sources. The platform does not represent
  the views of any institution.</p>
  <br>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  The methodology underpinning the dataset harmonisation, document corpus
  construction, and AI system design is documented in a companion methods paper
  (forthcoming). Full technical documentation including data processing decisions,
  corpus selection criteria, and model specifications will be published there.
  Feedback and collaboration enquiries are welcome via the Contact page.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="mb">
  <p class="mt">Data Source & Scope</p>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  <strong>Primary dataset:</strong> AidData (2025), China's Global Loans and Grants Dataset, Version 1.0 — the most comprehensive public record
  of Chinese official overseas finance. The tool uses an infrastructure-filtered
  project-level subset of 4,861 records (~$1.2 trillion in commitments, 2000–2023)
  across 183 countries. All financing figures represent <em>commitments</em>,
  not disbursements. Dataset coverage ends at 2023; for more recent trends the tool
  draws on indexed institutional reports where available.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="mb">
  <p class="mt">Limitations</p>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  Key limitations to keep in mind: approximately 19% of AidData records have
  undisclosed amounts; the document corpus is English-language only; AI retrieval
  is semantic and may miss relevant passages; and BRI project designation is
  contested — this tool uses AidData's broader Chinese official finance coverage,
  not Beijing's own designation list. All AI responses include an analytical note
  stating what the analysis can and cannot establish.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="mb">
  <p class="mt">How to Cite</p>
  <p style="color:{t['text']} !important;font-size:.86rem;line-height:1.8;margin:0">
  If referencing this tool in academic work, please cite the companion methods paper
  (forthcoming) and the underlying dataset:<br><br>
  Koshy, S. S. (forthcoming). <em>The BRI Monitor: A Research Platform for Chinese
  Infrastructure Finance Analysis.</em> [Methods paper, IIT Madras].<br><br>
  AidData. (2025). <em>China's Global Loans and Grants Dataset, Version 1.0.</em>
  Williamsburg, VA: AidData at William &amp; Mary.<br>
  Available at: <a href="https://www.aiddata.org/data/chinas-global-loans-and-grants-dataset-1-0" target="_blank" style="color:#2E6DA4">aiddata.org/data/chinas-global-loans-and-grants-dataset-1-0</a></p>
</div>""", unsafe_allow_html=True)

    st.markdown(f'''
<div class="pf">Copyright © The BRI Monitor 2026</div>''',
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — CONTACT
# ══════════════════════════════════════════════════════════════════
def show_contact():
    t = T(); inject_css(); render_navbar()
    st.markdown(f"""
<div style="background:#1B3A6B;padding:1.8rem 2rem;border-radius:6px;margin-bottom:1rem">
  <div style="color:#FFFFFF;font-size:1.5rem;font-weight:800;font-family:Arial,sans-serif;margin:0 0 .4rem">✉️ Contact &amp; Feedback</div>
  <p style="color:#C8DCF0;font-size:.88rem;font-family:Arial,sans-serif;margin:0">Questions, feedback, collaboration enquiries, or bug reports — all welcome.</p>
</div>""", unsafe_allow_html=True)

    # Full-width form, researcher section below
    st.markdown(f"""
<div class="ic"><p class="ct">Send a Message</p>""", unsafe_allow_html=True)
    with st.form("contact_form", clear_on_submit=True):
        cf1, cf2 = st.columns(2)
        with cf1:
            name  = st.text_input("Name (optional)")
        with cf2:
            email = st.text_input("Email address *")
        subj  = st.selectbox("Subject", [
            "General feedback","Research collaboration","Bug report",
            "Data question","Media / press enquiry","Other"])
        msg   = st.text_area("Message *", height=130,
            placeholder="Your question, feedback, or message…")
        sent  = st.form_submit_button("Send Message",
            use_container_width=True, type="primary")
    if sent:
        if not email or not msg:
            st.error("Email and message are required.")
        else:
            try:
                resp = http_req.post(FORMSPREE, data={
                    "name":name,"email":email,"subject":subj,"message":msg},
                    headers={"Accept":"application/json"})
                if resp.status_code == 200:
                    st.success("Message sent. Thank you — I will respond as soon as possible.")
                else:
                    st.error("Submission failed. Please try again.")
            except Exception as e:
                st.error(f"Could not send: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div class="ic">
  <p class="ct">About the Developer</p>
  <p style="color:{t['text']} !important;font-size:.85rem;line-height:1.75">
  Developed by <strong>Sanoop Sajan Koshy</strong><br>
  PhD Candidate, International Relations<br>
  Indian Institute of Technology Madras</p>
  <br>
  <p style="color:{t['muted']} !important;font-size:.77rem;line-height:1.65">
  This tool is currently in beta. All AI-generated outputs are derived
  exclusively from indexed academic and policy sources and should be treated
  as analytical starting points requiring verification against primary
  materials before citation. Bug reports and usability feedback are
  especially appreciated. Responses within 5–7 working days.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="pf">Copyright © The BRI Monitor 2026</div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
{
    "home":          show_home,
    "chat":          show_chat,
    "data_explorer": show_data_explorer,
    "documents":     show_documents,
    "about":         show_about,
    "contact":       show_contact,
}.get(st.session_state.page, show_home)()
