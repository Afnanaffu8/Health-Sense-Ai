"""HealthSense AI - risk model, report analyzer, doctor finder and local AI/voice assistant."""
import base64
import html
import io
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ML risk model (model.py) and retrieval (rag_engine.py) - degrade gracefully if missing
try:
    from model import predict_risk
    MODEL_ERROR = ""
except Exception as exc:  # e.g. dataset.csv missing
    predict_risk = None
    MODEL_ERROR = str(exc)
try:
    from rag_engine import get_context
except Exception:
    get_context = None
try:
    from model import HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD
except Exception:
    HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD = 60, 30

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".svg")

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Preferred chat models. The app checks Groq's live model list before using them.
# Do not add audio, moderation, embedding, or retired model IDs to this list.
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
GROQ_SKIP_WORDS = ("whisper", "guard", "tts", "orpheus", "embed", "safeguard", "playai", "distil-whisper")
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL_SIZE = "small"  # Optional local fallback; faster-whisper is not required on Streamlit Cloud.
MAX_REPORT_CHARS = 3000

PAGES = ["Home", "Health Risk Analysis", "Medical Report Analyzer", "AI Voice Chatbot"]
PAGE_ICONS = {"Home": "🏠", "Health Risk Analysis": "📊",
              "Medical Report Analyzer": "📄", "AI Voice Chatbot": "🎙️"}
NAV_LABELS = {"Home": "Home", "Health Risk Analysis": "Risk Analysis",
              "Medical Report Analyzer": "Report Analyzer", "AI Voice Chatbot": "AI Chatbot"}
DISCLAIMER = ("HealthSense AI is an educational project. It does not provide a medical "
              "diagnosis. Please consult a qualified doctor.")

EMERGENCY_PATTERNS = [
    r"\bchest pain\b", r"\bchest pressure\b", r"\btrouble breathing\b",
    r"\bdifficulty breathing\b", r"\bcan't breathe\b", r"\bsevere shortness of breath\b",
    r"\bsudden weakness\b", r"\bface droop\b", r"\bslurred speech\b",
    r"\buncontrolled bleeding\b", r"\bunconscious\b", r"\bpassed out\b",
    r"\bseizure\b", r"\bheart attack\b", r"\bstroke\b",
]
CONDITION_KEYWORDS = {
    "diabetes": ["diabet", "glucose", "hba1c", "blood sugar", "sugar", "insulin"],
    "heart": ["heart", "cardiac", "cholesterol", "ecg", "chest pain", "palpitation"],
    "hypertension": ["hypertens", "blood pressure", "bp", "high pressure"],
    "thyroid": ["thyroid", "tsh", "t3", "t4"],
    "kidney": ["kidney", "creatinine", "renal", "urea", "dialysis"],
    "liver": ["liver", "bilirubin", "sgot", "sgpt", "hepatic", "jaundice"],
    "lung": ["lung", "asthma", "breath", "wheez", "cough", "pulmonary"],
    "cancer": ["cancer", "tumor", "tumour", "oncolog", "malignant", "chemotherapy"],
    "fever": ["fever", "temperature", "infection"],
}
DOCTOR_KEYWORDS = {
    "diabetes": ["diabet", "endocrin"], "heart": ["heart", "cardio"],
    "hypertension": ["hypertens", "cardio"], "thyroid": ["thyroid", "endocrin"],
    "kidney": ["kidney", "nephro", "renal"], "liver": ["liver", "hepat", "gastro"],
    "lung": ["lung", "pulmon", "respir", "asthma"], "cancer": ["cancer", "oncolog"],
    "fever": ["fever", "infect", "general"],
}

DOCTOR_KEYWORDS["general"] = ["general physician", "general medicine", "general"]
SPECIALIST_LABEL = {
    "diabetes": "Endocrinologist / Diabetologist", "heart": "Cardiologist",
    "hypertension": "Cardiologist", "thyroid": "Endocrinologist", "kidney": "Nephrologist",
    "liver": "Gastroenterologist / Hepatologist", "lung": "Pulmonologist",
    "cancer": "Oncologist", "fever": "General Physician", "general": "General Physician",
}
CITY_PLACEHOLDER = "Select your city"
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
    "gurgaon": "Gurugram", "new delhi": "Delhi", "trivandrum": "Thiruvananthapuram",
    "mangalore": "Mangaluru", "mysore": "Mysuru", "vizag": "Visakhapatnam", "cochin": "Kochi",
    "pondicherry": "Puducherry", "baroda": "Vadodara", "banaras": "Varanasi",
    "bhubaneshwar": "Bhubaneswar", "calicut": "Kozhikode", "panjim": "Panaji",
}
DOCTOR_INTENT = re.compile(
    r"(suggest|recommend|find|need|want|which|what|best|good|show|list|nearby|near me|book|see|visit|"
    r"consult|where).{0,40}\b(doctor|doctors|dr|hospital|hospitals|specialist|specialists|physician|"
    r"clinic|clinics)\b"
    r"|\b(doctor|doctors|hospital|hospitals|specialist|specialists|clinic|clinics|appointment|referral)\b"
    r".{0,25}\b(near|nearby|in|for|suggest|recommend|list)\b", re.IGNORECASE)

st.set_page_config(page_title="HealthSense AI", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")


# =========================================================
# STATE
# =========================================================
def initialize_state():
    defaults = {
        "page": "Home", "navigation": "Home", "chat_history": [],
        "patient_data": {}, "last_audio_id": None, "report_text": "",
        "report_result": None, "risk_result": None,
        "awaiting_city": False, "chat_city_active": None,
        "chat_session_id": None, "chat_started_at": None,
        "chat_history_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Restore the most recent active conversation once per Streamlit session.
    if not st.session_state.chat_history_loaded:
        saved = load_chat_store()
        active = saved.get("active")
        if active and isinstance(active.get("messages"), list):
            st.session_state.chat_history = active["messages"]
            st.session_state.chat_session_id = active.get("id")
            st.session_state.chat_started_at = active.get("created_at")
        st.session_state.chat_history_loaded = True


CHAT_STORE_PATH = BASE_DIR / "chat_history.json"
MAX_SAVED_CONVERSATIONS = 50


def load_chat_store():
    """Load local conversation history. The file stays on this machine."""
    try:
        if CHAT_STORE_PATH.exists():
            with CHAT_STORE_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                data.setdefault("active", None)
                data.setdefault("conversations", [])
                return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"active": None, "conversations": []}


def save_chat_store(data):
    try:
        CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = CHAT_STORE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temp_path.replace(CHAT_STORE_PATH)
    except OSError as exc:
        st.warning(f"Could not save chat history locally: {exc}")


def save_active_conversation():
    messages = st.session_state.get("chat_history", [])
    if not messages:
        return
    if not st.session_state.get("chat_session_id"):
        st.session_state.chat_session_id = str(uuid.uuid4())
    if not st.session_state.get("chat_started_at"):
        st.session_state.chat_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = load_chat_store()
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "New conversation")
    title = " ".join(str(first_user).split())[:55] or "New conversation"
    data["active"] = {
        "id": st.session_state.chat_session_id,
        "title": title,
        "created_at": st.session_state.chat_started_at,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }
    save_chat_store(data)


def archive_active_conversation():
    save_active_conversation()
    data = load_chat_store()
    active = data.get("active")
    if active and active.get("messages"):
        archived = [c for c in data.get("conversations", []) if c.get("id") != active.get("id")]
        archived.insert(0, active)
        data["conversations"] = archived[:MAX_SAVED_CONVERSATIONS]
    data["active"] = None
    save_chat_store(data)


def clear_current_conversation():
    data = load_chat_store()
    data["active"] = None
    save_chat_store(data)
    st.session_state.chat_history = []
    st.session_state.last_audio_id = None
    st.session_state.chat_session_id = None
    st.session_state.chat_started_at = None
    st.session_state.awaiting_city = False
    st.session_state.chat_city_active = None
    st.session_state.pop("speak_pending", None)


def start_new_conversation():
    archive_active_conversation()
    clear_current_conversation()


def open_saved_conversation(conversation):
    archive_active_conversation()
    st.session_state.chat_history = conversation.get("messages", [])
    st.session_state.chat_session_id = conversation.get("id") or str(uuid.uuid4())
    st.session_state.chat_started_at = conversation.get("created_at")
    st.session_state.last_audio_id = None
    st.session_state.awaiting_city = False
    st.session_state.chat_city_active = None
    st.session_state.pop("speak_pending", None)
    save_active_conversation()


def navigate_to_page(page):
    st.session_state.page = page
    st.session_state.navigation = page


# =========================================================
# STYLES  (no external fonts or files, so the page loads fast)
# =========================================================
CSS = """
<style>
html, body, .stApp, [class*="css"] {
    font-family: "Segoe UI", Inter, -apple-system, Roboto, Arial, sans-serif;
}
.stApp { background:#F2F6FA; color:#273D50; }
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display:none !important; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1240px; }

/* Headings and text */
h1, h2, h3, h4 { color:#12304A !important; letter-spacing:-0.4px; font-weight:750; }
[data-testid="stMain"] .stMarkdown p { color:#273D50; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:#6B7F90 !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color:#243B53 !important; font-weight:600; }
[data-testid="stSpinner"] * { color:#273D50 !important; }
pre, code { background:#EEF3F8 !important; color:#12304A !important; }

/* Sidebar */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0E2A43 0%,#123A57 100%); border-right:none; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stSidebarCollapseButton"] { color:#CFE0EE !important; }
.brand { font-size:24px; font-weight:800; color:#FFFFFF; letter-spacing:-0.8px; }
.brand span { color:#4FD1C5; }
.brand-sub { color:#8FB0C8; font-size:12px; margin-top:2px; }
.side-label { color:#7FA3BF; font-size:11px; font-weight:700; letter-spacing:1.6px; text-transform:uppercase; margin:6px 0 4px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:9px 12px; border-radius:10px; width:100%; margin-bottom:2px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label * { color:#CFE0EE !important; font-weight:550; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:rgba(79,209,197,0.16); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color:#FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display:none; }
.pill { display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.07); border-radius:10px;
        padding:8px 12px; margin-bottom:6px; color:#E6EEF6; font-size:13px; }
.dot { height:9px; width:9px; border-radius:50%; display:inline-block; }
.dot.on { background:#34D399; box-shadow:0 0 0 3px rgba(52,211,153,0.22); }
.dot.off { background:#F87171; box-shadow:0 0 0 3px rgba(248,113,113,0.22); }
.dot.mid { background:#FBBF24; box-shadow:0 0 0 3px rgba(251,191,36,0.25); }
.side-note { color:#7FA3BF; font-size:11.5px; line-height:1.5; margin-top:14px; }

/* Page header */
.eyebrow { color:#087F8C; font-size:11.5px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.page-title { color:#12304A; font-size:32px; font-weight:800; letter-spacing:-1px; margin:2px 0 4px; }
.page-sub { color:#6B7F90; font-size:15px; margin-bottom:18px; }

/* Hero */
.st-key-hero { background:linear-gradient(135deg,#0E2A43 0%,#0B5E7A 60%,#087F8C 100%);
    border-radius:26px; padding:34px 36px; box-shadow:0 18px 40px rgba(14,42,67,0.22); }
.hero-badge { display:inline-block; background:rgba(255,255,255,0.14); color:#BDEDEA; font-size:11px;
    font-weight:800; letter-spacing:1.8px; padding:6px 12px; border-radius:999px; }
.hero-title { color:#FFFFFF; font-size:44px; font-weight:800; line-height:1.12; letter-spacing:-1.8px; margin:14px 0 12px; }
.hero-title span { color:#7EE8DC; }
.hero-text { color:#C9DCEB; font-size:16px; line-height:1.75; margin-bottom:18px; }
.chip { display:inline-block; background:rgba(255,255,255,0.12); color:#E3F1F7; font-size:12.5px;
    padding:6px 12px; border-radius:999px; margin:10px 8px 0 0; }
.st-key-hero_cta button { background:#FFFFFF !important; border:none !important; }
.st-key-hero_cta button * { color:#0E2A43 !important; font-weight:700 !important; }
.st-key-anatomy { background:#FFFFFF; border-radius:22px; padding:12px; text-align:center;
    box-shadow:0 10px 30px rgba(0,0,0,0.22); }
.st-key-anatomy img { max-height:520px; object-fit:contain; }
.img-note { color:#8A9BAA; font-size:11px; }

/* Cards */
[class*="st-key-svc_"] { background:#FFFFFF; border:1px solid #E2EAF0; border-radius:18px; padding:22px;
    box-shadow:0 6px 20px rgba(23,50,77,0.05); transition:0.2s ease; }
[class*="st-key-svc_"]:hover { box-shadow:0 12px 28px rgba(23,50,77,0.11); transform:translateY(-2px); }
.svc-icon { font-size:26px; margin-bottom:8px; }
.svc-title { color:#12304A; font-size:18px; font-weight:750; margin-bottom:6px; }
.svc-text { color:#6B7F90; font-size:13.5px; line-height:1.6; min-height:42px; }
.panel { background:#FFFFFF; border:1px solid #E2EAF0; border-radius:18px; padding:22px;
    box-shadow:0 6px 20px rgba(23,50,77,0.04); }
.metric { background:#FFFFFF; border:1px solid #E2EAF0; border-radius:16px; padding:16px 18px;
    box-shadow:0 5px 18px rgba(23,50,77,0.04); height:100%; }
.metric-label { color:#6B7F90; font-size:12px; font-weight:700; letter-spacing:0.8px; text-transform:uppercase; }
.metric-value { color:#12304A; font-size:26px; font-weight:800; letter-spacing:-0.6px; margin-top:4px; }
.metric-sub { font-size:13px; font-weight:650; margin-top:2px; color:#6B7F90; }
.metric-sub.good { color:#0F8F66; } .metric-sub.mid { color:#B7791F; } .metric-sub.bad { color:#C53030; }
.callout { border-radius:12px; padding:13px 16px; margin:10px 0; font-size:14.5px; line-height:1.55; }
.section { color:#12304A; font-size:20px; font-weight:750; letter-spacing:-0.4px; margin:22px 0 10px; }

/* Risk bar */
.riskbar-track { position:relative; display:flex; height:14px; border-radius:999px; overflow:hidden; margin-top:6px; }
.seg.low { background:#34D399; } .seg.mod { background:#FBBF24; } .seg.high { background:#F87171; }
.riskbar { position:relative; padding-top:2px; }
.marker { position:absolute; top:-4px; width:4px; height:22px; background:#12304A; border-radius:3px; transform:translateX(-50%); }
.riskbar-scale { display:flex; justify-content:space-between; color:#8A9BAA; font-size:11.5px; margin-top:4px; }

/* Doctors */
.doctor-card { background:#FFFFFF; border:1px solid #E2EAF0; border-left:4px solid #087F8C;
    border-radius:12px; padding:14px 16px; margin-bottom:10px; }
.doctor-name { color:#12304A; font-weight:700; font-size:16px; }
.doctor-spec { color:#087F8C; font-size:13px; font-weight:600; margin-left:8px; }
.doctor-meta { color:#5B6F80; font-size:13.5px; line-height:1.8; }
.doctor-meta a { color:#087F8C; font-weight:650; text-decoration:none; }
.badge { display:inline-block; background:#087F8C; color:#FFFFFF; font-size:11px; font-weight:700; padding:3px 9px; border-radius:999px; margin-right:6px; }
.badge.alt { background:#E8F3F6; color:#087F8C; }
.spec-title { color:#12304A; font-weight:750; font-size:16px; margin:14px 0 6px; }

/* Chat */
.st-key-chatbox { background:#F8FBFD; border:1px solid #E2EAF0; border-radius:16px; }
.chat-row { display:flex; margin:10px 4px; }
.chat-row.user { justify-content:flex-end; }
.bubble { max-width:80%; padding:12px 16px; border-radius:16px; line-height:1.6; font-size:14.5px; }
.bubble.assistant { background:#FFFFFF; color:#1F3347; border:1px solid #E2EAF0; border-bottom-left-radius:4px; }
.bubble.user { background:linear-gradient(135deg,#087F8C,#0B6E99); color:#FFFFFF; border-bottom-right-radius:4px; }
.voice-panel { background:linear-gradient(145deg,#FFFFFF,#EAF5F8); border:1px solid #D8EAF0;
    border-radius:20px; padding:22px; text-align:center; margin-bottom:8px; }
.voice-heading { color:#12304A; font-size:19px; font-weight:750; }
.voice-sub { color:#6B7F90; font-size:13px; }

/* Inputs: always white box with dark text */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"],
div[data-baseweb="select"] > div { background:#FFFFFF !important; border:1px solid #CBD5E1 !important; border-radius:10px !important; }
input, textarea { background:#FFFFFF !important; color:#12304A !important; -webkit-text-fill-color:#12304A !important;
    caret-color:#12304A !important; }
input::placeholder, textarea::placeholder { color:#7A8B9A !important; -webkit-text-fill-color:#7A8B9A !important; opacity:1 !important; }
textarea:disabled, input:disabled { -webkit-text-fill-color:#273D50 !important; opacity:1 !important; }
div[data-baseweb="select"] *, [data-testid="stNumberInput"] * { color:#12304A !important; }
[data-testid="stNumberInput"] button { background:#EEF3F8 !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background:#FFFFFF !important; color:#12304A !important; }
div[data-baseweb="popover"] li:hover { background:#E8F3F6 !important; }
[data-testid="stFileUploaderDropzone"] { background:#F8FBFD !important; border:1.5px dashed #9FB8C8 !important; border-radius:14px !important; }
[data-testid="stFileUploaderDropzone"] * { color:#243B53 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#FFFFFF !important; border:1px solid #087F8C !important; }
[data-testid="stFileUploaderDropzone"] button * { color:#087F8C !important; }
[data-testid="stExpander"], [data-testid="stExpander"] details {
    background:#FFFFFF !important; border:1px solid #E2EAF0 !important; border-radius:14px !important; }
[data-testid="stExpander"] summary { background:#FFFFFF !important; border-radius:14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:#12304A !important; fill:#12304A !important; }
[data-testid="stForm"] { background:#FFFFFF; border:1px solid #E2EAF0; border-radius:18px; padding:20px; }

/* Buttons: text always readable */
div.stButton > button, div.stFormSubmitButton > button, [data-testid="stDownloadButton"] > button {
    min-height:44px; border-radius:11px; font-weight:650; background:#FFFFFF !important;
    border:1px solid #CBD5E1 !important; color:#12304A !important; transition:0.18s ease; }
div.stButton > button *, div.stFormSubmitButton > button *, [data-testid="stDownloadButton"] > button * { color:inherit !important; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    border-color:#087F8C !important; color:#087F8C !important; }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
    background:linear-gradient(135deg,#087F8C,#0B6E99) !important; border:none !important; color:#FFFFFF !important; }
div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
    background:#12304A !important; color:#FFFFFF !important; }
@media(max-width:800px){ .hero-title{font-size:31px;} .st-key-hero{padding:22px;} }
</style>
"""


def add_styles():
    st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# UI BUILDING BLOCKS
# =========================================================
def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
                f'<div class="page-title">{html.escape(title)}</div>'
                f'<div class="page-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{html.escape(title)}</div>', unsafe_allow_html=True)


def callout(kind, text):
    palette = {"error": ("#FDECEC", "#E5484D", "#7A1F23", "⚠️"),
               "warning": ("#FFF6E5", "#F5A524", "#6B4300", "⚠️"),
               "success": ("#E7F7EF", "#19A879", "#0F5C43", "✅"),
               "info": ("#E8F3FA", "#2D8CC4", "#134563", "ℹ️")}
    bg, bar, fg, icon = palette[kind]
    st.markdown(f'<div class="callout" style="background:{bg};border-left:4px solid {bar};color:{fg}">'
                f'{icon} {html.escape(text)}</div>', unsafe_allow_html=True)


def metric_card(label, value, sub="", tone=""):
    sub_html = f'<div class="metric-sub {tone}">{html.escape(sub)}</div>' if sub else ""
    st.markdown(f'<div class="metric"><div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{html.escape(str(value))}</div>{sub_html}</div>',
                unsafe_allow_html=True)


def tone_for(level):
    return {"High Risk": "bad", "Moderate Risk": "mid", "Low Risk": "good"}.get(level, "")


def risk_bar(pct):
    pct = max(0.0, min(100.0, float(pct)))
    low, mod = MODERATE_RISK_THRESHOLD, HIGH_RISK_THRESHOLD - MODERATE_RISK_THRESHOLD
    high = 100 - HIGH_RISK_THRESHOLD
    st.markdown(
        f'<div class="riskbar"><div class="riskbar-track">'
        f'<div class="seg low" style="width:{low}%"></div><div class="seg mod" style="width:{mod}%"></div>'
        f'<div class="seg high" style="width:{high}%"></div></div>'
        f'<div class="marker" style="left:{pct}%"></div>'
        f'<div class="riskbar-scale"><span>0%</span><span>{MODERATE_RISK_THRESHOLD}%</span>'
        f'<span>{HIGH_RISK_THRESHOLD}%</span><span>100%</span></div></div>', unsafe_allow_html=True)


def show_risk_banner(level):
    if level == "High Risk":
        callout("error", "High risk detected. Please consult a doctor.")
    elif level == "Moderate Risk":
        callout("warning", "Moderate risk. Consider a check-up and healthy lifestyle changes.")
    elif level == "Low Risk":
        callout("success", "Low estimated risk. Maintain a healthy lifestyle.")


def md_to_html(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    return text.replace("\n", "<br>")


def chat_bubble(role, content):
    st.markdown(f'<div class="chat-row {role}"><div class="bubble {role}">{md_to_html(content)}</div></div>',
                unsafe_allow_html=True)


# =========================================================
# ANATOMY IMAGE (resized once and cached, so pages stay fast)
# =========================================================
def find_anatomy_file():
    if not ASSETS_DIR.is_dir():
        return None
    files = [p for p in sorted(ASSETS_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    preferred = [p for p in files if any(w in p.stem.lower() for w in ("anatomy", "human", "body"))]
    raster_first = sorted(preferred or files, key=lambda p: p.suffix.lower() == ".svg")
    return raster_first[0] if raster_first else None


@st.cache_data(show_spinner=False)
def load_anatomy(path_str, mtime):
    """Return (kind, data). Large images are shrunk to max 900px; transparent borders trimmed."""
    path = Path(path_str)
    if path.suffix.lower() == ".svg":
        return "svg", path.read_bytes()
    from PIL import Image
    img = Image.open(path)
    img.load()
    if img.mode in ("RGBA", "LA"):
        box = img.getchannel("A").getbbox()
        if box:
            img = img.crop(box)
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "png", buf.getvalue()


def show_anatomy():
    path = find_anatomy_file()
    if path is None:
        st.markdown('<div class="svc-text" style="padding:40px 10px">Add an image to the '
                    '<b>assets</b> folder, for example <b>assets/human_anatomy.png</b>.</div>',
                    unsafe_allow_html=True)
        return
    try:
        kind, data = load_anatomy(str(path), path.stat().st_mtime)
        if kind == "svg":
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="max-height:520px;max-width:100%" '
                        'alt="Human anatomy">', unsafe_allow_html=True)
        else:
            st.image(data, width="stretch")
        st.markdown(f'<div class="img-note">{html.escape(path.name)}</div>', unsafe_allow_html=True)
    except Exception as exc:
        st.markdown(f'<div class="svc-text">Could not load {html.escape(path.name)}: '
                    f'{html.escape(str(exc))}</div>', unsafe_allow_html=True)


# =========================================================
# DATA HELPERS
# =========================================================
@st.cache_data
def load_doctors():
    path = BASE_DIR / "doctors.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path).fillna("")
    needed = {"Name", "Specialization", "Disease", "Hospital", "City", "Contact", "MapLink"}
    return df if not (needed - set(df.columns)) else None


def detect_conditions(text):
    t = text.lower()
    return [c for c, words in CONDITION_KEYWORDS.items()
            if any(re.search(rf"\b{re.escape(w)}", t) for w in words)]


def valid_cities(df):
    if df is None:
        return []
    return sorted(c for c in df["City"].unique() if c and not re.fullmatch(r"[\d\s+\-()]+", str(c)))


def detect_city(text):
    """Find a known city (or an alias such as Bangalore) mentioned in the text."""
    t = text.lower()
    for alias, city in CITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return city
    for city in sorted(valid_cities(load_doctors()), key=len, reverse=True):
        if re.search(rf"\b{re.escape(city.lower())}\b", t):
            return city
    return None


def pick_doctors(df, keywords, city, limit=3):
    """Best-matching hospital departments in one city: right specialty first, specialty centres boosted."""
    if df is None or not keywords or not city or city == CITY_PLACEHOLDER:
        return pd.DataFrame()
    pattern = "|".join(re.escape(k) for k in keywords)
    sub = df[df["City"] == city]
    spec = sub["Specialization"].str.lower().str.contains(pattern, na=False)
    dis = sub["Disease"].str.lower().str.contains(pattern, na=False)
    sub = sub[spec | dis].copy()
    if sub.empty:
        return sub
    sub["_score"] = (spec[sub.index].astype(int) * 2 + dis[sub.index].astype(int)
                     + (sub["Type"] == "Specialty centre").astype(int) * 2 if "Type" in sub
                     else spec[sub.index].astype(int) * 2 + dis[sub.index].astype(int))
    return sub.sort_values("_score", ascending=False, kind="stable").head(limit)


def doctor_card_html(row, badge=None):
    link = str(row.get("MapLink", ""))
    map_html = (f'<a href="{html.escape(link)}" target="_blank">View on map ↗</a>'
                if link.startswith("http") else "")
    badge_html = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
    type_html = f'<span class="badge alt">{html.escape(str(row["Type"]))}</span>' if row.get("Type") else ""
    state = f", {row['State']}" if row.get("State") else ""
    return (f'<div class="doctor-card"><div><span class="doctor-name">{html.escape(str(row["Name"]))}</span>'
            f'<span class="doctor-spec">{html.escape(str(row["Specialization"]))}</span></div>'
            f'<div class="doctor-meta">🏥 <b>{html.escape(str(row["Hospital"]))}</b>, '
            f'{html.escape(str(row["City"]))}{html.escape(state)}<br>'
            f'📞 {html.escape(str(row["Contact"]))}<br>{map_html}</div>'
            f'<div style="margin-top:6px">{badge_html}{type_html}</div></div>')


def city_selector(key, label="Your city"):
    cities = valid_cities(load_doctors())
    return st.selectbox(label, [CITY_PLACEHOLDER] + cities, key=key)


def add_specialty(out, cond, reason):
    label = SPECIALIST_LABEL[cond]
    if all(sp["label"] != label for sp in out):
        out.append({"label": label, "keywords": DOCTOR_KEYWORDS[cond], "reason": reason})


def risk_specialties(res):
    """Which specialist(s) fit this risk result (at most two)."""
    out = []
    if res["glucose"] >= 126:
        add_specialty(out, "diabetes", f"Your glucose ({res['glucose']:.0f} mg/dL) is in the diabetic range.")
    elif res["level"] in ("Moderate Risk", "High Risk"):
        add_specialty(out, "diabetes", f"The ML model estimates a {res['pct']}% diabetes risk.")
    if res["systolic"] >= 140 or res["diastolic"] >= 90:
        add_specialty(out, "hypertension",
                      f"Your blood pressure ({res['systolic']}/{res['diastolic']}) is in the high range.")
    if not out and res["score"] >= 3:
        add_specialty(out, "general", "Several of your measurements are elevated.")
    return out[:2]


def condition_specialties(conditions, bp_high=False, risk_elevated=False):
    out = []
    for c in conditions:
        add_specialty(out, c, f"Your text mentions {c}.")
    if bp_high:
        add_specialty(out, "hypertension", "The blood pressure in your report is in the high range.")
    if risk_elevated and not out:
        add_specialty(out, "diabetes", "The ML model estimates an elevated risk.")
    return out[:2]


def doctor_recommendations(specialties, key):
    """Show the specialist type, then the top hospital match (plus a few alternatives) for the chosen city."""
    section("👨‍⚕️ Recommended doctor")
    for sp in specialties:
        st.markdown(f'<div class="callout" style="background:#E8F3FA;border-left:4px solid #2D8CC4;color:#134563">'
                    f'<b>See a {html.escape(sp["label"])}.</b> {html.escape(sp["reason"])}</div>',
                    unsafe_allow_html=True)
    city = city_selector(key, "Choose your city to see matching hospitals")
    if city == CITY_PLACEHOLDER:
        callout("info", "Select your city above and I will show the most suitable hospital for you.")
        return
    df = load_doctors()
    if df is None:
        callout("info", "Add a valid doctors.csv to see doctor suggestions.")
        return
    for sp in specialties:
        picks = pick_doctors(df, sp["keywords"], city, limit=3)
        if picks.empty:
            callout("info", f"No {sp['label']} is listed in {city} yet. Try a nearby city.")
            continue
        st.markdown(f'<div class="spec-title">{html.escape(sp["label"])} in {html.escape(city)}</div>',
                    unsafe_allow_html=True)
        st.markdown(doctor_card_html(picks.iloc[0], "Top suggestion"), unsafe_allow_html=True)
        if len(picks) > 1:
            with st.expander(f"More {sp['label']} options in {city}"):
                for _, row in picks.iloc[1:].iterrows():
                    st.markdown(doctor_card_html(row), unsafe_allow_html=True)
    st.caption("Suggestions are matched by specialty and city from the app's hospital directory. "
               "They are not a quality ranking. Check availability with the hospital.")


def risk_estimate(age, bmi, glucose, diastolic):
    """Run the ML model. Returns (percentage, level, error)."""
    if predict_risk is None:
        return None, None, f"Risk model unavailable: {MODEL_ERROR}"
    try:
        pct, level = predict_risk(age, bmi, glucose, diastolic)
        return pct, level, ""
    except ValueError as exc:
        return None, None, str(exc)


# =========================================================
# LOCAL AI (OLLAMA) + RAG
# =========================================================
@st.cache_resource(show_spinner=False)
def http():
    """HTTP session that ignores proxy settings, so localhost always works."""
    session = requests.Session()
    session.trust_env = False
    return session


@st.cache_data(ttl=10, show_spinner=False)
def ollama_models():
    """Installed Ollama models, or None if Ollama is offline."""
    try:
        r = http().get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        pass
    return None


def ollama_state():
    """'offline', 'no_model' (running but nothing installed) or 'ready'."""
    models = ollama_models()
    if models is None:
        return "offline"
    return "ready" if models else "no_model"


def active_model():
    models = ollama_models()
    if not models:
        return None
    return OLLAMA_MODEL if OLLAMA_MODEL in models else models[0]


PULL_HINT = "Open a terminal and run:  ollama pull llama3.2:3b  (a small, fast model), then try again."


def system_prompt(context_hits, doctor_note=""):
    prompt = """You are HealthSense AI, a professional and empathetic health information assistant.
Help users understand health concerns and prepare for conversations with qualified clinicians.

COMMUNICATION: respond in clear English with simple words; be calm and respectful;
ask one relevant follow-up question at a time; do not repeat answered questions.
Ask about: main symptom, when it started, duration, severity, worsening, other symptoms,
existing conditions, current medicines and allergies when relevant.

SAFETY: never claim to diagnose; never claim the user is definitely healthy; never invent
test results; never prescribe medicines or doses; never advise stopping prescribed medicines;
no false reassurance. For chest pain, severe breathing difficulty, stroke symptoms, seizures or
uncontrolled bleeding, advise immediate emergency care.

When enough information is available, organise the answer as: 1. Summary of your concern
2. Possible explanations (with uncertainty) 3. Recommended next steps 4. When to seek urgent care.
You are an educational assistant, not a doctor.

DOCTORS: this app has its own hospital and doctor directory and displays matching hospitals under your
reply once the user shares a city. NEVER tell the user to search Google, ask friends, or get a referral
in order to find a doctor. Name the type of specialist that fits, and if they want a doctor, say the
matching hospitals are listed below (or ask which city they are in)."""
    if doctor_note:
        prompt += "\n\n" + doctor_note
    if context_hits:
        refs = "\n".join(f"- ({h['source'].replace('_', ' ')}) {h['text']}" for h in context_hits)
        prompt += ("\n\nREFERENCE INFORMATION (use it when relevant, do not contradict it, "
                   "and do not invent facts beyond it):\n" + refs)
    return prompt


def groq_key():
    """Groq API key from an environment variable or Streamlit secrets (never hard-code it)."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return str(key).strip()


@st.cache_data(ttl=600, show_spinner=False)
def groq_live_models(key, models_url):
    """Ask Groq which models are available right now (empty list if the call fails)."""
    try:
        r = requests.get(models_url, headers={"Authorization": f"Bearer {key}"}, timeout=10)
        if r.ok:
            return [m["id"] for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError):
        pass
    return []


def groq_models():
    """Return only live chat models where possible; ignore a stale custom model setting."""
    custom = os.environ.get("GROQ_MODEL", "").strip()
    if not custom:
        try:
            custom = str(st.secrets.get("GROQ_MODEL", "")).strip()
        except Exception:
            custom = ""
    url = os.environ.get("GROQ_API_URL", GROQ_URL).replace("/chat/completions", "/models")
    live = groq_live_models(groq_key(), url)
    if live:
        ordered = ([custom] if custom in live else [])
        ordered += [m for m in GROQ_MODELS if m in live and m not in ordered]
        ordered += [m for m in live if m not in ordered and not any(w in m.lower() for w in GROQ_SKIP_WORDS)]
        return ordered[:8]
    # If the model-list endpoint is temporarily unavailable, try a safe production default
    # and then the other known production models. A bad ID is skipped by ask_groq().
    return list(dict.fromkeys(([custom] if custom else []) + GROQ_MODELS))


def ai_provider():
    """'ollama' (local) when ready, else 'groq' (cloud) when a key exists, else None."""
    if ollama_state() == "ready":
        return "ollama"
    return "groq" if groq_key() else None


def ask_groq(messages, num_predict=450):
    """Groq cloud chat (OpenAI-compatible). Returns (ok, text)."""
    headers = {"Authorization": f"Bearer {groq_key()}", "Content-Type": "application/json"}
    url = os.environ.get("GROQ_API_URL", GROQ_URL)
    last_error = ""
    tried = groq_models()
    for model in tried:
        payload = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": num_predict}
        if "gpt-oss" in model:  # reasoning model: keep thinking short so the answer is not cut off
            payload["reasoning_effort"] = "low"
            payload["max_tokens"] = num_predict + 800
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            last_error = "Groq took too long to answer."
            continue
        except requests.exceptions.RequestException as exc:
            return False, f"Could not reach Groq: {exc}"
        if r.status_code == 401:
            return False, "Groq rejected the API key (401). Check GROQ_API_KEY."
        if not r.ok:  # rate limit (429), retired model (404/400), ... -> try the next model
            last_error = f"{r.status_code}: {r.text[:200]}"
            continue
        try:
            text = r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError):
            last_error = "unexpected reply format"
            continue
        if text:
            return True, text
    return False, (f"Groq could not answer right now (tried: {', '.join(tried) or 'none'}; "
                   f"last error {last_error}). Please try again in a minute.")


def ask_ai(messages, num_predict=450):
    """Use local Ollama when available, otherwise Groq if a key is set. Returns (ok, text)."""
    provider = ai_provider()
    if provider == "ollama":
        return ask_ollama(messages, num_predict)
    if provider == "groq":
        return ask_groq(messages, num_predict)
    if ollama_state() == "no_model":
        return False, "Ollama is running but no model is installed. " + PULL_HINT
    return False, ("No AI is connected. Start Ollama, or add a free Groq key "
                   "(GROQ_API_KEY) to use cloud AI.")


def ask_ollama(messages, num_predict=450):
    """Return (ok, text)."""
    state = ollama_state()
    if state == "offline":
        return False, "Ollama is not running. Open the Ollama app and try again."
    if state == "no_model":
        return False, "Ollama is running but no model is installed. " + PULL_HINT
    model = active_model()
    payload = {"model": model, "messages": messages, "stream": False, "think": False,
               "keep_alive": "30m",
               "options": {"temperature": 0.4, "num_ctx": 4096, "num_predict": num_predict}}
    try:
        r = http().post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=240)
        if r.status_code == 400 and "think" in r.text.lower():  # model has no 'think' option
            payload.pop("think")
            r = http().post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=240)
        if not r.ok:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            return False, f"Ollama error ({r.status_code}) with model '{model}': {detail[:300]}"
        text = r.json().get("message", {}).get("content", "").strip()
        return (True, text) if text else (False, "The model returned an empty reply. Please try again.")
    except requests.exceptions.Timeout:
        return False, ("The model is taking too long (large models are slow on CPU). "
                       "Try a smaller one: ollama pull llama3.2:3b")
    except requests.exceptions.RequestException as exc:
        return False, f"AI connection error: {exc}"


def retrieve(query, top_k=3):
    return get_context(query, top_k=top_k) if get_context else []


def has_emergency_warning(message):
    return any(re.search(p, message, flags=re.IGNORECASE) for p in EMERGENCY_PATTERNS)


def chat_doctor_plan(user_text):
    """Work out whether to list doctors for this chat message. Returns a dict."""
    df = load_doctors()
    msg_city = detect_city(user_text)
    if msg_city:
        st.session_state.chat_city_active = msg_city
    selected = st.session_state.get("chat_city_sel")
    city = msg_city or st.session_state.chat_city_active or (
        selected if selected and selected != CITY_PLACEHOLDER else None)
    all_user_text = " ".join(m["content"] for m in st.session_state.chat_history if m["role"] == "user")
    conditions = detect_conditions(all_user_text[-1500:])
    intent = bool(DOCTOR_INTENT.search(user_text)) or (st.session_state.awaiting_city and bool(msg_city))
    specialties = condition_specialties(conditions) or ([] if not intent else
                                                         condition_specialties(["general"]))
    picks, note = [], ""
    if intent and df is not None:
        labels = " and ".join(sp["label"] for sp in specialties)
        if city:
            per = 2 if len(specialties) == 1 else 1
            for sp in specialties:
                chosen = pick_doctors(df, sp["keywords"], city, limit=per)
                picks += chosen.to_dict("records")
            lines = [f"- {r['Hospital']} ({r['Name']})" for r in picks] or ["- (none listed in this city)"]
            note = (f"DOCTOR DIRECTORY MATCHES in {city} (shown to the user under your reply):\n"
                    + "\n".join(lines) + f"\nSuggested specialist: {labels}. Say these hospitals are listed "
                    "below. Do not invent other hospitals or phone numbers.")
        else:
            note = (f"The user wants a doctor but has not shared a city. Suggested specialist: {labels}. "
                    "Briefly say which specialist fits and ask which city they are in.")
    return {"intent": intent, "city": city, "picks": picks, "note": note}


def add_chat_message(message):
    message = message.strip()
    if not message:
        return
    st.session_state.chat_history.append({"role": "user", "content": message})
    doctors = []
    if has_emergency_warning(message):
        answer = ("Your message mentions a possible emergency warning sign. Please contact your "
                  "local emergency number or seek emergency medical care immediately. Do not "
                  "wait for this chat. If possible, ask someone nearby to help.")
        st.session_state.speak_pending = True
    else:
        plan = chat_doctor_plan(message)
        hits = retrieve(message)
        msgs = [{"role": "system", "content": system_prompt(hits, plan["note"])}]
        msgs += [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.chat_history[-14:]]
        ok, answer = ask_ai(msgs)
        if not ok and ai_provider() is None and hits:
            answer = ("No AI model is connected right now, so here is what my health knowledge base says:\n\n"
                      + "\n\n".join(h["text"] for h in hits[:2]))
        st.session_state.speak_pending = ok
        doctors = plan["picks"]
        if plan["intent"] and not plan["city"]:
            st.session_state.awaiting_city = True
            answer += "\n\n📍 Tell me your city (or choose it above the chat) and I'll list matching hospitals."
        elif plan["city"] and plan["intent"]:
            st.session_state.awaiting_city = False
    entry = {"role": "assistant", "content": answer}
    if doctors:
        entry["doctors"] = doctors
    st.session_state.chat_history.append(entry)
    save_active_conversation()


# =========================================================
# SPEECH (imported only when used, so startup stays fast)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper using pip.") from exc
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)


def transcribe_audio(audio_bytes):
    """Transcribe microphone audio with Groq in cloud mode, or local Whisper if installed."""
    key = groq_key()
    if key:
        headers = {"Authorization": f"Bearer {key}"}
        files = {"file": ("healthsense_recording.webm", audio_bytes, "audio/webm")}
        data = {"model": "whisper-large-v3-turbo", "language": "en", "response_format": "json"}
        try:
            response = requests.post(
                GROQ_TRANSCRIPTION_URL, headers=headers, files=files, data=data, timeout=90
            )
            if response.ok:
                transcript = response.json().get("text", "").strip()
                if transcript:
                    return transcript
                raise RuntimeError("Groq transcription returned an empty transcript.")
            detail = response.text[:300]
            raise RuntimeError(f"Groq voice transcription failed ({response.status_code}): {detail}")
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Groq voice transcription: {exc}") from exc

    # Optional offline/local transcription. Install faster-whisper only if you need this mode.
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        segments, _ = model.transcribe(path, language="en", beam_size=3, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        Path(path).unlink(missing_ok=True)


def clean_for_speech(text):
    text = re.sub(r"[*_#`>]", "", text)
    text = re.sub(r"(?m)^\s*[-•]\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()[:1500]


def build_speech_html(text, autoplay):
    """Small player that reads the reply aloud with the browser's built-in voice (offline)."""
    payload = json.dumps(clean_for_speech(text)).replace("<", "\\u003c")
    return f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <button id="play" style="background:#087F8C;color:#fff;border:none;border-radius:10px;padding:9px 16px;font-weight:600;cursor:pointer">🔊 Listen to reply</button>
  <button id="stop" style="background:#fff;color:#12304A;border:1px solid #CBD5E1;border-radius:10px;padding:9px 14px;font-weight:600;cursor:pointer">⏹ Stop</button>
  <span id="msg" style="color:#6B7F90;font-size:12px"></span>
</div>
<script>
const text = {payload};
const AUTOPLAY = {"true" if autoplay else "false"};
const msg = document.getElementById("msg");
let started = false;
function pickVoice() {{
  const v = window.speechSynthesis.getVoices();
  return v.find(x => /en-IN/i.test(x.lang)) || v.find(x => /^en/i.test(x.lang)) || null;
}}
function speak() {{
  if (!("speechSynthesis" in window)) {{ msg.textContent = "Speech is not supported in this browser."; return; }}
  started = true;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  const voice = pickVoice();
  if (voice) {{ u.voice = voice; u.lang = voice.lang; }} else {{ u.lang = "en-US"; }}
  u.rate = 1.0;
  u.onstart = () => {{ msg.textContent = "Speaking..."; }};
  u.onend = () => {{ msg.textContent = ""; }};
  u.onerror = () => {{ msg.textContent = "Click 'Listen to reply' to hear it."; }};
  window.speechSynthesis.speak(u);
}}
document.getElementById("play").onclick = speak;
document.getElementById("stop").onclick = () => {{ window.speechSynthesis.cancel(); msg.textContent = ""; }};
if (AUTOPLAY) {{
  if ("speechSynthesis" in window) {{
    window.speechSynthesis.onvoiceschanged = () => {{ if (!started) speak(); }};
    setTimeout(() => {{ if (!started) speak(); }}, 500);
  }}
}}
</script>
"""


def render_speech(text, autoplay):
    page = build_speech_html(text, autoplay)
    if hasattr(st, "iframe"):  # newer Streamlit
        st.iframe(page, height=60)
    else:
        components.html(page, height=60)


def get_mic_recorder():
    try:
        from streamlit_mic_recorder import mic_recorder
        return mic_recorder
    except ImportError:
        return None


# =========================================================
# SCREENING + REPORT EXTRACTION
# =========================================================
def calculate_bmi(weight, height):
    if weight <= 0 or height <= 0:
        raise ValueError("Enter valid height and weight.")
    return weight / ((height / 100) ** 2)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "mid"
    if bmi < 25:
        return "Healthy range", "good"
    if bmi < 30:
        return "Overweight", "mid"
    return "Obese", "bad"


def threshold_screen(age, bmi, glucose, systolic, diastolic):
    score = 0
    score += 1 if age >= 60 else 0
    score += 1 if bmi >= 30 else 0
    score += 2 if glucose >= 200 else (1 if glucose >= 126 else 0)
    if systolic >= 180 or diastolic >= 120:
        score += 3
    elif systolic >= 140 or diastolic >= 90:
        score += 2
    elif systolic >= 130 or diastolic >= 80:
        score += 1
    return ("Several elevated" if score >= 3 else "Few elevated"), score


def extract_report_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf.") from exc
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
    if name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Install python-docx.") from exc
        doc = Document(io.BytesIO(data))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        rows = [" | ".join(c.text.strip() for c in r.cells) for t in doc.tables for r in t.rows]
        return "\n".join(paras + rows).strip()
    if name.endswith(".txt"):
        return data.decode("utf-8-sig", errors="replace").strip()
    raise ValueError("Unsupported file format.")


def extract_values(text):
    """Find age, BMI, glucose and blood pressure (systolic/diastolic) in report text."""
    def num(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None
    bp = re.search(r"(?:blood pressure|bp)[^0-9]{0,15}(\d{2,3})\s*/\s*(\d{2,3})", text, re.IGNORECASE)
    return {
        "age": num(r"age[^0-9]{0,10}(\d{1,3})"),
        "bmi": num(r"bmi[^0-9]{0,15}(\d{1,2}\.?\d*)"),
        "glucose": num(r"glucose[^0-9]{0,15}(\d{2,3}\.?\d*)"),
        "systolic": float(bp.group(1)) if bp else num(r"(?:blood pressure|bp)[^0-9]{0,15}(\d{2,3})"),
        "diastolic": float(bp.group(2)) if bp else None,
    }


def clamp(value, low, high, default):
    return float(min(max(value, low), high)) if value is not None else float(default)


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():
    st.sidebar.markdown('<div class="brand">HealthSense <span>AI</span></div>'
                        '<div class="brand-sub">Intelligent healthcare workspace</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown('<div class="side-label" style="margin-top:22px">Main menu</div>',
                        unsafe_allow_html=True)
    st.sidebar.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed",
                     format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
    st.session_state.page = st.session_state.navigation

    provider = ai_provider()
    state = ollama_state()
    if provider == "ollama":
        ai_dot, ai_text = "on", "Local AI connected (Ollama)"
    elif provider == "groq":
        ai_dot, ai_text = "on", "Cloud AI connected (Groq)"
    elif state == "no_model":
        ai_dot, ai_text = "mid", "Ollama running, no model"
    else:
        ai_dot, ai_text = "off", "AI offline (demo mode)"
    model_on = predict_risk is not None
    st.sidebar.markdown('<div class="side-label" style="margin-top:22px">System status</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="pill"><span class="dot {ai_dot}"></span>{ai_text}</div>'
        f'<div class="pill"><span class="dot {"on" if model_on else "off"}"></span>'
        f'Risk model {"ready" if model_on else "unavailable"}</div>'
        '<div class="side-note">Health information only. Not a substitute for professional care.</div>',
        unsafe_allow_html=True)


# =========================================================
# PAGES
# =========================================================
def render_home():
    with st.container(key="hero"):
        left, right = st.columns([1.1, 0.9], vertical_alignment="center", gap="large")
        with left:
            st.markdown("""<span class="hero-badge">AI-POWERED HEALTHCARE</span>
                <div class="hero-title">Welcome to<br><span>HealthSense AI</span></div>
                <div class="hero-text">Your intelligent health companion: a trained risk model, medical
                report insights, doctor suggestions and a private AI assistant you can talk to.</div>""",
                        unsafe_allow_html=True)
            st.button("Start Health Assessment  →", key="hero_cta", type="primary",
                      on_click=navigate_to_page, args=("Health Risk Analysis",))
            st.markdown('<span class="chip">🔒 Private local AI</span><span class="chip">🎙️ English voice input</span>'
                        '<span class="chip">🩺 Doctor finder</span>', unsafe_allow_html=True)
        with right:
            with st.container(key="anatomy"):
                show_anatomy()

    section("Explore your health tools")
    cards = [
        ("01", "📊", "Health Risk Analysis", "BMI, an ML risk estimate and doctor suggestions when risk is raised."),
        ("02", "📄", "Medical Report Analyzer", "Read reports, estimate risk and get plain-language insights."),
        ("03", "🎙️", "AI Voice Chatbot", "Talk or type with your local AI health assistant."),
    ]
    for col, (num, icon, title, desc) in zip(st.columns(3, gap="medium"), cards):
        with col, st.container(key=f"svc_{num}"):
            st.markdown(f'<div class="svc-icon">{icon}</div><div class="svc-title">{title}</div>'
                        f'<div class="svc-text">{desc}</div>', unsafe_allow_html=True)
            st.button("Open  →", key=f"open_{num}", on_click=navigate_to_page, args=(title,))

    section("Your workspace")
    data = st.session_state.patient_data
    if data:
        a, b, c = st.columns(3, gap="medium")
        with a:
            metric_card("Last BMI", f"{data['bmi']:.1f}")
        with b:
            metric_card("Age", f"{data['age']} years")
        with c:
            metric_card("ML risk estimate", data.get("ml_level") or "Not available")
    else:
        callout("info", "Your latest screening overview will appear here after you complete a health assessment.")
    st.caption(DISCLAIMER)


def render_health_analysis():
    page_header("HEALTH SCREENING", "Health Risk Analysis",
                "Enter your measurements to get an ML risk estimate and a threshold-based screen.")
    with st.form("health_assessment_form"):
        section("Personal measurements")
        left, right = st.columns(2, gap="large")
        with left:
            age = st.number_input("Age (years)", 1, 120, 35)
            weight = st.number_input("Weight (kg)", 20.0, 300.0, 65.0, step=0.5)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.5)
        with right:
            glucose = st.number_input("Blood glucose (mg/dL)", 40.0, 500.0, 100.0)
            systolic = st.number_input("Systolic pressure (mmHg)", 70, 250, 120)
            diastolic = st.number_input("Diastolic pressure (mmHg)", 40, 150, 80)
        submitted = st.form_submit_button("Calculate risk", type="primary", width="stretch")

    if submitted:
        bmi = calculate_bmi(weight, height)
        pct, level, error = risk_estimate(age, bmi, glucose, diastolic)
        screen, score = threshold_screen(age, bmi, glucose, systolic, diastolic)
        st.session_state.patient_data = {
            "age": age, "bmi": bmi, "glucose": glucose, "systolic": systolic,
            "diastolic": diastolic, "result": screen, "ml_pct": pct, "ml_level": level}
        st.session_state.risk_result = {
            "bmi": bmi, "pct": pct, "level": level, "error": error, "screen": screen,
            "score": score, "glucose": glucose, "systolic": systolic, "diastolic": diastolic}

    res = st.session_state.risk_result
    if not res:
        return

    section("Results")
    cat, cat_tone = bmi_category(res["bmi"])
    a, b, c = st.columns(3, gap="medium")
    with a:
        metric_card("Body Mass Index", f"{res['bmi']:.1f}", cat, cat_tone)
    with b:
        metric_card("ML risk estimate", f"{res['pct']}%" if res["pct"] is not None else "N/A",
                    res["level"] or "", tone_for(res["level"]))
    with c:
        metric_card("Threshold screen", res["screen"], f"{res['score']} indicator point(s)",
                    "bad" if res["score"] >= 3 else "good")
    if res["pct"] is not None:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        risk_bar(res["pct"])
    if res["error"]:
        callout("warning", res["error"])
    else:
        show_risk_banner(res["level"])

    specialties = risk_specialties(res)
    if specialties:
        doctor_recommendations(specialties, "risk_city")

    callout("info", "The ML estimate comes from a model trained on the public Pima Indians Diabetes "
                    "dataset (diabetes only). It is an educational estimate, not a diagnosis.")


def render_report_analyzer():
    page_header("DOCUMENT WORKSPACE", "Medical Report Analyzer",
                "Upload a report or type symptoms. The app finds key values, estimates risk, "
                "explains the findings and suggests doctors.")
    uploaded = st.file_uploader("Choose a report", type=["pdf", "docx", "txt"])
    typed = st.text_area("Or type your report / symptoms here:", height=90)
    st.caption("Supported: PDF, DOCX, TXT. Scanned PDFs need OCR and are not supported yet.")
    if st.button("Read report", type="primary", width="stretch"):
        try:
            text = extract_report_text(uploaded) if uploaded else typed.strip()
            if not text:
                callout("warning", "No readable text found. Upload a file or type something.")
            else:
                st.session_state.report_text = text
                st.session_state.report_result = None
        except Exception as exc:
            callout("error", str(exc))

    text = st.session_state.report_text
    if not text:
        return

    with st.expander(f"Extracted text ({len(text):,} characters)"):
        st.text_area("Report content", text, height=220, disabled=True)

    section("Detected values (edit if needed)")
    found = extract_values(text)
    missing = [k for k, v in found.items() if v is None]
    if missing:
        callout("info", f"Not found in the report: {', '.join(missing)}. Please check the defaults below.")
    c1, c2, c3, c4, c5 = st.columns(5)
    age = c1.number_input("Age", 1, 120, int(clamp(found["age"], 1, 120, 35)))
    bmi = c2.number_input("BMI", 10.0, 70.0, clamp(found["bmi"], 10, 70, 24.0))
    glucose = c3.number_input("Glucose", 40.0, 500.0, clamp(found["glucose"], 40, 500, 100.0))
    systolic = c4.number_input("Systolic", 70, 250, int(clamp(found["systolic"], 70, 250, 120)))
    diastolic = c5.number_input("Diastolic", 40, 150, int(clamp(found["diastolic"], 40, 150, 80)))

    if st.button("Analyze report", type="primary", width="stretch"):
        pct, level, error = risk_estimate(age, bmi, glucose, diastolic)
        conditions = detect_conditions(text)
        hits = retrieve(" ".join(conditions) if conditions else text[:200], top_k=3)
        summary, summary_error = None, ""
        prompt = ("Summarise this patient text in plain language, give general lifestyle tips, "
                  "and say when to see a doctor. Do not diagnose or suggest medicines.\n\n"
                  f"Patient text:\n{text[:MAX_REPORT_CHARS]}")
        with st.spinner("HealthSense AI is analysing the report..."):
            ok, answer = ask_ai([{"role": "system", "content": system_prompt(hits)},
                                     {"role": "user", "content": prompt}], num_predict=500)
        if ok:
            summary = answer
        else:
            summary_error = answer
        st.session_state.report_result = {
            "pct": pct, "level": level, "error": error, "conditions": conditions,
            "hits": hits, "summary": summary, "summary_error": summary_error,
            "bp_high": systolic >= 140 or diastolic >= 90}

    res = st.session_state.report_result
    if not res:
        return

    section("Analysis")
    a, b = st.columns(2, gap="medium")
    with a:
        metric_card("ML risk estimate", f"{res['pct']}%" if res["pct"] is not None else "N/A",
                    res["level"] or "", tone_for(res["level"]))
    with b:
        metric_card("Conditions mentioned", ", ".join(res["conditions"]) or "None detected")
    if res["pct"] is not None:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        risk_bar(res["pct"])
    if res["error"]:
        callout("warning", res["error"])
    else:
        show_risk_banner(res["level"])

    if res["summary"]:
        section("🤖 AI summary")
        st.markdown(f'<div class="panel">{md_to_html(res["summary"])}</div>', unsafe_allow_html=True)
    else:
        callout("info", "AI summary unavailable: " + (res.get("summary_error") or "no AI is connected") + " The rest of the analysis still works.")

    if res["hits"]:
        section("📚 Related health information")
        for h in res["hits"]:
            st.markdown(f'<div class="panel" style="margin-bottom:10px;padding:14px 18px">'
                        f'{html.escape(h["text"])}<div class="img-note" style="margin-top:6px">'
                        f'Source: {html.escape(h["source"].replace("_", " "))}</div></div>',
                        unsafe_allow_html=True)

    specialties = condition_specialties(
        res["conditions"], res["bp_high"], res["level"] in ("Moderate Risk", "High Risk"))
    if specialties:
        doctor_recommendations(specialties, "report_city")

    if res["summary"]:
        st.download_button("Download summary",
                           data=f"Report summary\n\n{res['summary']}\n\n{DISCLAIMER}",
                           file_name="report_summary.txt", mime="text/plain")
    st.caption(DISCLAIMER)


def render_chatbot():
    page_header("PERSONAL AI HEALTH ASSISTANT", "AI Health Assistant",
                "Describe your health concern naturally. Answers use an AI model "
                "and the built-in health knowledge base.")
    provider = ai_provider()
    if provider == "ollama":
        callout("success", f"Local AI connected (Ollama model: {active_model()})")
    elif provider == "groq":
        callout("success", "Cloud AI connected (Groq). Your messages are sent to Groq to generate replies.")
    elif ollama_state() == "no_model":
        callout("warning", "Ollama is running but no model is installed. " + PULL_HINT)
    else:
        callout("warning", "No AI is connected, so I answer from my built-in health knowledge base. "
                           "Start Ollama, or add a free Groq key (GROQ_API_KEY) for full AI replies.")
    with st.expander("🔧 AI connection check"):
        st.write("Active AI:", {"ollama": "Ollama (local)", "groq": "Groq (cloud)"}.get(ai_provider(), "none"))
        st.write("Installed Ollama models:", ", ".join(ollama_models() or []) or "none")
        if st.button("Send a test message", key="ai_test"):
            started = time.time()
            with st.spinner("Testing... the first call can take up to a minute while the model loads."):
                ok, answer = ask_ai([{"role": "user", "content": "Reply with one short sentence."}], 40)
            took = round(time.time() - started, 1)
            callout("success" if ok else "error", f"{'Works' if ok else 'Failed'} ({took}s): {answer}")
    st.toggle("🔊 Speak replies aloud", value=True, key="voice_on")
    city_selector("chat_city_sel", "Your city (optional, so I can list hospitals near you)")

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "assistant", "content": (
            "Hello! I'm your HealthSense AI assistant. Tell me what health concern you would "
            "like help with. You can speak or type your message.")})
        save_active_conversation()

    # Saved conversations are stored locally in chat_history.json.
    with st.expander("📚 Conversation History", expanded=False):
        saved_data = load_chat_store()
        saved_chats = saved_data.get("conversations", [])
        if saved_chats:
            for index, conversation in enumerate(saved_chats):
                title = conversation.get("title") or "Saved conversation"
                date_label = conversation.get("updated_at") or conversation.get("created_at") or ""
                history_col, open_col = st.columns([4, 1])
                with history_col:
                    st.markdown(f"**{html.escape(title)}**")
                    st.caption(date_label)
                with open_col:
                    if st.button("Open", key=f"open_saved_chat_{index}", width="stretch"):
                        open_saved_conversation(conversation)
                        st.rerun()
        else:
            st.caption("No saved conversations yet. Start a chat and use New Conversation to save it here.")

    section("Conversation")
    with st.container(height=400, key="chatbox"):
        for m in st.session_state.chat_history:
            chat_bubble(m["role"], m["content"])
            for i, row in enumerate(m.get("doctors", [])):
                st.markdown(doctor_card_html(row, "Suggested" if i == 0 else None), unsafe_allow_html=True)

    history = st.session_state.chat_history
    last_reply = next((m for m in reversed(history) if m["role"] == "assistant"), None)
    if st.session_state.get("voice_on") and last_reply and len(history) > 1:
        render_speech(last_reply["content"], st.session_state.pop("speak_pending", False))
    else:
        st.session_state.pop("speak_pending", None)

    voice_description = ("Record your message and stop when finished. Groq transcribes it when a cloud key is set; "
                         "otherwise an optional local Whisper installation is used.")
    st.markdown('<div style="height:12px"></div><div class="voice-panel"><div class="voice-heading">'
                'Speak to your assistant</div><div class="voice-sub">' + html.escape(voice_description)
                + '</div></div>', unsafe_allow_html=True)

    mic_recorder = get_mic_recorder()
    if mic_recorder is None:
        callout("info", "Voice input needs streamlit-mic-recorder in requirements.txt. "
                        "You can still type below while the app is rebuilding.")
    else:
        try:
            audio = mic_recorder(start_prompt="🎙️ Start recording", stop_prompt="⏹️ Stop recording",
                                 just_once=True, use_container_width=True, format="webm",
                                 key="healthsense_voice_recorder")
            if audio and audio.get("bytes"):
                audio_id = audio.get("id")
                if audio_id != st.session_state.last_audio_id:
                    st.session_state.last_audio_id = audio_id
                    with st.spinner("Converting your speech to text..."):
                        transcript = transcribe_audio(audio["bytes"])
                    if transcript:
                        with st.spinner("HealthSense AI is preparing a response..."):
                            add_chat_message(transcript)
                        st.rerun()
                    else:
                        callout("warning", "No speech was recognized. Please try again.")
        except Exception as exc:
            callout("error", "Voice input failed. Check the voice service configuration or type your message instead.")
            with st.expander("Technical details"):
                st.code(str(exc))

    section("Type your message")
    with st.form("healthsense_text_chat", clear_on_submit=True):
        typed = st.text_area("Message", placeholder="Describe your symptoms or health concern...",
                             height=85, label_visibility="collapsed")
        send = st.form_submit_button("Send message", type="primary", width="stretch")
    if send and typed.strip():
        with st.spinner("HealthSense AI is thinking... (the first reply can take up to a minute while the model loads)"):
            add_chat_message(typed)
        st.rerun()
    clear_col, new_col = st.columns(2)
    with clear_col:
        if st.button("🧹 Clear Conversation", width="stretch"):
            clear_current_conversation()
            st.rerun()
    with new_col:
        if st.button("＋ New Conversation", type="primary", width="stretch"):
            start_new_conversation()
            st.rerun()
    st.caption(DISCLAIMER + " For urgent symptoms, seek emergency medical care.")


def render_back_home():
    """Small 'Back to Home' button at the top of every page except Home."""
    if st.session_state.page != "Home":
        left, _ = st.columns([1, 4])
        with left:
            st.button("←  Back to Home", key="back_home_top", on_click=navigate_to_page,
                      args=("Home",), width="stretch")


# =========================================================
# MAIN
# =========================================================
def main():
    initialize_state()
    add_styles()
    render_sidebar()
    render_back_home()
    {"Home": render_home, "Health Risk Analysis": render_health_analysis,
     "Medical Report Analyzer": render_report_analyzer,
     "AI Voice Chatbot": render_chatbot}[st.session_state.page]()


if __name__ == "__main__":
    main()