# ================================================================
# AXIS - Penerjemah Cerdas Bahasa Daerah Sulawesi Tenggara
# Deployment dengan Streamlit + Hugging Face Hub
# ================================================================

import os
import re
import gc
import torch
import streamlit as st

# ================================================================
# IMPORT
# ================================================================

try:
    from transformers.models.mbart import MBartForConditionalGeneration
except ImportError:
    from transformers import MBartForConditionalGeneration

from indobenchmark import IndoNLGTokenizer
from huggingface_hub import snapshot_download

# ================================================================
# KONFIGURASI
# ================================================================

# Ganti dengan repo Hugging Face kamu, format: "username/nama-repo"
HF_REPO_ID = "fadhhhhhhhltn/axis-indobart"

MODEL_DIR = "./model_axis_indobart"
MAX_SRC_LENGTH = 64
MAX_TGT_LENGTH = 64

# ================================================================
# DOWNLOAD MODEL DARI HUGGING FACE HUB
# ================================================================

def download_model_from_hf():
    """Download seluruh file model dari Hugging Face Hub.

    snapshot_download otomatis melakukan caching, jadi kalau file
    sudah ada dan valid, tidak akan didownload ulang. Ini juga yang
    menghindari masalah quota seperti pada Google Drive.
    """
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=MODEL_DIR,
        )
        return True
    except Exception as e:
        st.error(f"Gagal mengunduh model dari Hugging Face: {e}")
        return False

# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Load model"""

    if not download_model_from_hf():
        st.stop()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load tokenizer
    try:
        tokenizer = IndoNLGTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=False,
            local_files_only=True,
        )
    except Exception:
        tokenizer = IndoNLGTokenizer.from_pretrained(
            "indobenchmark/indobart-v2",
            use_fast=False,
        )
        new_tokens = ["<2buton>", "<2muna>", "<2tolaki>", "<2indo>"]
        existing = tokenizer.all_special_tokens
        tokens_to_add = [t for t in new_tokens if t not in existing]
        if tokens_to_add:
            tokenizer.add_special_tokens({
                "additional_special_tokens": tokens_to_add
            })

    # Load model
    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True,
    )

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')
        model = model.float()
        model = model.to(device)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer, device

# ================================================================
# FUNGSI POST-PROCESSING
# ================================================================

def detect_case_pattern(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "lower"
    if all(c.isupper() for c in letters):
        return "upper"
    if text and text[0].isupper():
        return "sentence"
    return "lower"

def clean_translation_output(text):
    if not text:
        return text

    for tag in ["<2buton>", "<2muna>", "<2tolaki>", "<2indo>"]:
        text = text.replace(tag, "").strip()

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    cleaned = []
    prev = None
    repeat_count = 0
    for w in words:
        if w == prev:
            repeat_count += 1
            if repeat_count < 2:
                cleaned.append(w)
        else:
            cleaned.append(w)
            repeat_count = 0
        prev = w

    return " ".join(cleaned).strip()

def restore_casing(translated_text_lower, original_input_text):
    pattern = detect_case_pattern(original_input_text)
    text = translated_text_lower.strip()

    if pattern == "upper":
        return text.upper()
    elif pattern == "sentence" and len(text) > 0:
        return text[0].upper() + text[1:]
    return text

# ================================================================
# KONFIGURASI BAHASA
# ================================================================

LANG_DISPLAY = {
    "indo": "Bahasa Indonesia",
    "buton": "Bahasa Buton",
    "muna": "Bahasa Muna",
    "tolaki": "Bahasa Tolaki",
}

LANG_TAGS = {
    "indo": "<2indo>",
    "buton": "<2buton>",
    "muna": "<2muna>",
    "tolaki": "<2tolaki>",
}

# ================================================================
# FUNGSI TERJEMAHAN
# ================================================================

def translate_text(text, source_lang, target_lang, model, tokenizer, device):

    valid_langs = {"indo", "buton", "muna", "tolaki"}
    if source_lang not in valid_langs or target_lang not in valid_langs:
        raise ValueError(f"Bahasa harus salah satu dari: {valid_langs}")

    if source_lang == target_lang:
        raise ValueError("Bahasa sumber dan target tidak boleh sama")

    original_text = text
    text_lower = text.lower().strip()
    target_tag = LANG_TAGS[target_lang]
    source_with_tag = f"{target_tag} {text_lower}"

    inputs = tokenizer(
        source_with_tag,
        return_tensors="pt",
        max_length=MAX_SRC_LENGTH,
        truncation=True,
    ).to(device)

    input_token_len = inputs["input_ids"].shape[1]

    if target_lang == "indo":
        max_new_tok = min(max(10, int(input_token_len * 1.2)), MAX_TGT_LENGTH)
        min_new_tok = max(3, int(input_token_len * 0.3))
        length_pen = 0.8
    else:
        max_new_tok = min(max(12, int(input_token_len * 1.5)), MAX_TGT_LENGTH)
        min_new_tok = max(3, int(input_token_len * 0.4))
        length_pen = 1.0

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tok,
            min_new_tokens=min_new_tok,
            num_beams=5,
            length_penalty=length_pen,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True,
            forced_eos_token_id=tokenizer.eos_token_id,
        )

    result_raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    result_clean = clean_translation_output(result_raw)
    result_final = restore_casing(result_clean, original_text)

    return result_final

# ================================================================
# FUNGSI SWAP BAHASA
# ================================================================

def swap_languages():
    current_src = st.session_state.source_lang
    current_tgt = st.session_state.target_lang
    st.session_state.source_lang = current_tgt
    st.session_state.target_lang = current_src
    st.session_state.result_text = ""

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================

st.set_page_config(
    page_title="AXIS - Penerjemah Bahasa Daerah",
    page_icon="",
    layout="wide"
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 3px solid #e94560;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: 2px;
    }
    .main-header h1 .highlight {
        color: #e94560;
    }
    .main-header p {
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: 0.25rem;
    }
    .main-header .sub {
        font-size: 0.9rem;
        color: #adb5bd;
        letter-spacing: 4px;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border: 2px solid #dee2e6;
        min-height: 120px;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .result-box.has-result {
        background: #f0f7ff;
        border-color: #e94560;
    }
    .result-box .placeholder {
        color: #adb5bd;
    }
    .stButton button {
        background: #e94560;
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: #d63851;
        color: white;
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.3);
    }
    .stButton button:disabled {
        background: #adb5bd;
        box-shadow: none;
    }
    .stButton button.secondary {
        background: #f0f2f5;
        color: #1a1a2e;
    }
    .stButton button.secondary:hover {
        background: #e0e0e0;
        box-shadow: none;
    }
    .stButton button.swap-btn {
        background: transparent;
        color: #6c757d;
        border: 2px solid #dee2e6;
        padding: 0.5rem;
        font-size: 1.2rem;
        min-height: 44px;
    }
    .stButton button.swap-btn:hover {
        border-color: #e94560;
        color: #e94560;
        background: #fef0f2;
        box-shadow: none;
    }
    .info-text {
        color: #6c757d;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #e94560;
    }
    .sidebar-info p {
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .lang-label {
        font-weight: 600;
        color: #1a1a2e;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .stSelectbox label {
        font-weight: 600;
        color: #1a1a2e;
    }
    .stTextArea label {
        font-weight: 600;
        color: #1a1a2e;
    }
    .stAlert {
        padding: 0.75rem 1rem;
        border-radius: 8px;
    }
    .sidebar-langs {
        margin: 0.5rem 0 0 0;
        padding: 0;
        list-style: none;
    }
    .sidebar-langs li {
        padding: 0.2rem 0;
        color: #495057;
        font-size: 0.9rem;
    }
    .sidebar-langs li::before {
        content: "- ";
        color: #e94560;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 0.5rem;
    }
    .footer-links span {
        color: #adb5bd;
        font-size: 0.8rem;
    }
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .footer-links { flex-direction: column; gap: 0.25rem; align-items: center; }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD MODEL
# ================================================================

with st.spinner("Memuat model penerjemah..."):
    model, tokenizer, device = load_model_and_tokenizer()

# ================================================================
# INISIALISASI SESSION STATE
# ================================================================

if "source_lang" not in st.session_state:
    st.session_state.source_lang = "indo"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "buton"
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "result_text" not in st.session_state:
    st.session_state.result_text = ""

# ================================================================
# HEADER
# ================================================================

st.markdown("""
<div class="main-header">
    <h1>AXIS</h1>
    <p>Penerjemah Cerdas Bahasa Daerah Sulawesi Tenggara</p>
    <div class="sub">Buton  |  Muna  |  Tolaki</div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("### TENTANG AXIS")
    st.markdown("""
    <div class="sidebar-info">
        <p>AXIS adalah penerjemah cerdas untuk bahasa daerah
        Sulawesi Tenggara yang dibangun menggunakan model IndoBART.</p>
        <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #6c757d;">
            Model fine-tuned dengan data parallel corpus
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### BAHASA DIDUKUNG")
    st.markdown("""
    <ul class="sidebar-langs">
        <li>Bahasa Indonesia</li>
        <li>Bahasa Buton</li>
        <li>Bahasa Muna</li>
        <li>Bahasa Tolaki</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.75rem; color: #adb5bd; text-align: center;">
        Version 1.0.0
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT
# ================================================================

col1, col2, col3 = st.columns([2, 0.6, 2])

with col1:
    st.markdown('<div class="lang-label">DARI</div>', unsafe_allow_html=True)
    source_lang = st.selectbox(
        "Pilih bahasa sumber",
        options=list(LANG_DISPLAY.keys()),
        format_func=lambda x: LANG_DISPLAY[x],
        key="source_lang",
        label_visibility="collapsed"
    )

with col2:
    st.write("")
    st.write("")
    st.button(
        "\u21c4",
        on_click=swap_languages,
        help="Tukar bahasa sumber dan target",
        use_container_width=True,
        key="swap_button"
    )

with col3:
    st.markdown('<div class="lang-label">KE</div>', unsafe_allow_html=True)
    target_lang = st.selectbox(
        "Pilih bahasa target",
        options=list(LANG_DISPLAY.keys()),
        format_func=lambda x: LANG_DISPLAY[x],
        key="target_lang",
        label_visibility="collapsed"
    )

st.markdown("---")

col_input, col_output = st.columns(2)

with col_input:
    st.markdown("### KALIMAT SUMBER")
    source_text = st.text_area(
        "Masukkan kalimat",
        placeholder="Tulis kalimat yang akan diterjemahkan...",
        height=150,
        key="source_text",
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        translate_clicked = st.button(
            "Terjemahkan",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        clear_clicked = st.button(
            "Kosongkan",
            use_container_width=True
        )

with col_output:
    st.markdown("### HASIL TERJEMAHAN")

    if st.session_state.result_text:
        st.markdown(f"""
        <div class="result-box has-result">
            {st.session_state.result_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box">
            <span class="placeholder">Hasil terjemahan akan muncul di sini</span>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# LOGIC
# ================================================================

if translate_clicked:
    text = source_text.strip()

    if not text:
        st.warning("Masukkan kalimat yang akan diterjemahkan.")
    elif source_lang == target_lang:
        st.warning("Bahasa sumber dan target tidak boleh sama.")
    else:
        with st.spinner("Menerjemahkan..."):
            try:
                result = translate_text(
                    text,
                    source_lang,
                    target_lang,
                    model,
                    tokenizer,
                    device
                )
                st.session_state.result_text = result
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

if clear_clicked:
    st.session_state.source_text = ""
    st.session_state.result_text = ""
    st.rerun()

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div class="info-text">
    <div>AXIS - Penerjemahan berbasis AI untuk bahasa daerah Sulawesi Tenggara</div>
    <div class="footer-links">
        <span>Tekan Ctrl+Enter untuk menerjemahkan</span>
    </div>
</div>
""", unsafe_allow_html=True)
