"""
Document CV — Streamlit Web Dashboard
Professional Dark Edition
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Customer Support | Document Retrieval",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Professional Dark Theme CSS ─────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #1c2128;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #8b949e;
        border: 1px solid #30363d;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #58a6ff;
        background-color: #1c2128;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c2128 !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #238636;
        color: white;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s;
    }
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline loader (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading AI Model (first time only, ~30–60s)...")
def load_pipeline():
    from core.pipeline import DocumentPipeline
    return DocumentPipeline(ocr_languages=["en"], use_gpu=False, dpi=150)


@st.cache_resource
def load_db():
    try:
        from database.db_manager import DocumentDB
        return DocumentDB()
    except Exception as e:
        return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## **AI Customer Support**")
    st.divider()

    max_pages = st.slider("Max PDF Pages", 1, 20, 5)
    min_confidence = st.slider("Min Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

    st.markdown("### **Visual Features**")
    show_detections = st.checkbox("Enable Region Detection", value=True)
    show_entities = st.checkbox("Auto-Extract Entities", value=True)
    save_to_db = st.checkbox("Persistent Storage", value=True)

    st.divider()
    st.markdown("### **System Health**")
    db = load_db()
    if db:
        try:
            stats = db.get_stats()
            if stats and "total_documents" in stats:
                st.metric("Processed Assets", stats["total_documents"])
                if stats.get("avg_confidence"):
                    st.metric("System Accuracy", f"{stats['avg_confidence']:.1%}")
            else:
                st.info("System Ready.")
        except Exception as e:
            st.warning(f"Could not load stats: {e}")
    else:
        st.error("Database Offline")

    st.divider()
    st.caption("Powered by AI Customer Support ")


# ─── Header ──────────────────────────────────────────────────────────────────

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<p class="main-title">AI Customer Support</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Document Retrieval & Information Extraction Engine</p>', unsafe_allow_html=True)

with header_col2:
    st.image("https://img.icons8.com/fluency/200/000000/document-delivery.png", width=120)

tab1, tab2, tab3, tab4 = st.tabs(["🚀 ANALYZE", "📂 ARCHIVE", "📈 ANALYTICS", "🔍 RETRIEVAL"])


# ─── Tab 1: Analyze ───────────────────────────────────────────────────────────

with tab1:
    st.markdown("### **Input Document**")
    uploaded = st.file_uploader(
        "Drop PDF or High-Res Image here",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_info, col_action = st.columns([2, 1])
        with col_info:
            st.info(f"**Selected File:** {uploaded.name} ({len(uploaded.getvalue())/1024:.1f} KB)")

        with col_action:
            analyze_btn = st.button("EXECUTE ANALYSIS", type="primary", use_container_width=True)

        if analyze_btn:
            progress = st.progress(0, text="🔧 Initializing pipeline...")
            status = st.empty()

            try:
                progress.progress(10, text="🤖 Loading AI model (first run may take ~60s)...")
                pipeline = load_pipeline()

                progress.progress(25, text="📂 Reading file...")
                file_bytes = uploaded.getvalue()

                progress.progress(40, text="🔍 Running OCR & detection (please wait)...")
                t0 = time.perf_counter()
                result = pipeline.process(file_bytes, file_name=uploaded.name, max_pages=max_pages)
                elapsed = time.perf_counter() - t0

                progress.progress(85, text="💾 Saving to database...")
                if save_to_db and db:
                    try:
                        db.save_result(result)
                    except Exception as e:
                        st.warning(f"Could not save to database: {e}")

                progress.progress(100, text="✅ Analysis complete!")
                time.sleep(0.5)
                progress.empty()
                status.empty()

            except Exception as e:
                progress.empty()
                st.error(f"❌ Analysis failed: {e}")
                st.stop()

            st.success(f"✅ Analysis Successful | Time: {elapsed:.2f}s")

            doc_type = getattr(result, 'document_type', 'unknown')
            doc_conf = getattr(result, 'document_confidence', 0.0)
            page_cnt = getattr(result, 'page_count', 0)
            total_text = getattr(result, 'total_text', '')
            tokens = len(total_text.split()) if total_text else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Class", doc_type.upper())
            m2.metric("Confidence", f"{doc_conf:.1%}")
            m3.metric("Page Count", page_cnt)
            m4.metric("Tokens", tokens)

            st.divider()

            pages = getattr(result, 'pages', [])
            if not pages:
                st.warning("No page data returned from pipeline.")
            else:
                for page in pages:
                    with st.expander(f"PAGE {page.page_number} DATA STREAM", expanded=True):
                        p1, p2 = st.columns([2, 1])
                        with p1:
                            st.markdown("**DIGITIZED TEXT**")
                            ocr_text = getattr(page.ocr, 'full_text', None) if hasattr(page, 'ocr') else None
                            if ocr_text:
                                st.code(ocr_text, language=None)
                            else:
                                st.caption("No OCR text available")

                        with p2:
                            if show_detections and hasattr(page, 'detections') and page.detections:
                                try:
                                    st.markdown("**LAYOUT SEGMENTATION**")
                                    det_dict = page.detections.to_dict()
                                    label_counts = det_dict.get("label_counts", {})
                                    if label_counts:
                                        for label, count in label_counts.items():
                                            st.markdown(f'<span class="tag">{label} ({count})</span>', unsafe_allow_html=True)
                                    else:
                                        st.caption("No detections found")
                                except Exception as e:
                                    st.caption(f"Detection error: {e}")

                            if show_entities and hasattr(page, 'entities') and page.entities:
                                st.markdown("**EXTRACTED ENTITIES**")
                                for etype, vals in page.entities.items():
                                    if vals:
                                        st.caption(f"**{etype.upper()}**")
                                        for v in vals[:5]:
                                            st.code(v, language=None)


# ─── Tab 2: Archive ───────────────────────────────────────────────────────────

with tab2:
    st.markdown("### **Document Archive**")
    if db:
        try:
            docs = db.list_documents(limit=50)
            if docs and len(docs) > 0:
                df = pd.DataFrame(docs)
                required_cols = ["file_name", "document_type", "confidence", "page_count", "created_at"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = "N/A"
                df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
                df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce').dt.strftime("%b %d, %Y %H:%M")
                st.dataframe(df[required_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Archive Empty.")
        except Exception as e:
            st.error(f"Could not load archive: {e}")
    else:
        st.info("Database not available.")


# ─── Tab 3: Analytics ─────────────────────────────────────────────────────────

with tab3:
    st.markdown("### **Intelligence Insights**")
    if db:
        try:
            stats = db.get_stats()
            if stats and stats.get("total_documents", 0) > 0:
                import plotly.express as px
                c1, c2 = st.columns(2)
                with c1:
                    by_type = stats.get("by_type", [])
                    if by_type:
                        df_type = pd.DataFrame(by_type)
                        if "count" in df_type.columns and "document_type" in df_type.columns:
                            fig1 = px.pie(df_type, values="count", names="document_type", hole=0.5, title="Document Composition")
                            fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig1, use_container_width=True)
                        else:
                            st.info("Incomplete document type data")
                    else:
                        st.info("No document type data")

                with c2:
                    label_counts = stats.get("detection_label_counts", [])
                    if label_counts:
                        df_labels = pd.DataFrame(label_counts)
                        if "count" in df_labels.columns and "label" in df_labels.columns:
                            fig2 = px.bar(df_labels, x="count", y="label", orientation='h', title="Feature Frequency")
                            fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("Incomplete label data")
                    else:
                        st.info("No detection label data")
            else:
                st.info("Insufficient data for analytics. Process at least one document first.")
        except Exception as e:
            st.error(f"Analytics error: {e}")
    else:
        st.info("Database not available.")


# ─── Tab 4: Retrieval ─────────────────────────────────────────────────────────

with tab4:
    st.markdown("### **Neural Retrieval**")
    q = st.text_input("Enter search parameters (e.g., 'invoice amount due')", placeholder="Search intelligence index...")
    if q and db:
        try:
            results = db.search(q)
            if results and len(results) > 0:
                for r in results:
                    file_name = r.get('file_name', 'Unknown file')
                    doc_type = r.get('document_type', 'unknown')
                    confidence = r.get('confidence', 0.0)
                    snippet = r.get('snippet', 'No snippet available')
                    with st.expander(f"📄 {file_name} | MATCH: {doc_type.upper()}"):
                        st.markdown(f"**MATCH RELEVANCE:** {confidence:.1%}")
                        st.markdown("**CONTEXT SNIPPET:**")
                        st.markdown(f"> ...{snippet}...", unsafe_allow_html=True)
            else:
                st.warning("No matches found in neural index.")
        except Exception as e:
            st.error(f"Search error: {e}")
    elif q and not db:
        st.error("Database is not available for search.")