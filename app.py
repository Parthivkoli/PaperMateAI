import streamlit as st
from arxiv import Search, SortCriterion, Client
import requests
import PyPDF2
import io
import re
from transformers import pipeline

# Custom CSS for styling (Dark mode friendly)
st.markdown("""
<style>
    .summary-box {
        padding: 20px;
        background: var(--background-secondary);
        color: var(--text-color);
        border-radius: 10px;
        margin-top: 20px;
    }
    .pdf-viewer {
        width: 100%;
        height: 600px;
        border: none;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load different summarization models
@st.cache_resource
def load_models():
    return {
        "brief": pipeline("summarization", model="facebook/bart-large-cnn"),
        "abstract": pipeline("summarization", model="t5-small"),
        "detailed": pipeline("summarization", model="t5-base")
    }

# Clean text for summarization
def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'http\S+', '', text)
    return text[:15000]  # Limit text size for summarization

# **App Title**
st.title("🧠 PaperMate: AI Research Companion")
st.markdown("🚀 **Discover, Read, and Summarize Research Effortlessly!**")

# Ensure session states exist
if "papers" not in st.session_state:
    st.session_state.papers = []
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "view_paper" not in st.session_state:
    st.session_state.view_paper = False
if "summarization_type" not in st.session_state:
    st.session_state.summarization_type = "abstract"

models = load_models()

# **🔍 Search Bar on the Main Page**
st.markdown("### 🔎 Search for Research Papers")
query = st.text_input("Enter research topic (e.g., 'AI in Healthcare'):")
search_btn = st.button("🔍 Search")

# **Sidebar for Summarization Settings**
with st.sidebar:
    st.header("📝 Summarization Settings")
    summarization_type = st.radio("Choose Summary Type", ["Brief", "Abstract", "Detailed"])
    st.session_state.summarization_type = summarization_type.lower()

    # **Dynamic Summary Length Sliders**
    if summarization_type == "Brief":
        max_length = st.slider("Brief Summary Length (words)", 50, 150, 100)
    elif summarization_type == "Abstract":
        max_length = st.slider("Abstract Summary Length (words)", 200, 400, 300)
    elif summarization_type == "Detailed":
        max_length = st.slider("Detailed Summary Length (words)", 500, 1000, 700)
    else:
        max_length = 300  # Default fallback

# **Search Functionality**
if search_btn and query:
    with st.spinner("🔭 Scanning arXiv for latest research..."):
        client = Client()
        search = client.results(Search(
            query=query,
            max_results=12,
            sort_by=SortCriterion.Relevance
        ))
        st.session_state.papers = list(search)

# **Show Papers Below the Search Bar**
if not st.session_state.view_paper:
    if st.session_state.papers:
        st.subheader(f"📄 Found {len(st.session_state.papers)} Papers")
        for i, paper in enumerate(st.session_state.papers):
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <h3>{paper.title}</h3>
                    <p>{paper.summary[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"📖 Read Paper", key=f"select_{i}"):
                    st.session_state.selected_paper = {
                        "title": paper.title,
                        "summary": paper.summary,
                        "authors": [a.name for a in paper.authors],
                        "published": str(paper.published.date() if paper.published else "Unknown"),
                        "category": paper.primary_category,
                        "pdf_url": paper.pdf_url
                    }
                    st.session_state.view_paper = True
                    st.rerun()

# **Show Selected Paper in a Separate View**
if st.session_state.view_paper and st.session_state.selected_paper:
    paper = st.session_state.selected_paper
    st.markdown("---")

    # **Paper details in a separate view**
    st.subheader(f"📄 {paper['title']}")
    st.caption(f"📅 {paper['published']} | 👥 {', '.join(paper['authors'][:3])}...")
    st.caption(f"📚 {paper['category']}")

    # Abstract & PDF Viewer
    with st.expander("📖 Read Abstract", expanded=True):
        st.write(paper["summary"])
    
    st.markdown(f"### 📑 Read Full Paper")
    pdf_url = paper["pdf_url"]
    if pdf_url:
        st.markdown(f'<iframe class="pdf-viewer" src="{pdf_url}"></iframe>', unsafe_allow_html=True)
    else:
        st.error("PDF not available for this paper.")

    # **Summarization Button**
    if st.button("✨ Generate Summary"):
        with st.spinner("📥 Downloading and analyzing paper..."):
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()

                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Extract text safely
                text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages if page.extract_text()])
                
                if not text.strip():
                    st.error("⚠️ Unable to extract readable text from this PDF. It may be an image-based scan.")
                else:
                    text = clean_text(text)

                    summarizer = models[st.session_state.summarization_type]
                    summary = summarizer(
                        text,
                        max_length=max_length,
                        min_length=int(max_length / 2),
                        do_sample=False
                    )[0]["summary_text"]

                    st.session_state.summary = summary  # Store summary in session_state
            
            except requests.exceptions.RequestException as req_error:
                st.error(f"⚠️ Network error while downloading PDF: {str(req_error)}")
            except PyPDF2.errors.PdfReadError as pdf_error:
                st.error(f"⚠️ Error processing PDF: {str(pdf_error)}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")

    # Display summary if generated
    if st.session_state.summary:
        st.markdown("---")
        st.markdown(f"### 📝 {summarization_type.capitalize()} Summary")
        st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)

    # **Back to Search**
    if st.button("🔙 Back to Search"):
        st.session_state.view_paper = False
        st.rerun()