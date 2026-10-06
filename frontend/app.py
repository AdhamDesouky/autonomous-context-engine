# frontend/app.py
import time
import os
import streamlit as st
import requests

# 1. UI Configuration
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="■",
    layout="centered"
)

# 2. Custom CSS Injection (Typography, Gradients, Clean Chat Bubbles)
css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #9FA8DA, #80CBC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 2.8rem;
        padding-bottom: 0.5rem;
        margin-bottom: 0px;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 6rem; /* Extra padding so chat input doesn't overlap */
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# 3. Header Section
st.markdown('<h1 class="title-gradient">Research Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #78909C; margin-bottom: 2rem; font-size: 1.1rem;'>Ask me anything about the indexed literature.</p>", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/research/query")
UPLOAD_URL = os.getenv(
    "UPLOAD_URL",
    "http://localhost:8000/api/v1/research/documents",
)

with st.sidebar:
    st.subheader("Index documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index selected PDFs"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                upload_response = requests.post(
                    UPLOAD_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=600,
                )
            if upload_response.ok:
                result = upload_response.json()
                st.success(result["detail"])
            else:
                st.error(f"{uploaded_file.name}: {upload_response.text}")

# 4. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Inject an initial friendly greeting
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm your research assistant. What specific details would you like me to extract today?", 
        "thinking": None
    })

# 5. Render Historical Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If the message has a saved thinking trace, render it as a collapsed status box
        if msg.get("thinking"):
            with st.status("Agentic Pipeline Execution Log", expanded=False, state="complete"):
                st.write("Initiated backend connection...")
                st.write("Executed Hybrid RRF Search (Dense + Sparse)...")
                st.write("Graded chunks & synthesized context...")
        
        st.markdown(msg["content"])
        for source in msg.get("sources", []):
            location = source["source_file"]
            if source.get("page"):
                location += f" · page {source['page']}"
            with st.expander(f"Source: {location}"):
                if source.get("heading"):
                    st.caption(source["heading"])
                st.write(source.get("snippet", ""))

# 6. Chat Input & Execution Logic
if prompt := st.chat_input("Ask a research question..."):
    
    # 6a. Render and store the User's message
    st.session_state.messages.append({"role": "user", "content": prompt, "thinking": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6b. Render the Assistant's response
    with st.chat_message("assistant"):
        sources = []
        
        # The 'thinking' UI: Intentionally collapsed by default using expanded=False
        with st.status("Executing Agentic Pipeline...", expanded=False) as status:
            st.write("Initiating backend connection...")
            time.sleep(0.4) 
            st.write("Executing Hybrid RRF Search (Dense + Sparse)...")
            
            try:
                response = requests.post(API_URL, json={"query": prompt})
                
                if response.status_code == 200:
                    st.write("Grading chunks & synthesizing context...")
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    status.update(label="Extraction Complete", state="complete")
                    
                    # Store response in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "thinking": True,
                        "sources": sources,
                    })
                    
                else:
                    status.update(label="Pipeline Failure", state="error")
                    answer = f"Error [{response.status_code}]: {response.text}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "thinking": False,
                        "sources": [],
                    })
                    
            except requests.exceptions.ConnectionError:
                status.update(label="Connection Failed", state="error")
                answer = "Backend unreachable. Ensure FastAPI is running on port 8000."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "thinking": False,
                    "sources": [],
                })

        # Render the final output below the thinking box
        st.markdown(answer)
        for source in sources:
            location = source["source_file"]
            if source.get("page"):
                location += f" · page {source['page']}"
            with st.expander(f"Source: {location}"):
                if source.get("heading"):
                    st.caption(source["heading"])
                st.write(source.get("snippet", ""))