import streamlit as st
import chromadb
import os
from groq import Groq
from pypdf import PdfReader

# ============================================
# SETUP — runs ONCE (cached)
# ============================================

@st.cache_resource
def setup():
    client_groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="career_assistant")

    docs_folder = "career-assistant-docs"

    def load_pdf(filepath):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_document(content, filename):
        if filename.endswith(".md"):
            chunks = content.split("\n## ")
            chunks = ["## " + c if not c.startswith("## ") else c for c in chunks]
        else:
            # For PDFs — split by single newline first, then fallback
            chunks = content.strip().split("\n\n")
            if len(chunks) <= 2:
                # Fixed-size chunking for dense PDFs like resume
                chunks = [content[i:i+800] for i in range(0, len(content), 800)]
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks

    chunk_counter = 0
    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)

        if filename.endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        elif filename.endswith(".pdf"):
            content = load_pdf(filepath)
        else:
            continue

        # Metadata tagging
        if "readme" in filename.lower():
            doc_type = "project_readme"
        elif "handbook" in filename.lower() or "interview" in filename.lower():
            doc_type = "interview_prep"
        elif "notes" in filename.lower() or "week" in filename.lower():
            doc_type = "study_notes"
        elif "resume" in filename.lower() or "adireddi" in filename.lower():
            doc_type = "resume"
        else:
            doc_type = "general"

        chunks = chunk_document(content, filename)
        for chunk in chunks:
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_chunk_{chunk_counter}"],
                metadatas=[{"source": filename, "type": doc_type}]
            )
            chunk_counter += 1

    return client_groq, collection

client_groq, collection = setup()

# ============================================
# SCOPE CHECK — detect off-topic questions
# ============================================

def is_off_topic(user_question):
    off_topic_keywords = [
        "write a", "write me", "generate code", "create a script",
        "python code", "javascript", "html code", "hello world",
        "weather", "news", "stock price", "recipe", "translate",
        "joke", "story", "poem", "song", "movie", "sport score"
    ]
    return any(keyword in user_question.lower() for keyword in off_topic_keywords)

REDIRECT_MESSAGE = """I'm focused on answering questions about Gireesh's ML/AI journey, skills, and projects — I can't help with general tasks outside that scope.

Try asking me something like:
- *"What projects has Gireesh built?"*
- *"Explain Gireesh's RAG pipeline"*
- *"What are Gireesh's technical skills?"*
- *"Tell me about Zoro v3"*"""

# ============================================
# RAG FUNCTION
# ============================================

MAX_CHUNK_LENGTH = 1500   # per chunk
MAX_CONTEXT_LENGTH = 3500 # total context cap

def career_assistant_answer(user_question, chat_history = None):
    # Building Context aware query
    if chat_history and len(chat_history)> 0:
        # Get last assistant message to understand conversation context
        last_messages = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        context_query = " ".join([m["content"][:100] for m in last_messages]) + " " + user_question
    else:
        context_query = user_question

    # Use context_query for retrieval, user_question for display
    results = collection.query(
        query_texts=[context_query],
        n_results=3
    )

    # Scope check first
    if is_off_topic(user_question):
        return REDIRECT_MESSAGE, []

    # Primary retrieval
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )
    retrieved_chunks = results['documents'][0]
    retrieved_sources = results['metadatas'][0]

    # Project-specific retrieval
    project_keywords = [
        "project", "built", "deployed", "strongest", "best",
        "zoro", "chatbot", "equation", "bangalore", "mnist",
        "nyc", "taxi", "handwritten", "face mask"
    ]
    if any(keyword in user_question.lower() for keyword in project_keywords):
        project_results = collection.query(
            query_texts=[user_question],
            n_results=3,
            where={"type": "project_readme"}
        )
        retrieved_chunks += project_results['documents'][0]
        retrieved_sources += project_results['metadatas'][0]

    # Resume-specific retrieval — expanded keywords
    resume_keywords = [
        "sql", "skills", "experience", "strongest", "best",
        "resume", "profile", "certifications", "certificate",
        "education", "background", "tools", "technologies",
        "contact", "email", "phone", "linkedin", "github",
        "degree", "college", "cgpa", "languages", "deployment"
    ]
    if any(keyword in user_question.lower() for keyword in resume_keywords):
        resume_results = collection.query(
            query_texts=[user_question],
            n_results=5,   # more results for resume since it's dense
            where={"type": "resume"}
        )
        retrieved_chunks += resume_results['documents'][0]
        retrieved_sources += resume_results['metadatas'][0]

    # Remove duplicates while keeping sources aligned
    seen = set()
    unique_chunks = []
    unique_sources = []
    for chunk, source in zip(retrieved_chunks, retrieved_sources):
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
            unique_sources.append(source)

    # Trim each chunk
    trimmed_chunks = [chunk[:MAX_CHUNK_LENGTH] for chunk in unique_chunks]
    context = "\n\n".join(trimmed_chunks)

    # Cap total context
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [Context truncated]"

    prompt = f"""You are Gireesh's Personal AI Career Assistant.
You have access to Gireesh's complete ML/AI learning notes, project documentation, and resume.
Answer questions about Gireesh's skills, projects, learning journey, and technical knowledge.
Use ONLY the information provided below — never make things up.
If the answer isn't in the provided information, say so honestly.

Information:
{context}

Question: {user_question}
"""
    response = client_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content, unique_sources

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Gireesh's AI Career Assistant", page_icon="🤖")
st.title("🤖 Gireesh's Personal AI Career Assistant")
st.write("Ask me anything about Gireesh's skills, projects, or ML/AI journey!")

# Clear button
if st.button("🔄 Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Quick suggestion buttons
st.write("Try asking:")
col1, col2, col3 = st.columns(3)
quick_question = None
with col1:
    if st.button("What are his projects?"):
        quick_question = "Give me the summary of every project that Gireesh has done and mentioned in his resume"
with col2:
    if st.button("What is RAG?"):
        quick_question = "What is RAG?"
with col3:
    if st.button("His SQL skills?"):
        quick_question = "What are Gireesh's SQL skills?"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display ALL past messages with their sources
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Show sources for EVERY assistant message in history
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources used"):
                seen_sources = set()
                for source in message["sources"]:
                    label = f"{source['source']} ({source['type']})"
                    if label not in seen_sources:
                        seen_sources.add(label)
                        st.write(f"• `{source['source']}` — {source['type']}")

# Get user input
typed_input = st.chat_input("Ask anything about Gireesh...")
user_input = quick_question if quick_question else typed_input

# Process and respond
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get answer
    reply, sources = career_assistant_answer(user_input, st.session_state.messages)

    # Store in session state WITH sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources
    })

    # Display assistant reply
    with st.chat_message("assistant"):
        st.write(reply)
        if sources:
            with st.expander("📚 Sources used"):
                seen_sources = set()
                for source in sources:
                    label = f"{source['source']} ({source['type']})"
                    if label not in seen_sources:
                        seen_sources.add(label)
                        st.write(f"• `{source['source']}` — {source['type']}")