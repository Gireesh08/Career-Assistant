# 🤖 Gireesh's Personal AI Career Assistant

A RAG-powered personal AI assistant that knows everything about Gireesh's ML/AI journey — his projects, skills, learning notes, and resume — and answers questions about them instantly, grounded in real documents, not guesswork.

**Live Demo:** [career-assistant.streamlit.app](https://career-assistant-44s2lwafhkvfbrvmb9amcc.streamlit.app/)
**GitHub:** [github.com/Gireesh08/Career-Assistant](https://github.com/Gireesh08/Career-Assistant)

---

## 💡 Why This Was Built

Preparing for technical interviews means constantly revisiting notes, README files, and project documentation scattered across multiple folders — answering questions like "why did you choose Naive Bayes here?" or "walk me through your RAG pipeline" requires mentally pulling together information from many different sources at once.

This project solves that problem by turning an entire ML/AI learning journey into a single, queryable knowledge base. Instead of digging through documents manually, you ask a question and the assistant retrieves the exact relevant context from your real notes and project docs, then generates a grounded, accurate answer.

It's also a direct application of the RAG skills built across Week 9 of the learning roadmap — rather than demoing RAG on generic data, this project uses genuinely personal, meaningful content as the knowledge source.

---

## 🧠 How It Works

```
13 personal documents (notes, READMEs, resume, interview handbook)
                    ↓
        Load → Chunk → Embed → Store in ChromaDB
                    ↓
            User asks a question
                    ↓
    ChromaDB retrieves semantically closest chunks
    (with smart metadata filtering per question type)
                    ↓
    Retrieved chunks + question → grounded prompt → LLM
                    ↓
    Accurate, document-grounded answer + sources shown
```

### Document Ingestion Pipeline

**13 documents** are automatically loaded, chunked, and stored at startup:

| Document Type | Files Included | Chunking Strategy |
|---|---|---|
| Study Notes (.md) | Week 3-5, Week 7-9, Week 8 notes | Split by `##` headers — one topic per chunk |
| Project READMEs (.md) | Zoro v1/v2/v3, Handwritten Equation Solver, Bangalore House Price, Face Mask Detector | Split by `##` headers |
| Resume (.pdf) | Gireesh Naidu Adireddi.pdf | Fixed 800-character chunks (handles dense PDF layout) |
| Interview Handbook (.pdf) | ML Interview Handbook Week 1-2 | Fixed 800-character chunks |

**Total chunks stored: 178** — each one a self-contained, independently retrievable unit.

### Smart Retrieval — Three-Layer Strategy

Not all questions are equal — the retrieval system adapts based on what's being asked:

1. **Primary retrieval** — semantic similarity search across ALL 178 chunks (always runs first)
2. **Project-specific retrieval** — if the question mentions "project," "Zoro," "deployed," "built," etc. → also searches specifically within `project_readme` tagged chunks
3. **Resume-specific retrieval** — if the question mentions "skills," "SQL," "certifications," "contact," "education," etc. → also searches specifically within `resume` tagged chunks

This metadata-filtered multi-layer approach dramatically improves answer accuracy for domain-specific questions that would otherwise get lost in a generic similarity search across 13 diverse documents.

### Scope Restriction

Off-topic requests (writing code, general knowledge questions, unrelated tasks) are detected and redirected with a friendly message — keeping the assistant focused on its purpose without feeling robotic.

---

## 📚 Knowledge Base

The assistant ingests Gireesh's complete documented ML/AI journey:

- **Learning Notes:** NLP, Transformers, Attention Mechanism, LLMs, Prompt Engineering, RAG, Vector Databases, Classical ML, Neural Networks, CNNs, Transfer Learning
- **Project Documentation:** Every deployed project's README — tech stack, approach, results, limitations, and design decisions
- **Resume:** Full technical skills, education, certifications, and contact information
- **Interview Handbook:** Structured interview prep notes covering Weeks 1-2 ML concepts

---

## ⚠️ Engineering Decisions and Discoveries

Building this project surfaced genuinely interesting challenges — worth documenting honestly:

**1. PDF chunking required a different strategy than markdown**
LaTeX-compiled PDFs don't produce clean `\n\n` paragraph breaks after text extraction — the whole resume was being stored as one giant chunk, causing retrieval to consistently return only the header section. Fixed by switching to fixed 800-character chunks for PDFs, which correctly splits the resume into granular, retrievable sections (contact, skills, projects, education, certifications).

**2. Context length cap required careful tuning**
With 3 primary + 3 metadata-filtered chunks, combined context easily exceeded the LLM's input token limit (413 errors). Solved by applying a 1500-character per-chunk trim AND a 3500-character total context cap — balancing retrieval breadth against prompt size constraints.

**3. Conversation context needed to inform retrieval**
Follow-up questions like "explain in more detail" carry no topic signal on their own — sending them directly to ChromaDB retrieved unrelated chunks. Fixed by enriching the retrieval query with recent conversation history before searching, so ChromaDB understands what "in more detail" actually refers to.

**4. Sources needed to persist across Streamlit reruns**
Streamlit re-runs the entire script on every interaction — source metadata stored in a plain variable was lost between messages. Fixed by including sources inside each `st.session_state.messages` entry, so every past answer's sources remain visible throughout the conversation.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Vector Database:** ChromaDB (local, in-memory with `@st.cache_resource` for session persistence)
- **Embedding:** ChromaDB's default embedding model (auto-applied at ingestion)
- **LLM:** Groq API — `openai/gpt-oss-120b`
- **PDF Loading:** PyPDF2
- **Interface:** Streamlit (chat UI, quick-suggestion buttons, source expanders, conversation reset)

---

## 📂 Project Structure

```
Career-Assistant/
  ├── app.py
  ├── requirements.txt
  └── career-assistant-docs/
        ├── week_03-05_ML-notes.md
        ├── Week4_Neural_Networks_Notes.md
        ├── Week5_CNN_ComputerVision_Notes.md
        ├── Week7_to_9_Notes.md
        ├── ML_Projects_Complete_Notes.md
        ├── zoro_v1_readme.md
        ├── zoro_v2_readme.md
        ├── zoro_v3_readme.md
        ├── equation_solver_readme.md
        ├── bangalore_readme.md
        ├── face_mask_detector_readme.md
        ├── Gireesh Naidu Adireddi.pdf
        └── ML_Interview_Handbook_Week1-2.pdf
```

---

## 🚀 Running Locally

> **Note:** The live demo works out of the box — no setup needed. These steps are only for running your own local copy.

```bash
pip install -r requirements.txt
```

Add your Groq API key to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_api_key_here"
```

Then run:
```bash
streamlit run app.py
```

---

## 💬 Example Questions to Try

- *"What projects has Gireesh built and what are their results?"*
- *"Explain the Attention Mechanism from Gireesh's notes"*
- *"What are Gireesh's SQL skills and certifications?"*
- *"Why did Gireesh choose Naive Bayes for Zoro v1?"*
- *"What is RAG and how did Gireesh implement it?"*
- *"Tell me about Gireesh's contact details and LinkedIn"*

## 👨‍💻 Author

**Gireesh Naidu Adireddi**

**GitHub:** [github.com/Gireesh08](https://github.com/Gireesh08)
**LinkedIn:** [linkedin.com/in/gireesh-adireddi-071362284](https://www.linkedin.com/in/gireesh-adireddi-071362284/)
