# 🛍️ Zoro v3 — RAG-Powered E-commerce Assistant

A Retrieval-Augmented Generation (RAG) powered customer support chatbot for an e-commerce store — the third iteration of Zoro, upgrading from a pure LLM-based approach to a document-grounded retrieval system.

**Live Demo:** [zoro-chatbot-v3-cpl85xurmkyvbupuz7vvdz.streamlit.app](https://zoro-chatbot-v3-cpl85xurmkyvbupuz7vvdz.streamlit.app/)
**GitHub:** [github.com/Gireesh08/Zoro-Chatbot-v3](https://github.com/Gireesh08/Zoro-Chatbot-v3)

---

## 🧠 How It Works

Unlike v2, which relied entirely on the LLM's own knowledge (guided only by a system prompt with hardcoded FAQs), v3 separates KNOWLEDGE from GENERATION — store policies live in a real document that gets retrieved dynamically at query time, not baked into the prompt.

```
User Question
      ↓
Convert question to vector (embedding)
      ↓
Search ChromaDB for the most semantically similar document chunks
      ↓
Retrieve top 2 matching chunks (by meaning, not exact keywords)
      ↓
Combine retrieved chunks + original question into a grounded prompt
      ↓
Groq (Llama 3.3) generates a natural, document-grounded answer
```

---

## 📄 Knowledge Base

Store knowledge is stored in a plain `store_knowledge.txt` file, chunked into 11 topic sections:
- Return Policy
- Delivery Information
- Payment Methods
- Order Management
- Warranty Information
- Customer Support
- Product Categories
- Size and Exchange Information
- Discounts and Offers
- Installation Services
- Product Availability

Each section is stored as a separate vector in ChromaDB — enabling precise, topic-level retrieval instead of searching across one giant document blob.

---

## 📊 Before vs After: RAG vs v1 (TF-IDF)

| Query | v1 (TF-IDF/Cosine Similarity) | v3 (RAG) |
|---|---|---|
| *"How do I get my money back?"* | ❌ No match — zero word overlap with "refund" | ✅ Correctly retrieved Return Policy chunk |
| *"How many days for delivery?"* | ❌ Wrongly matched "cash on delivery" FAQ | ✅ Correctly retrieved Delivery chunk |
| *"Any good winter jackets available?"* | ❌ Wrongly matched unrelated FAQ | ✅ Correctly identified Clothing & Accessories |
| *"Is there a guarantee on electronics?"* | ❌ No match — "guarantee" vs "warranty" | ✅ Correctly retrieved Warranty chunk |
| *"Can I send the item back?"* | ❌ No match — "send back" vs "return" | ✅ Correctly retrieved Return Policy chunk |

RAG's embedding-based retrieval understands MEANING, not just exact word overlap — directly solving every documented v1 limitation.

---

## ⚠️ Key Design Decisions

**1. Built without LangChain** — the full RAG pipeline (chunk → embed → store → retrieve → generate) was implemented manually using ChromaDB and Groq directly, so every component is fully understood rather than abstracted away by a framework.

**2. `@st.cache_resource` for setup** — ChromaDB collection creation and document loading are cached using Streamlit's `cache_resource` decorator, ensuring the setup runs ONCE regardless of how many times Streamlit reruns the script, preventing duplicate collection errors.

**3. "ONLY use provided information" instruction** — the generation prompt explicitly instructs the LLM to answer ONLY from retrieved chunks, never from its own training knowledge — directly addressing the hallucination risk discovered during v2 testing.

**4. Paragraph-based chunking** — the knowledge document is split by blank lines (one topic per chunk), keeping each chunk semantically focused for more precise retrieval.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Vector Database:** ChromaDB (local, in-memory)
- **LLM API:** Groq (Llama 3.3 70B)
- **Interface:** Streamlit (chat UI with session state, quick-suggestion buttons, conversation reset)

---

## 📂 Project Structure

```
Zoro-v3/
  ├── app.py
  ├── store_knowledge.txt
  └── requirements.txt
```

---

## 🚀 Running Locally

> **Note:** The live demo above works out of the box — no setup needed. These steps are only for running your own local copy.

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
