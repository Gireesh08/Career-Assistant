# 🛍️ Zoro v2 — LLM-Powered E-commerce Assistant

An LLM-powered customer support chatbot for an e-commerce store, built using a carefully engineered system prompt instead of trained classifiers — the natural-language upgrade to the original TF-IDF/Naive Bayes based Zoro.

**Live Demo:** [zoro-chatbot-v2-yldzrz9kzjlcmapp2wf3wmg.streamlit.app](https://zoro-chatbot-v2-yldzrz9kzjlcmapp2wf3wmg.streamlit.app)
**GitHub:** [github.com/Gireesh08/Zoro-Chatbot-v2](https://github.com/Gireesh08/Zoro-Chatbot-v2)

---

## 🧠 How It Works

Unlike the original version — which combined a trained TF-IDF + Naive Bayes classifier with a separate TF-IDF + Cosine Similarity FAQ matcher — this version uses a **single LLM call**, guided entirely by a structured system prompt.

```
User Message
     ↓
Full conversation history (system prompt + all prior messages) sent to the LLM
     ↓
LLM interprets intent using natural language understanding (not exact word matching)
     ↓
Response generated, grounded in the store's categories, FAQs, and explicit scope/behavior rules
```

### Key Components

- **System Prompt** — defines Zoro's identity, the store's 4 product categories, and all customer support FAQs, written directly into the prompt (no separate training step required).
- **Conversation Memory** — the full message history (not just the latest message) is sent on every call, giving Zoro genuine multi-turn context — something the original classifier-based version did not have.
- **Low Temperature (0.3)** — tuned for consistent, factual responses appropriate for customer support, rather than creative variability.
- **Scope Restriction** — explicit instructions preventing Zoro from answering unrelated requests (e.g., writing code or essays), keeping it focused on its intended role.
- **Anti-Hallucination Guardrail** — explicit instructions preventing Zoro from inventing product names, sections, or details not present in the system prompt.

---

## 📊 Before vs. After: Solving Real Limitations

The original version's documented limitations were used as direct test cases for this version:

| Query | v1 (TF-IDF/Cosine Similarity) | v2 (LLM-Powered) |
|---|---|---|
| *"How do I get my money back if delivery is late?"* | ❌ No match — zero word overlap with "refund" | ✅ Correctly linked to the return/refund policy |
| *"How many days for delivery?"* | ❌ Wrongly matched the "cash on delivery" FAQ | ✅ Correctly answered with delivery timeframe |
| *"Any good winter jackets available?"* | ❌ Wrongly matched an unrelated FAQ via the shared word "available" | ✅ Correctly identified the Clothing & Accessories category |

This confirms the core reasoning behind the shift toward LLM-based systems: semantic understanding of meaning succeeds where exact word-frequency matching fails.

---

## ⚠️ New Issues Discovered (and Fixed) During Testing

Building an LLM-based system introduces a different category of problems than a classical ML pipeline — surfaced through direct testing:

1. **Hallucination.** When asked for book recommendations, an early version of Zoro invented a "Staff Picks" section and specific genres that don't exist in the store's actual data. **Fix:** an explicit instruction was added prohibiting the model from inventing any product details, sections, or names not present in the system prompt.

2. **Lack of scope boundaries.** Without explicit restriction, Zoro would answer completely unrelated requests (e.g., "write me a Python script," "write an essay") just like a general-purpose LLM, rather than staying in character as a store assistant. **Fix:** an explicit scope restriction was added, instructing the model to decline unrelated requests and redirect the conversation back to store topics.

3. **Response length calibration.** An overly strict "keep answers short" instruction made responses feel robotic and reduced the natural helpfulness an LLM can provide. An overly loose instruction produced excessively long, padded answers. **Fix:** the style guide was tuned to scale response length to the complexity of the question, rather than enforcing a fixed length.

These findings highlight that LLM-based systems trade one set of failure modes (rigid keyword matching) for another (hallucination, scope drift) — both of which require deliberate prompt engineering to manage.

---

## 🛠️ Tech Stack

- **Language:** Python
- **LLM API:** Groq (Llama 3.3 70B)
- **Interface:** Streamlit (chat UI with `st.session_state`, quick-suggestion buttons, conversation reset)

---

## 📂 Project Structure

```
Zoro-v2/
  ├── app.py
  └── requirements.txt
```

---

## 🚀 Running Locally

> **Note:** The live demo above already works out of the box — no API key needed to use it, since the key is securely configured on the deployment itself. The steps below are only needed if you want to run your **own copy** of this project locally.

```bash
pip install -r requirements.txt
```

Add your own Groq API key via Streamlit secrets (`.streamlit/secrets.toml`):
```toml
GROQ_API_KEY = "your_api_key_here"
```

Then run:
```bash
streamlit run app.py
```