# 🛍️ Zoro — E-commerce Customer Assistant

A retrieval-based chatbot that answers customer FAQs and predicts product categories for an e-commerce store.

**Live Demo:** [zoro-chatbot.streamlit.app](https://zoro-chatbot-ouwhwviikrupumezdbminh.streamlit.app/)
**GitHub:** [github.com/Gireesh08/Zoro-Chatbot](https://github.com/Gireesh08/Zoro-Chatbot)

---

## 🧠 How It Works

```
User Message
     ↓
[1] FAQ Matcher (TF-IDF + Cosine Similarity)
     → Confident match? → Return FAQ answer
     ↓ (no confident match)
[2] Product Category Classifier (TF-IDF + Multinomial Naive Bayes)
     → Confident prediction? → Return category suggestion
     ↓ (no confident prediction)
[3] Graceful Fallback
     → "I'm sorry, I didn't quite understand that..."
```

### Components

**1. Product Category Classifier**
- Trained on a real e-commerce dataset (6,000+ product descriptions across 4 categories: Books, Electronics, Clothing & Accessories, Household)
- Pipeline: Text → TF-IDF vectorization (46,289-word vocabulary) → Multinomial Naive Bayes
- **Test accuracy: 96%** (see `classification_report` below)
- Includes a confidence threshold (via `predict_proba`) so low-confidence guesses fall back gracefully instead of forcing a wrong answer

**2. FAQ Matcher**
- 20 hand-curated customer support questions (returns, delivery, payments, warranty, etc.)
- Pipeline: Text → TF-IDF vectorization → Cosine Similarity against all FAQ vectors → best match above threshold
- Falls back to `None` if no match is confident enough, allowing the category classifier to attempt a response instead

**3. Streamlit Chat Interface**
- Persistent conversation history using `st.session_state`
- Trained models/vectorizers loaded via Pickle (no retraining on each app launch)

---

## 📊 Model Performance

```
                        precision    recall  f1-score   support

                 Books       0.98      0.93      0.95      1503
Clothing & Accessories       0.97      0.98      0.98      1492
           Electronics       0.96      0.97      0.96      1475
             Household       0.93      0.96      0.95      1530

              accuracy                           0.96      6000
             macro avg       0.96      0.96      0.96      6000
          weighted avg       0.96      0.96      0.96      6000
```

---

## ⚠️ Known Limitations

Testing this bot surfaced some real limitations of word-frequency-based NLP (TF-IDF):

1. **No synonym/semantic understanding.** A query like *"How do I get my money back?"* fails to match the FAQ *"Can I get a refund?"* — because TF-IDF only matches exact words, not meaning. "Money back" and "refund" share zero vocabulary overlap.

2. **Generic word collisions on short queries.** Short queries can accidentally match the wrong FAQ purely because they share one common word. For example, *"How many days for delivery?"* matched the "cash on delivery" FAQ instead of the delivery-time FAQ, simply because both share the word "delivery." Similarly, *"Any good winter jackets available?"* incorrectly matched a books-related FAQ, purely due to the shared word "available."

3. **No true "I don't know" for the classifier.** Multinomial Naive Bayes always produces a probability distribution across all known categories — even for gibberish input, it will pick *something*. A confidence threshold (`predict_proba`) was added to catch the most obvious low-confidence cases, but the model has no genuine concept of an out-of-distribution input.

---

## 🛠️ Tech Stack

- **Language:** Python
- **NLP/ML:** scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `cosine_similarity`)
- **Data handling:** Pandas, NumPy
- **Model persistence:** Pickle
- **Interface:** Streamlit (chat UI with `st.session_state`)
- **Dataset:** E-commerce product description dataset (Books, Electronics, Clothing & Accessories, Household)

---

## 📂 Project Structure

```
Zoro/
  ├── app.py                # Streamlit chat interface
  ├── chatbot_logic.py      # FAQ matching + classification logic
  ├── requirements.txt
  ├── model.pkl              # Trained Naive Bayes classifier
  ├── vectorizer.pkl         # TF-IDF vectorizer (product classifier)
  ├── faq_vectorizer.pkl     # TF-IDF vectorizer (FAQ matcher)
  ├── faq_vectors.pkl        # Pre-computed FAQ vectors
  └── faq_data.pkl           # FAQ questions + answers
```

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---


