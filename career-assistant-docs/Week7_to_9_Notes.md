# Gireesh's ML/AI Learning Notes — Weeks 7 to 9
**Purpose:** Study revision + RAG source document for Personal AI Career Assistant
**Format:** Each topic is a self-contained section for clean RAG chunking

---

# WEEK 7 — NLP + Transformers

---

## Tokenization
- **Definition:** The process of breaking raw text into smaller units called tokens — these can be words, sub-words, or characters, depending on the method used.
- **When to Use:** Always the FIRST step in any NLP pipeline, before any further text processing or embedding.
- **Why We Use This:** Computers cannot process raw text directly — they need numbers. Tokenization is the first step that converts text into manageable units that can then be converted into numbers.
- **Key Detail:** A token is NOT always a whole word. Example: "tokenization" → "token" + "ization" (2 tokens). Rule of thumb: ~1 token ≈ 4 characters of English text.
- **Code Example:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I love mangoes")
for token in doc:
    print(token.text)
# Output: I, love, mangoes
```
- **Explain Like a Beginner:** Imagine you're reading a sentence word by word, picking up each word as a separate card — that's tokenization. Instead of reading the whole sentence as one blur, you hold each word (or word-part) as its own separate piece, which makes it easier to understand and process individually.
- **Interview One-Liner:** Tokenization breaks raw text into smaller units (tokens) so a computer can process and convert them into numbers — it's always the first step in any NLP pipeline.

---

## Stemming and Lemmatization
- **Definition:** Both reduce words to their base/root form. Stemming chops off word endings using rules (fast but crude). Lemmatization uses a dictionary to find the actual root word (slower but accurate).
- **When to Use:** Used during text preprocessing to standardize words so the model treats "running," "ran," and "runs" as the same concept.
- **Why We Use This:** Without this step, a model treats "run," "running," and "ran" as three completely separate, unrelated words — wasting features and reducing accuracy.
- **Key Difference:** Stemming: "running" → "run" (just cuts the suffix). Lemmatization: "ran" → "run" (looks up the actual dictionary base form — more accurate).
- **Code Example:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats are running and they ran yesterday")
for token in doc:
    print(token.text, "→", token.lemma_)
# Output: cats → cat, running → run, ran → run
```
- **Explain Like a Beginner:** Imagine all different forms of a cricket shot (drive, drove, driving) being filed under one single card labeled "drive" in your notes — so when you search for it, you find all variations under one place. That's what stemming and lemmatization do for words.
- **Interview One-Liner:** Stemming and lemmatization both reduce words to their root form to standardize text — stemming is faster but crude, lemmatization is slower but accurate using a dictionary lookup.

---

## Stop Words
- **Definition:** Common, low-meaning filler words (like "the," "is," "a," "and," "in") that appear frequently in almost every sentence but carry little actual meaning on their own.
- **When to Use:** During text preprocessing, BEFORE vectorization — removing them reduces noise in word-frequency-based methods like TF-IDF and Bag of Words.
- **Why We Use This:** Words like "the" and "is" appear in almost every sentence with huge counts — they clutter the data without adding any signal. Removing them forces the model to focus on meaningful words.
- **Code Example:**
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(["The cat is sitting on the mat"])
print(vectorizer.get_feature_names_out())
# Output: ['cat', 'mat', 'sitting'] — stop words removed
```
- **Honest Caution:** Removing "not" as a stop word can flip meaning entirely — "This is not good" becomes "good" after removal. Be careful in sentiment analysis tasks.
- **Explain Like a Beginner:** Like peeling the skin and stems off vegetables before cooking — technically part of the vegetable, but discarded because they don't add flavor to the final dish. Stop words are the "stems and peels" of sentences.
- **Interview One-Liner:** Stop words are frequent, low-information filler words removed during preprocessing so models focus on meaningful terms — but must be used carefully since words like "not" can change meaning if dropped.

---

## Bag of Words (BoW)
- **Definition:** A text representation method that converts a sentence into a list of numbers by simply counting how many times each word appears — completely ignoring word order and grammar.
- **When to Use:** Quick baseline text classification tasks where word frequency is informative (spam detection, sentiment, category classification).
- **Why We Use This:** Computers can only process numbers. BoW is the simplest way to convert text into numbers — count each word's occurrences and use that count as a feature.
- **Key Weakness:** Treats all words as equally important regardless of how common they are across all documents — AND completely ignores word order ("dog bites man" = "man bites dog" in BoW).
- **Code Example:**
```python
from sklearn.feature_extraction.text import CountVectorizer
sentences = ["I love mangoes", "I love apples and mangoes"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names_out())
# Output: ['and', 'apples', 'love', 'mangoes']
print(X.toarray())
# Output: [[0, 0, 1, 1], [1, 1, 1, 1]]
```
- **Key Detail:** `vocabulary_` stores word-to-column-position mapping (NOT word counts). Counts appear only in the actual transformed matrix.
- **Explain Like a Beginner:** Like emptying a bag of ingredients onto a table and counting each type — 2 onions, 3 tomatoes, 1 garlic. You know WHAT's there and HOW MUCH, but not the ORDER they were added in. That's Bag of Words — it counts what words are in a sentence but throws away the order completely.
- **Interview One-Liner:** Bag of Words converts text to number-vectors by counting word occurrences — simple and fast, but loses all word-order information and treats all words as equally important regardless of rarity.

---

## N-Grams / Bag of N-Grams
- **Definition:** An extension of Bag of Words that counts groups of N consecutive words together as single units instead of just individual words.
- **When to Use:** When capturing short phrase-level context matters — e.g., "not good" should be recognized as a unit, not just "not" and "good" separately.
- **Why We Use This:** Plain Bag of Words splits "not good" into two separate words, losing the negation. Bigrams (N=2) keep "not good" together as one feature, preserving the negative meaning.
- **Key Parameter:** `ngram_range=(1,2)` — includes both single words (unigrams) AND word-pairs (bigrams) as features simultaneously.
- **Code Example:**
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(["This phone is not good", "This phone is very good"])
print(vectorizer.get_feature_names_out())
# Shows both single words AND pairs like "not good", "very good"
```
- **Tradeoff:** Vocabulary size EXPLODES with larger N — more features, more memory, slower training (curse of dimensionality).
- **Explain Like a Beginner:** Reading words in connected chunks instead of one at a time. "Not good" read as two separate words loses its meaning — but read as ONE connected chunk, it's clearly negative. N-grams force the model to see these connected chunks.
- **Interview One-Liner:** N-grams group N consecutive words as single features to capture short phrase-level context that Bag of Words misses — at the cost of a much larger vocabulary and higher memory usage.

---

## TF-IDF (Term Frequency — Inverse Document Frequency)
- **Definition:** A smarter text representation that scores each word by how often it appears in ONE document (TF) multiplied by how RARE it is across ALL documents (IDF) — so common words everywhere get down-weighted, and rare, distinctive words get boosted.
- **When to Use:** Text classification, information retrieval, FAQ matching — wherever word importance matters more than raw frequency.
- **Why We Use This Over Bag of Words:** BoW treats "the" and "refund" equally if they appear the same number of times. TF-IDF correctly recognizes "the" appears in every document (low IDF, down-weighted) while "refund" is rare and distinctive (high IDF, up-weighted).
- **Key Parameters:**
  - `stop_words='english'` — removes filler words before scoring
  - `ngram_range` — works same as CountVectorizer
  - `max_features` — limits vocabulary to top N most important words
  - `min_df` / `max_df` — ignores words that appear in too few or too many documents
- **Code Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
sentences = ["The phone battery drains fast", "The phone camera is amazing"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
# "phone" appears in BOTH — low IDF, down-weighted
# "battery" appears in only ONE — high IDF, up-weighted
```
- **Viewing IDF scores:**
```python
for word in vectorizer.get_feature_names_out():
    index = vectorizer.vocabulary_.get(word)
    print(f"{word}: {vectorizer.idf_[index]}")
```
- **Explain Like a Beginner:** A word that appears in EVERY document (like "the") isn't telling you anything special about any particular document. But a word that appears in ONLY ONE document (like "battery-drain") is telling you something very specific. TF-IDF gives high scores to words that are frequent in THIS document but rare elsewhere — like a keyword that uniquely identifies this document from all others.
- **Interview One-Liner:** TF-IDF scores each word by (frequency in this document) × (rarity across all documents) — so common words everywhere get low scores and distinctive words get high scores, making it smarter than plain word counting.

---

## Cosine Similarity
- **Definition:** A measure of similarity between two vectors based on the ANGLE between them — not their size. Score ranges from -1 (opposite) to 0 (unrelated) to 1 (identical meaning/direction).
- **When to Use:** Comparing text documents, matching queries to FAQs, finding similar sentences — any time you want to measure meaning-similarity between two vectorized pieces of text.
- **Why We Use This (Not Plain Distance):** Two sentences can mean the same thing but one is longer (repeated words make the distance bigger) — cosine similarity ignores size and only cares about DIRECTION, which maps better to "meaning" for text.
- **Code Example:**
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(user_vector, faq_vectors)
best_match_index = similarity.argmax()
```
- **Explain Like a Beginner:** Two batsmen playing a shot in almost the exact same direction — the angle between their "shot arrows" is tiny → high cosine similarity (very similar shots). If one hits leg-side and the other hits cover (90° apart) → low similarity (completely different shots). Cosine similarity measures the angle between two arrows (vectors), not how hard they hit.
- **Interview One-Liner:** Cosine similarity measures the angle between two vectors rather than their distance — a score of 1 means same direction (highly similar), 0 means unrelated, -1 means opposite — used for text because it ignores document length and focuses purely on directional meaning.

---

## Word Embeddings and Word2Vec
- **Definition:** Word Embeddings are dense vector representations of words where similar-meaning words end up numerically close together in a shared "meaning space." Word2Vec is the algorithm that LEARNS these embeddings by training a neural network on a fill-in-the-blank prediction task.
- **When to Use:** When you need to capture semantic meaning and relationships between words — beyond simple word counting. Better than TF-IDF for understanding synonyms and word relationships.
- **Why Better Than TF-IDF:** TF-IDF treats "king" and "queen" as completely unrelated (different columns). Word2Vec understands they're semantically similar (close vectors) because they appear in similar contexts across training data.
- **Two Approaches:**
  - **CBOW (Continuous Bag of Words):** Given surrounding words → predict the middle word. Faster, good for frequent words.
  - **Skip-gram:** Given the middle word → predict surrounding words. Slower, better for rare words.
- **Famous Example:** king - man + woman ≈ queen (meaning-level math is possible with good embeddings)
- **How Numbers Are Learned:** Start random → train on fill-in-the-blank task → backpropagation nudges vectors slightly after each wrong prediction → millions of repetitions → vectors organically reflect real meaning relationships.
- **Code Example (Gensim):**
```python
from gensim.models import Word2Vec
sentences = [["the", "cat", "sat"], ["the", "dog", "ran"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
print(model.wv['cat'])           # 100-number vector for "cat"
print(model.wv.most_similar('cat'))  # words closest in meaning
```
- **spaCy Vector Sizes:** Both `en_core_web_md` and `en_core_web_lg` give 300 numbers per word — `lg` covers a larger vocabulary, but vector SIZE is the same.
- **Related Tools:** Gensim = library for training/using embeddings. fastText = improvement over Word2Vec that handles unseen/rare words by learning from sub-word chunks (e.g., "footballer" = "foot" + "ball" + "er").
- **Explain Like a Beginner:** GPS coordinates for word meaning — "apple" and "banana" have similar coordinates (fruit neighborhood), while "car" is in a completely different neighborhood. Word2Vec places every word on this meaning-map by watching which words tend to appear near each other across millions of sentences.
- **Interview One-Liner:** Word2Vec learns dense word vectors by training a neural network to predict words from context — similar-meaning words end up with similar vectors, enabling semantic math like king - man + woman ≈ queen.

---

## spaCy vs NLTK
- **Definition:** Both are NLP libraries, but built for different purposes. NLTK = comprehensive research/teaching toolkit. spaCy = fast, production-ready pipeline.
- **Key Difference:** NLTK is like a huge textbook with every possible NLP experiment — great for LEARNING. spaCy is like a streamlined factory assembly line — built for SHIPPING real products fast.
- **spaCy Pipeline Components (`nlp.pipe_names`):** `tok2vec` → `tagger` → `parser` → `attribute_ruler` → `lemmatizer` → `ner` — each station adds more understanding to the text, in sequence.
- **When to Use Which:** Learn NLP concepts with NLTK-style teaching; build real chatbots/apps with spaCy.
- **Interview One-Liner:** NLTK is the research/teaching toolkit for learning NLP fundamentals; spaCy is the fast, production-grade library used for actually building real NLP applications.

---

## NLP Pipeline
- **Definition:** A series of ordered processing steps that convert raw text into structured, usable data a computer can work with — cleaning → tokenizing → normalizing → vectorizing → modeling.
- **Steps in Order:**
  1. Text Cleaning (remove noise, lowercase)
  2. Tokenization (split into tokens)
  3. Stop Word Removal (remove filler words)
  4. Stemming/Lemmatization (standardize word forms)
  5. Vectorization (convert to numbers — BoW, TF-IDF, or embeddings)
  6. Feed into Model
- **Explain Like a Beginner:** Like a restaurant kitchen assembly line — raw ingredients (text) pass through cleaning, cutting, marinating, and cooking stations before becoming a finished dish (numbers a model can use). Each station adds value; skipping one affects the final result.
- **Interview One-Liner:** An NLP pipeline is the ordered series of preprocessing steps (clean → tokenize → normalize → vectorize) that transforms raw text into machine-usable numbers — each step builds on the previous one.

---

## Attention Mechanism
- **Definition:** A neural network mechanism that lets each word in a sentence "look at" every other word and decide how much to focus on each one when building its own representation — instead of processing words in strict sequence.
- **When to Use:** Core component of Transformers — used in every modern LLM (GPT, BERT, Claude, etc.).
- **Why It Was Invented:** Old RNN/LSTM models processed words one at a time in order, forgetting early words in long sentences. Attention lets every word directly connect to every other word, regardless of distance.
- **Query, Key, Value (Q, K, V):**
  - **Query (Q):** What this word is "searching for" (its question)
  - **Key (K):** What each other word "offers" as a label/tag
  - **Value (V):** The actual content/information that word carries
  - Process: Q is matched against all Ks → scores computed → converted to weights via Softmax → weighted sum of Vs = final output
- **Multi-Head Attention:** Running self-attention MULTIPLE times in parallel, each head learning to focus on a different type of relationship (grammar, reference, tone, etc.) — then combining all outputs.
- **Explain Like a Beginner:** "The animal didn't cross the street because it was too tired." — when processing "it," your brain PAYS MORE ATTENTION to "animal" than "street" to figure out what "it" refers to. Attention does this mathematically for every word — each word scores every other word's relevance to understanding itself.
- **Interview One-Liner:** The Attention Mechanism lets each word directly attend to every other word in a sentence simultaneously (via Q/K/V matching), replacing the sequential word-by-word processing of older RNN models and enabling much better understanding of long-range dependencies.

---

## Transformer Architecture
- **Definition:** A neural network architecture built entirely around the Attention Mechanism — designed for understanding and generating sequences of text, and the foundation of every modern LLM.
- **Two Halves:**
  - **Encoder:** Reads and deeply understands the input (used alone in BERT)
  - **Decoder:** Generates output word-by-word (used alone in GPT/Claude)
- **Key Steps:**
  1. Tokenization + Embedding (words → vectors)
  2. Positional Encoding (adds word-order info, since all words processed simultaneously)
  3. Encoder: Multi-Head Self-Attention → Add & Normalize → Feed-Forward → Add & Normalize (repeated N times)
  4. Decoder: Masked Self-Attention (only looks backward) → Cross-Attention (glances at Encoder output) → Feed-Forward (repeated N times)
  5. Final output layer → probability distribution over vocabulary → pick next word
- **Key Terms:**
  - **Positional Encoding:** Adds unique position signal to each word's vector since all words are processed simultaneously (not in sequence)
  - **Masked Self-Attention:** Decoder only looks at words generated SO FAR — prevents "peeking" at future words
  - **Cross-Attention:** Decoder attends to Encoder's output — the bridge between understanding input and generating output
  - **Residual Connections (Add):** Original input + processed output combined — preserves information, prevents vanishing gradient
  - **Layer Normalization (Normalize):** Rescales numbers to keep them stable as they flow through many layers
- **Parallelization Advantage:** Since all words can attend to all others simultaneously (unlike RNNs), Transformers can be trained MUCH faster on GPUs.
- **Explain Like a Beginner:** A Transformer reads the whole sentence at once (like taking a photo of the whole page rather than reading letter-by-letter), lets every word "look at" every other word through Attention to understand relationships, refines this understanding through multiple stacked layers, then generates output one word at a time — each new word informed both by what's already been said AND by the original input's deep understanding.
- **Interview One-Liner:** A Transformer uses stacked multi-head self-attention layers to process all words simultaneously (replacing sequential RNNs), with an Encoder building deep input understanding and a Decoder generating output via masked self-attention + cross-attention — this architecture is the foundation of all modern LLMs.

---

## Hugging Face Transformers
- **Definition:** A Python library providing pre-trained Transformer models (BERT, GPT, T5, etc.) and easy-to-use tools for NLP tasks — so you can use powerful models without training from scratch.
- **Key Tool — `pipeline()`:** A high-level shortcut that automatically loads a suitable pretrained model for a named task.
- **Available Tasks:**
  - `"sentiment-analysis"` → positive/negative classification
  - `"text-generation"` → continues/completes text
  - `"ner"` → Named Entity Recognition
  - `"text-classification"` → general text categorization
- **Code Example:**
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
result = sentiment(["I love this product!", "This is terrible."])
print(result)
# [{'label': 'POSITIVE', 'score': 0.999}, {'label': 'NEGATIVE', 'score': 0.998}]
```
- **Key Insight:** `pipeline("sentiment-analysis")` is a shortcut that automatically downloads and loads a real pretrained model (like DistilBERT) — not magic, just a convenient wrapper.
- **Direct Model Loading:**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```
- **Interview One-Liner:** Hugging Face's `pipeline()` loads a real pretrained Transformer model behind the scenes for a given task — allowing you to use state-of-the-art NLP capabilities in one line without training anything yourself.

---

# WEEK 8 — LLM Fundamentals + Prompt Engineering

---

## Tokens and Context Window
- **Token Definition:** The basic unit an LLM processes — not always a whole word. Could be a word, part of a word, or punctuation. Rule of thumb: ~1 token ≈ 4 characters of English.
- **Example:** "tokenization" → "token" + "ization" (2 tokens). "cat" = 1 token.
- **Why Tokens Matter:** API pricing is measured in tokens; every model has a maximum TOKEN limit (context window), not a word or character limit.
- **Context Window Definition:** The maximum number of tokens (your input + prior conversation + any pasted documents, ALL combined) the model can "see" at once when predicting its next token.
- **Key Insight:** The model has NO memory beyond the context window — anything older than the window is literally invisible, not forgotten by choice.
- **Analogy:** A magic index card of limited size — once full, anything written earlier simply isn't visible anymore, even if the conversation happened.
- **Interview One-Liner:** A token is the basic text unit an LLM processes (~4 characters), and the context window is the hard limit on how many tokens the model can see at once — anything outside this window is completely invisible to the model.

---

## Next-Token Prediction
- **Definition:** The ONLY core mechanism an LLM uses — given everything visible so far, predict the single most likely NEXT token, add it, then repeat this process until the response is complete.
- **Key Insight:** There's no separate "thinking" or "reasoning" step happening outside this loop — even a complex, multi-paragraph answer is just this one prediction repeated thousands of times.
- **Connection to Transformers:** The Attention Mechanism (Week 7) is HOW the model decides what the next token should be — each prediction uses all visible tokens attending to each other via Q/K/V.
- **Explain Like a Beginner:** Like a very sophisticated autocomplete — it doesn't "think" about the whole answer at once, it just picks the next most likely word, then the next, then the next, until it naturally reaches a stopping point.
- **Interview One-Liner:** Next-token prediction is the single mechanism behind all LLM output — the model repeatedly predicts one token at a time based on everything in its context window, with no separate "reasoning" step outside this loop.

---

## LLM API Setup and Basic Call
- **Definition:** An API (Application Programming Interface) that lets you send prompts to a hosted LLM and receive responses back over the internet — without running the model locally.
- **Why Groq:** Free tier available, no credit card required, uses open-source models (Llama 3.3), and uses the same OpenAI-compatible API structure — good for learning.
- **Basic Code:**
```python
from groq import Groq
client = Groq(api_key="YOUR_GROQ_API_KEY")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "What is 2 + 2?"}]
)
print(response.choices[0].message.content)
```
- **Response Structure:** `response.choices[0].message.content` — `choices` is a list (supports multiple responses), `[0]` grabs the first, `.message.content` extracts the plain text.
- **Provider Compatibility Note:** Groq and OpenAI share the same `.chat.completions.create()` structure (many providers copy OpenAI's format). Gemini uses a COMPLETELY different structure — not just a name swap.
- **Interview One-Liner:** An LLM API sends your prompt to a hosted model over the internet and returns a generated response — the key structure is `messages=[{"role": ..., "content": ...}]` which supports multi-turn conversations by growing the message list.

---

## Conversation Memory
- **Definition:** LLMs have NO built-in memory between separate API calls — "memory" is simulated by re-sending the ENTIRE growing message history (user + assistant turns) on every call.
- **Why This Works:** The model's context window sees all past messages as if they're happening right now — it's not "remembering," it's re-reading the full transcript each time.
- **Code Example:**
```python
conversation_history = []

def chat(user_message):
    conversation_history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

print(chat("My name is Gireesh."))
print(chat("What's my name?"))  # correctly says "Gireesh"
```
- **Key Rule:** Always append BOTH user messages AND assistant replies to the history — otherwise the model won't know what it previously said, causing incoherence.
- **Connection to Streamlit:** `st.session_state.messages` in Zoro v2 serves the EXACT same purpose — persistent storage of the growing conversation list across Streamlit reruns.
- **Interview One-Liner:** LLM conversation memory is an illusion maintained by re-sending the entire conversation history on every API call — the model has no actual memory, it just re-reads the growing transcript each time.

---

## System Prompt vs User Prompt
- **Definition:** Three roles exist in an LLM conversation — `"system"` (private instructions shaping the model's behavior/identity), `"user"` (the human's actual message), and `"assistant"` (the model's replies).
- **System Prompt:** Set ONCE at the start, before any conversation — defines who the model is, what it knows, how it should behave, and what it should/shouldn't do. Invisible to the end user.
- **Key Power:** The model itself doesn't change between calls — ONLY the system prompt changes. This means the same underlying model can behave as a customer support bot, a code assistant, or a creative writer just by changing the system prompt text.
- **Code Example:**
```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are Zoro, a friendly e-commerce assistant. Cash on delivery IS available."},
        {"role": "user", "content": "Do you offer cash on delivery?"}
    ]
)
```
- **Without System Prompt:** Model says "I'm just an AI, I don't have store information."
- **With System Prompt:** Model confidently says "Yes, cash on delivery is available!"
- **Zoro v2 Application:** System prompt contained store categories, 20 FAQ pairs, scope restriction rules, anti-hallucination instruction, and style guidelines — all working together.
- **Interview One-Liner:** The system prompt is a private "briefing" given to the model before any conversation begins — it defines identity, knowledge, tone, and boundaries without the user seeing it, and is the primary mechanism for customizing LLM behavior for specific applications.

---

## Zero-Shot vs Few-Shot Prompting
- **Zero-Shot Definition:** Asking the model to perform a task with NO examples — relying entirely on its general training knowledge to understand what you want.
- **Few-Shot Definition:** Providing a few worked examples of the task BEFORE the actual question — showing the model the pattern, format, and style you want it to follow.
- **Key Practical Difference:** Zero-shot often gives verbose, explanatory answers. Few-shot constrains the FORMAT to match your examples — same correct answer, but cleaner and more consistent structure.
- **Code Example (Few-Shot):**
```python
prompt = """Classify each message as 'Complaint', 'Question', or 'Compliment'.

Message: "My order arrived damaged."
Category: Complaint

Message: "Do you ship to Kerala?"
Category: Question

Message: "The delivery was so fast, thank you!"
Category:"""
```
- **Why Few-Shot Format Works:** The model completes the pattern — it sees "Category: [label]" twice, so it continues with just the label, not an explanation.
- **When to Use Which:** Zero-shot for tasks the model handles well generally. Few-shot when you need specific output FORMAT or when zero-shot gives inconsistent results.
- **Interview One-Liner:** Zero-shot asks for a task with no examples (relies on general training); few-shot provides worked examples first to constrain output format and style — few-shot is particularly valuable when you need consistent, structured output rather than just a correct answer.

---

## Chain-of-Thought (CoT) Prompting
- **Definition:** Explicitly asking the model to reason step-by-step BEFORE giving a final answer — improves accuracy on complex, multi-step reasoning tasks and makes the reasoning transparent and verifiable.
- **Why It Works:** When a model writes reasoning steps first, each step becomes context for the next — "showing its work" helps it arrive at better answers than jumping straight to a guess.
- **Key Real Finding:** In our own testing, both direct and CoT prompting got the same correct answer on a math problem — but only CoT revealed WHERE a judgment call (rounding) was made, making it verifiable. CoT's value is transparency and reliability on HARDER problems, not necessarily always being "more correct."
- **Code Example:**
```python
# Direct (no CoT)
"A store had 120 items. Sold 35% Monday, 20% of remaining Tuesday. How many left? Just give the number."

# Chain-of-Thought
"A store had 120 items. Sold 35% Monday, 20% of remaining Tuesday. How many left? Think step by step before answering."
```
- **Interview One-Liner:** Chain-of-Thought prompting instructs the model to reason step-by-step before answering — it improves reliability on complex tasks and makes reasoning transparent and verifiable, even when both approaches reach the same answer.

---

## Temperature and max_tokens
- **Temperature Definition:** Controls randomness in the model's token selection. Low (0-0.3) = near-deterministic, consistent answers. High (0.8-1.5) = more varied, creative, unpredictable answers.
- **Real Test Result:** "Name one random animal" — temperature 0.0 gave "Kangaroo" both times. Temperature 1.5 gave "Quokka," "Kangaroo," "Koala" across three runs. Confirmed temperature genuinely works.
- **Important Nuance:** Temperature's effect can be HIDDEN if the prompt has one overwhelmingly dominant "correct" answer (e.g., "facts about the ocean") — the model keeps gravitating to that dominant answer even at high temperature. Use prompts with many equally valid options to actually SEE temperature's effect.
- **When to Use Low Temperature:** Customer support bots, factual Q&A, classification tasks — anywhere consistency matters more than creativity.
- **When to Use High Temperature:** Creative writing, brainstorming, generating varied options.
- **max_tokens Definition:** A HARD cap on how many tokens the model generates in its response. Cuts off mid-sentence once reached — not a suggestion, a firm limit.
- **Code Example:**
```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Write one sentence about the ocean."}],
    temperature=0.1,   # low = consistent
    max_tokens=30      # hard cutoff at 30 tokens
)
```
- **Interview One-Liner:** Temperature controls response randomness (low = consistent/deterministic, high = varied/creative); max_tokens is a hard ceiling on response length — both are key parameters for tuning LLM output behavior for specific use cases.

---

## Hallucination
- **Definition:** When an LLM confidently generates false, invented information that sounds plausible but doesn't exist in reality or in the provided context.
- **Real Example from Zoro v2:** When asked for book recommendations, an early version confidently invented a "Staff Picks" section and specific genres that were never mentioned in the system prompt.
- **Why It Happens:** LLMs are trained to generate plausible-sounding text — when they lack specific information, they "fill in the gaps" with statistically likely content rather than admitting ignorance.
- **Fix:** Explicit instruction in system prompt — "NEVER invent details not present in the information provided. If you don't know, say so."
- **Interview One-Liner:** Hallucination is when an LLM generates confident-sounding but factually false information — prevented by explicit anti-hallucination instructions in the system prompt and by using RAG to ground answers in retrieved real content.

---

## Scope Restriction
- **Definition:** Without explicit boundaries, an LLM assistant answers ANY request like a general-purpose model — writing code, essays, or anything else — instead of staying in its intended role.
- **Real Example from Zoro v2:** Without scope restriction, Zoro would write Python code or full essays when asked, completely breaking its customer support character.
- **Fix:** System prompt instruction — "You ONLY help with questions related to this store. If asked something unrelated, politely decline and redirect."
- **Interview One-Liner:** Scope restriction uses system prompt instructions to keep an LLM assistant focused on its intended domain — without it, the model behaves as a general-purpose assistant regardless of the character/role defined for it.

---

# WEEK 9 — RAG + Vector Databases

---

## Sentence and Document Embeddings
- **Definition:** Vector representations of WHOLE sentences or documents (not just individual words) — where sentences with similar MEANING end up with numerically close vectors, even using completely different words.
- **Why Not Just Average Word Vectors:** Simple averaging loses word order — "Dog bites man" and "Man bites dog" get the IDENTICAL averaged vector even though they mean opposite things. Proper sentence embedding models (like SBERT) preserve context.
- **How SBERT Works:** A Transformer model (same Attention architecture as Week 7) specifically FINE-TUNED to produce one meaningful vector per sentence — not averaging, but genuine contextual understanding of the whole sentence at once.
- **Pipeline:** Raw text → Tokenize → Embed each word → Transformer processes all together → Output ONE vector for the whole sentence
- **Why This Matters for RAG:** RAG's retrieval step compares the user's QUESTION vector against stored DOCUMENT CHUNK vectors — if these embeddings don't capture meaning properly, retrieval fails (same "money back" vs "refund" problem as Zoro v1, but at the sentence level).
- **Interview One-Liner:** Sentence embeddings produce a single vector representing the meaning of an entire sentence — models like SBERT use Transformer architecture to preserve context and word order, unlike naive word-vector averaging which loses sentence structure entirely.

---

## Vector Databases
- **Definition:** A specialized database built specifically to store and efficiently search through large numbers of vectors — finding the most semantically SIMILAR ones to a query vector, using Approximate Nearest Neighbor (ANN) algorithms.
- **Why Not Regular SQL:** SQL is optimized for EXACT matches (`WHERE price = 500`). Vector similarity search needs "find the vector closest in MEANING to this query" — a fundamentally different operation SQL was never built for.
- **ANN (Approximate Nearest Neighbor):** Instead of comparing every vector one-by-one (slow for millions), ANN algorithms pre-organize vectors into smart structures and only check a relevant subset — trading a tiny bit of perfect accuracy for massive speed gains.
- **Why "Approximate":** Deliberately sacrifices perfect mathematical accuracy for speed — for most real applications, finding the "very likely closest" match instantly beats finding the "guaranteed exact closest" match slowly.
- **Three Key Options:**
  - **FAISS:** Raw speed LIBRARY (not a full database) by Meta — extremely fast, GPU support, but YOU build ID mapping, persistence, and metadata yourself. Best for massive scale.
  - **ChromaDB:** Full, easy-to-use local DATABASE — handles embeddings, IDs, metadata, persistence automatically. Best for prototypes, learning, small-to-medium apps.
  - **Pinecone:** Fully managed CLOUD-HOSTED service — you don't run it yourself. Best for production apps at scale.
- **Interview One-Liner:** A vector database stores embeddings and finds the most semantically similar ones to a query using ANN algorithms — unlike SQL which does exact matches, vector DBs do meaning-based similarity search, trading a tiny accuracy margin for massive speed gains over brute-force comparison.

---

## FAISS vs ChromaDB vs Pinecone
- **FAISS:** Library only (not full database). Gives you fast ANN search — nothing else. You build: embedding generation, ID-to-text mapping, persistence (save/load index to disk), metadata filtering, and index TYPE selection yourself. Advantage: raw speed ceiling, GPU support. Use when: massive scale (millions/billions of vectors), maximum performance needed.
- **ChromaDB:** Full batteries-included local database. Auto-generates embeddings, manages IDs, stores metadata, handles persistence. Advantage: fastest to get started, most beginner-friendly. Use when: learning, prototypes, small-to-medium apps (your Zoro v3 use case).
- **Pinecone:** Managed cloud service — runs on their servers. Advantage: no infrastructure to manage, scales automatically. Use when: production apps with real traffic needing reliability and scale. Usually paid for serious usage.
- **Analogy:** FAISS = industrial kitchen (powerful raw tools, you set up everything). ChromaDB = furnished home kitchen (ready to use immediately). Pinecone = restaurant chain (full professional infrastructure, you just plug in).
- **Interview One-Liner:** FAISS is a raw speed library requiring manual setup of everything else; ChromaDB is a full local database with batteries included; Pinecone is a managed cloud service — choose based on scale needs and willingness to manage infrastructure.

---

## RAG (Retrieval-Augmented Generation)
- **Definition:** A technique that improves LLM answers by first RETRIEVING relevant information from an external document store, then using that retrieved content to GROUND the LLM's generation — combining the best of document search and language generation.
- **The Core Problem RAG Solves:** LLMs are trained once with a fixed knowledge cutoff — they can't answer accurately about things that changed after training, or about YOUR specific private data (like a company's return policy).
- **Key Insight — No Retraining Needed:** The LLM itself never changes. Only the DOCUMENT STORE gets updated. Anyone can add/edit/delete documents in the vector database instantly — Zoro v3 automatically answers from new policy documents without any retraining.
- **Open-Book Exam Analogy:** An LLM without RAG = closed-book exam (answers from memory, which may be outdated). RAG = open-book exam (quickly checks the actual textbook before answering, so the answer is grounded in current, real content).
- **Full Pipeline:**
```
Document → Chunk into paragraphs → Embed each chunk → Store in ChromaDB
                                                              ↓
User Question → Embed question → Search ChromaDB for similar chunks
                                                              ↓
Retrieved chunks + User question → Prompt to LLM → Grounded answer
```
- **RAG vs Fine-tuning:**
  - **RAG:** Update KNOWLEDGE instantly (edit documents, no retraining) — best for frequently changing facts.
  - **Fine-tuning:** Change model BEHAVIOR/STYLE permanently (requires retraining) — best for teaching a new skill or writing style, not facts.
- **What Happens If Retrieval Fails:** If ChromaDB retrieves irrelevant chunks (bad query wording, missing coverage), the LLM will still try to answer from those wrong chunks — "garbage in, garbage out." Retrieval quality directly determines answer quality.
- **Interview One-Liner:** RAG retrieves relevant document chunks from a vector database based on semantic similarity, then feeds those chunks alongside the user's question to an LLM — grounding the answer in real, current documents rather than the model's fixed training knowledge.

---

## Chunking
- **Definition:** The process of splitting a large document into smaller, focused pieces (chunks) before embedding and storing in a vector database — so retrieval can pinpoint the specific relevant section, not blend everything together.
- **Why Not Store the Whole Document as One Vector:** A single embedding of a 50-page document blends ALL topics together — a question about "delivery timing" would match weakly against a giant blended vector instead of strongly against the specific delivery paragraph.
- **Simple Chunking (used in Zoro v3):**
```python
chunks = document.strip().split("\n\n")  # split on blank lines (paragraph breaks)
```
- **Why Paragraph-Based Works for Clean Documents:** Each paragraph in our `store_knowledge.txt` covers exactly one topic — splitting by blank lines gives clean, topic-focused chunks.
- **Real-World Note:** Messy PDFs and unstructured text don't have clean paragraph breaks — more advanced methods use fixed character counts with overlap, or sentence-boundary detection. LangChain's `CharacterTextSplitter` handles this for complex documents.
- **Interview One-Liner:** Chunking splits documents into focused, retrievable pieces before embedding — smaller chunks enable precise topic-level retrieval instead of blending unrelated content into one undifferentiated vector.

---

## ChromaDB — Hands-On
- **Definition:** A full-featured, easy-to-use vector database that handles embedding generation, storage, ID management, metadata, and similarity search automatically.
- **Key Methods:**
  - `client.create_collection(name="...")` — creates a new collection (like a table in SQL)
  - `client.get_or_create_collection(name="...")` — gets existing OR creates new (safe for Streamlit reruns)
  - `collection.add(documents=[...], ids=[...])` — stores documents (auto-embeds them)
  - `collection.query(query_texts=[...], n_results=2)` — finds most similar chunks
  - `collection.count()` — returns number of stored documents
- **IDs Explained:** Every document MUST have a unique ID string — matched to `documents` list by POSITION. Purpose: reliable lookup/update/delete later. Can be descriptive ("return_policy") or numeric ("0", "1", "2").
- **Output Structure:**
```python
{'ids': [['return', 'payment']], 
 'documents': [['Return Policy text...', 'Payment Methods text...']], 
 'distances': [[1.18, 1.55]]}
```
- **Distances:** LOWER distance = MORE similar (closer in meaning-space). Opposite of similarity score.
- **`@st.cache_resource` for Streamlit:** Since Streamlit reruns the entire script on every interaction, use this decorator to ensure ChromaDB setup (collection creation + document loading) runs ONLY ONCE, not on every rerun.
```python
@st.cache_resource
def setup():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="store_knowledge")
    # load, chunk, add documents
    return collection
collection = setup()
```
- **Persistence:** Default `chromadb.Client()` is in-memory (lost when session ends). Use `chromadb.PersistentClient(path="...")` to save to disk — point path to Google Drive in Colab for cross-session persistence.
- **Interview One-Liner:** ChromaDB is a batteries-included vector database that auto-generates embeddings and handles storage, IDs, metadata, and ANN similarity search — making it the fastest way to add vector retrieval to a Python application.

---

## Full RAG Pipeline — Zoro v3
- **Definition:** The complete, production-style RAG pipeline combining document chunking, ChromaDB retrieval, and Groq LLM generation — built without frameworks, understanding every component.
- **Complete Code Pattern:**
```python
# Setup: load, chunk, store (run once)
with open("store_knowledge.txt", "r") as f:
    document = f.read()
chunks = document.strip().split("\n\n")
collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

# Per-query RAG function
def rag_answer(user_question):
    # Retrieve
    results = collection.query(query_texts=[user_question], n_results=2)
    retrieved_chunks = results['documents'][0]
    
    # Build grounded prompt
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer using ONLY the information below. If not in the information, say you don't know.
Information: {context}
Question: {user_question}"""
    
    # Generate
    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
```
- **Real Test Results vs v1:**
  - "How do I get my money back?" → v1 ❌ (zero word overlap with "refund") → v3 ✅ (semantic retrieval)
  - "Is there a guarantee on electronics?" → v1 ❌ ("guarantee" vs "warranty") → v3 ✅
  - "Can I send the item back?" → v1 ❌ ("send back" vs "return") → v3 ✅
  - Gibberish → v3 ✅ ("I don't know")
  - "Do you sell laptops?" → v3 ✅ ("I don't know" — respected instruction even when wrong chunk retrieved)
- **"ONLY use provided information" Instruction:** Critical anti-hallucination guard — even when ChromaDB retrieves a loosely-related wrong chunk, this instruction prevents the LLM from inventing an answer.
- **Interview One-Liner:** A RAG pipeline separates knowledge (stored in a vector database) from generation (handled by an LLM) — retrieval finds semantically relevant document chunks per query, which are then passed to the LLM as grounding context, preventing hallucination and enabling knowledge updates without any model retraining.

---

*End of Weeks 7-9 Notes*
*Generated from: actual conversations, debugging sessions, and hands-on project building*
*Purpose: Study revision + RAG source document for Personal AI Career Assistant*
