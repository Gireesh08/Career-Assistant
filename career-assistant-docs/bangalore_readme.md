# 🏠 Bangalore House Price Prediction

A full stack Machine Learning web application that predicts house prices in Bangalore based on location, square feet, BHK and number of bathrooms!

🔗 **Live Demo → [Try it here!](https://bangalore-house-price-prediction-v7ht.onrender.com)**

---

## 📸 Preview

> Enter location, square feet, BHK and bathrooms → get instant price prediction!

---

## 🎯 What This Project Does

- Takes user inputs → location, square feet, BHK, bathrooms
- Runs them through a trained Machine Learning model
- Predicts the house price in Lakhs instantly!

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Scikit-learn, NumPy, Pandas |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript, jQuery |
| Deployment | Render |
| Development | Google Colab, VS Code |

---

## 📊 Model Performance

| Model | Accuracy (R² Score) |
|-------|-------------------|
| **Linear Regression** | **81.8% ✅ Winner!** |
| Decision Tree | 71.8% |
| Lasso Regression | 68.7% |

> GridSearchCV was used to find the best model and hyperparameters automatically!

---

## 🔄 Project Workflow

```
Raw Data
   ↓
Data Cleaning & EDA
   ↓
Feature Engineering
   ↓
Outlier Removal
   ↓
Model Training (GridSearchCV)
   ↓
Flask API (Backend)
   ↓
HTML/CSS/JS (Frontend)
   ↓
Deployed on Render 🌍
```

---

## 📁 Project Structure

```
BHP/
│
├── 📁 Client/
│   ├── app.html          → Main webpage
│   ├── app.css           → Styling
│   └── app.js            → Flask API connection
│
├── 📁 Model/
│   ├── House_Prediction.ipynb        → Full ML notebook
│   ├── columns.json                  → Feature columns
│   └── banglore_home_prices_model.pickle  → Trained model
│
├── 📁 Server/
│   ├── Server.py         → Flask routes
│   └── util.py           → Model loading & prediction
│
├── 📄 requirements.txt   → Python dependencies
├── 📄 Procfile           → Render deployment config
└── 📄 README.md          → You are here!
```

---

## 🚀 How to Run Locally

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/Gireesh08/Bangalore-house-price-prediction.git
cd Bangalore-house-price-prediction
```

**Step 2 — Install dependencies:**
```bash
pip install flask numpy scikit-learn
```

**Step 3 — Run the Flask server:**
```bash
python Server/Server.py
```

**Step 4 — Open in browser:**
```
http://127.0.0.1:5000
```

---

## 🧹 Data Cleaning Steps

- Removed null values and irrelevant columns
- Converted `total_sqft` ranges like `1000-1200` to average `1100`
- Engineered `price_per_sqft` feature
- Removed price per sqft outliers location-wise (mean ± std)
- Removed BHK outliers where larger BHK cost less than smaller BHK
- Applied One Hot Encoding for locations using `get_dummies`

---

## 📦 Dependencies

```
flask
numpy
scikit-learn
pandas
matplotlib
seaborn
```

---

## 🔗 Links

- 🌐 **Live App** → [bangalore-house-price-prediction-v7ht.onrender.com](https://bangalore-house-price-prediction-v7ht.onrender.com)
- 📓 **Kaggle** → [gireeshadireddi](https://www.kaggle.com/gireeshadireddi)
- 💼 **LinkedIn** → [Gireesh Adireddi](https://www.linkedin.com/in/gireesh-adireddi-071362284/)
- 🐙 **GitHub** → [Gireesh08](https://github.com/Gireesh08)

---

## 👨‍💻 Author

**Gireesh Adireddi**

> *"Built this as part of my ML learning journey — from data cleaning to live deployment in 3 weeks!"*

---

⭐ **If you found this helpful, please star the repo!**
