# 🩺 HealthSense AI: Health Risk Prediction & Analytics

An educational healthcare app that estimates diabetes risk with machine learning, reads medical reports, suggests the right type of specialist and matching hospitals in India, and includes an AI health assistant that answers from a built-in knowledge base.

**🚀 Live demo:** https://health-risk-prediction-ai-bzhsb5qsdg8zf2kmvprxtz.streamlit.app/

> ⚠️ **Disclaimer:** This is an educational project. It does not diagnose any condition and is not medical advice. Always consult a qualified doctor.

<!-- Add 3 screenshots here: Home, Risk Analysis results, AI Chatbot -->

## ✨ Features

| Page | What it does |
|---|---|
| **Health Risk Analysis** | Enter age, weight, height, glucose and blood pressure. Get BMI, an ML risk estimate (Low / Moderate / High) with a risk bar, and a threshold-based screen. |
| **Doctor recommendations** | Names the specialist that fits your results (for example Endocrinologist or Cardiologist) and shows the top matching hospital in your city, with alternatives and a map link. |
| **Medical Report Analyzer** | Upload a PDF, DOCX or TXT report. Finds age, BMI, glucose and BP, estimates risk, summarises it in plain language, and shows related health facts. |
| **AI Health Assistant** | Chat by text or voice. Answers use a health knowledge base (retrieval) plus an LLM, show doctor suggestions when asked, warn about emergencies, and can read replies aloud. |

## 🧠 How it works

```
User input / report ──► Risk model (Random Forest) ──► Risk level + specialist advice
                                                     └─► Hospital directory (by city)
User question ──► Retrieval over knowledge base (TF-IDF) ──► LLM (Ollama locally, or Groq in the cloud)
                                                             └─► Emergency check + doctor suggestions
```

- **Risk model:** Random Forest trained on the public *Pima Indians Diabetes* dataset (724 rows after removing missing values). Inputs: age, BMI, glucose, diastolic blood pressure.
- **Retrieval:** health notes in `knowledge/` are split into passages and ranked with TF-IDF cosine similarity. The best passages are given to the LLM so answers stay grounded.
- **AI engine:** uses local **Ollama** when available, otherwise **Groq** (cloud) if a key is set, otherwise answers directly from the knowledge base.
- **Doctor directory:** `doctors.csv` lists 121 well-known hospitals in 42 Indian cities with their relevant departments and a Google Maps link. It lists hospital departments, not individual doctors.

## 📊 Model performance

Evaluated on a 20% held-out test set (145 patients):

| Metric | Score |
|---|---|
| Accuracy | 0.68 |
| ROC-AUC | 0.74 |
| Recall (high-risk class) | 0.68 |
| Precision (high-risk class) | 0.52 |

**Limitations (please read):**
- It predicts **diabetes risk only**, from just four inputs.
- The dataset is small and comes from one population (Pima women, 21 and older), so results may not generalise to everyone.
- Scores are modest. Treat the output as an educational estimate, never as a diagnosis.
- Hospital and city pairs should be verified before relying on them.

## 🛠️ Tech stack

Python · Streamlit · scikit-learn · pandas · TF-IDF retrieval · Ollama / Groq (LLMs) · pypdf · python-docx

## 💻 Run locally

```bash
git clone https://github.com/Afnanaffu8/health-risk-prediction-ai.git
cd health-risk-prediction-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python model.py                # trains the model and prints metrics
python -m streamlit run app.py
```

**AI options (pick one, or neither):**
- **Local AI (private):** install [Ollama](https://ollama.com), then run `ollama pull llama3.2:3b`.
- **Cloud AI:** create a free key at console.groq.com/keys, then create `.streamlit/secrets.toml` containing `GROQ_API_KEY = "your-key"`. Never commit this file.
- **Neither:** the chatbot falls back to the built-in knowledge base.

Optional voice input: `pip install streamlit-mic-recorder faster-whisper`.

## 📁 Project structure

```
app.py            # Streamlit app (UI, pages, AI routing)
model.py          # Training, saving and prediction of the risk model
rag_engine.py     # Retrieval over the knowledge base
dataset.csv       # Training data (Pima Indians Diabetes, cleaned)
doctors.csv       # Hospital directory by city and specialty
knowledge/        # Health notes used for retrieval
assets/           # Images
requirements.txt
```

## 🔮 Roadmap

- Explainability chart (SHAP) showing why a result was given
- FastAPI backend and database for assessment history
- Unit tests, Docker, and CI
- Heart and kidney risk models

## 👤 Author

**Afnan**: Built as a learning project in AI and healthcare technology.
