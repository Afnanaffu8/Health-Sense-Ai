# 🩺 HealthSense AI

An educational healthcare application that combines machine learning, medical report analysis, hospital recommendations, and an AI-powered health assistant in one platform.

HealthSense AI helps users explore potential diabetes risk, understand basic medical report information, identify relevant medical specialists, and ask general health-related questions through text or voice.

**🚀 Live Demo:** https://health-sense-ai.streamlit.app/

> ⚠️ **Medical Disclaimer:** HealthSense AI is an educational project, not a medical diagnostic tool. Its predictions and AI-generated responses are not medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

<## 📸 Application Screenshots  
1.### Home Page
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1d4c39e1-2f02-4c87-ba4d-7d8315a60726" />
2.### Health Risk Analysis
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/07b8ddb3-1954-4ec0-8d8a-b9e2cfe59100" />
3.### Medical Report Analyzer
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6ae9cf29-745f-448f-b0ea-f5753f257baf" />
4.### AI Health Assistant
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/122bd31e-fc92-4618-968b-9e888776da61" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f01bcd52-1d59-4afc-8c98-5b3f8e853114" />




---

## ✨ Features

| Module | Description |
|---|---|
| **Health Risk Analysis** | Collects health inputs such as age, weight, height, glucose, and blood pressure to calculate BMI and estimate diabetes risk using a machine learning model. |
| **Risk Assessment Results** | Displays a risk category (Low, Moderate, or High), a visual risk indicator, and educational health information. |
| **Doctor Recommendations** | Suggests relevant medical specialties based on the assessment and helps users find matching hospitals in Indian cities. |
| **Medical Report Analyzer** | Accepts PDF, DOCX, and TXT medical reports, extracts relevant health information, and presents a simplified summary. |
| **AI Health Assistant** | Allows users to ask health-related questions through text or voice and receive AI-assisted responses based on a built-in health knowledge base. |
| **Voice Interaction** | Supports voice input and spoken responses where the required voice components are available. |
| **Emergency Awareness** | Identifies potentially urgent situations in conversations and advises users to seek appropriate medical attention. |

## 🧠 How It Works

### Health Risk Assessment

1. The user enters basic health information.
2. The application calculates BMI and processes the selected health indicators.
3. A trained Random Forest model estimates diabetes risk.
4. The application displays the estimated risk category and relevant specialist recommendations.

### Medical Report Analysis

1. The user uploads a medical report.
2. The application extracts readable text from the supported document.
3. Relevant health indicators are identified where available.
4. The extracted information is presented in a simplified summary with related health information.

### AI Health Assistant

1. The user asks a question using voice or text.
2. The application searches its built-in health knowledge base using TF-IDF retrieval.
3. Relevant information is passed to the available AI engine.
4. The assistant generates a response, with emergency-awareness checks and specialist suggestions where appropriate.

## ⚙️ Technology and Implementation

- **Machine Learning:** Random Forest classifier for diabetes risk estimation.
- **Data Processing:** Python and pandas for handling health information.
- **Information Retrieval:** TF-IDF vectorization and cosine similarity for retrieving relevant health knowledge.
- **AI Integration:** Ollama for local language-model inference or Groq for cloud-based inference.
- **Document Processing:** pypdf and python-docx for extracting text from medical reports.
- **Application Interface:** Streamlit for the interactive web application.
- **Hospital Directory:** CSV-based hospital information organized by city and medical specialty.

## 📊 Model Performance

The diabetes risk model was evaluated using a held-out test dataset.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 0.68 |
| ROC-AUC | 0.74 |
| Recall (High-Risk Class) | 0.68 |
| Precision (High-Risk Class) | 0.52 |

**Model limitations:**

- The model estimates diabetes risk only; it does not predict other diseases.
- It uses a limited set of health indicators.
- The training dataset is based on the Pima Indians Diabetes dataset and may not represent all populations.
- Model performance is not sufficient for clinical decision-making.
- Hospital and department information should be independently verified before visiting.

These results are intended for educational evaluation, not clinical use.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Application | Streamlit |
| Machine Learning | scikit-learn, Random Forest |
| Data Processing | pandas |
| Information Retrieval | TF-IDF, cosine similarity |
| AI / LLM | Ollama, Groq |
| Document Processing | pypdf, python-docx |
| Voice Features | streamlit-mic-recorder, faster-whisper (optional) |
| Data Storage | CSV and local project files |
| Deployment | Streamlit Community Cloud |

## 💻 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/Afnanaffu8/Health-Sense-Ai.git
cd Health-Sense-Ai
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the machine learning model**

```bash
python model.py
```

**5. Start the Streamlit application**

```bash
python -m streamlit run app.py
```

The application should open in your browser at:

`http://localhost:8501`

## 🤖 AI Configuration

HealthSense AI supports different AI execution options.

**Option 1: Local AI with Ollama**

- Install Ollama from https://ollama.com/
- Download a supported model:

```bash
ollama pull qwen2.5:3b
```

- Start the Ollama service and configure the application to use the installed model.

**Option 2: Cloud AI with Groq**

- Create an API key at https://console.groq.com/keys
- Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

- Never upload API keys or secrets to GitHub.

**Option 3: Knowledge Base Fallback**

The application can provide knowledge-base-based responses when an external or local LLM is unavailable, depending on the configured fallback implementation.

## 📁 Project Structure

```text
Health-Sense-Ai/
│
├── app.py                 # Main Streamlit application
├── model.py               # Machine learning model training and prediction
├── rag_engine.py          # Health knowledge retrieval engine
├── dataset.csv            # Diabetes dataset
├── doctors.csv            # Hospital directory
├── requirements.txt       # Python dependencies
│
├── knowledge/             # Health knowledge base
├── assets/                # Application images and visual assets
└── .streamlit/             # Streamlit configuration and secrets
```

*The structure above represents the main project components; additional files may be present in the repository.*

## 🔮 Future Enhancements

- SHAP-based model explainability and feature importance visualization.
- Database integration for securely managing assessment history.
- FastAPI backend integration.
- Automated testing and improved validation.
- Docker-based deployment.
- Expansion to additional health-risk prediction models.
- Improved medical knowledge retrieval and multilingual voice interaction.

## 👨‍💻 Author

**Afnan Qureshi**

Built as an educational project exploring the integration of machine learning, healthcare analytics, document processing, and AI-powered conversational systems.

**Project Repository:** https://github.com/Afnanaffu8/Health-Sense-Ai/

**Live Application:** https://health-sense-ai.streamlit.app/

---

*HealthSense AI — Bringing Machine Learning and AI-Assisted Health Information Together.*
