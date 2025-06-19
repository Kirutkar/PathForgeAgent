# 🧠 PathForge Agent App

PathForge Agent App is an AI-powered resume and career assistant that uses multiple LangChain agents to help users analyze and enhance their resumes, discover career roles, and receive personalized guidance.

🔗 [Live App on Hugging Face]https://huggingface.co/spaces/Kiruthikaramalingam/PathForgeAgent

---

## 🚀 Features

- 📄 **Upload Resume (PDF)** to:
  - Get Resume Strength (Weak / Average / Strong) – ML-powered (XGBoost)
  - Predict Suitable Job Role – GPT-based agent
  - Receive Structured Resume Feedback – GPT-based agent

- 💬 **Ask Questions**:
  - "How strong is my resume?"
  - "What role suits my skills?"
  - "How can I improve?"
  - "Will I get a job in data science as a fresher?"

- ✅ Multi-Agent Logic using LangChain with `ZERO_SHOT_REACT_DESCRIPTION`
- 🤖 Integrates XGBoost, LangChain Tools, OpenAI GPT-4o
- 🎨 Clean and interactive Gradio UI
- ☁️ Deployed on Hugging Face Spaces

---

## 🧠 Agents Used

| Agent | Description | Type |
|-------|-------------|------|
| **Resume Strength Tool** | Predicts resume strength using ML (XGBoost) | `Tool.from_function` |
| **Role Prediction Tool** | GPT-based role classification from a defined role list | LangChain Tool |
| **Resume Feedback Tool** | GPT-based detailed resume review and improvements | LangChain Tool |
| **Career Mentor Agent** | GPT-based Q&A for open-ended career questions | LangChain Tool |

---

## 🛠 Tech Stack

- `LangChain`, `OpenAI GPT-4o`
- `XGBoost`, `Scikit-learn`, `PyMuPDF (fitz)`
- `Gradio`, `Joblib`, `NumPy`
- Hosted on Hugging Face Spaces

---

## 📦 How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

---

## 💡 Example Queries

- “Can you analyze my resume?”
- “How to improve my resume to get into AI roles?”
- “What role fits me based on my uploaded resume?”
- “What certifications can I take to boost my chances?”

---

## 👩‍💻 Built by
**Kiruthika Ramalingam**  
🚀 AI Agent Challenge Participant – Decoding Data Science  
🌍 [LinkedIn]www.linkedin.com/in/kiruthika-ramalingam
