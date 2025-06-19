import fitz
import joblib
import numpy as np
import gradio as gr
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import openai
import os


openai.api_key=os.getenv("OPENAI_API_KEY")

# STEP 2: Load model and vectorizer
model = joblib.load("xgb_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Hybrid thresholds
q_low = 0.5166
q_high = 2.8319

# Weighted keywords
weighted_keywords = {
    'llm': 3.5, 'langchain': 3.5, 'openai': 3, 'data analysis': 2,
    'sql': 2, 'teaching': 3, 'crm': 3, 'project management': 3.5
}

# Resume text extraction
def extract_resume_text(file):
    doc = fitz.open(file.name)
    return " ".join([page.get_text() for page in doc]).strip()

# Resume strength
def predict_strength(resume_text):
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]
    score = sum(weight for kw, weight in weighted_keywords.items() if kw in resume_text.lower())
    norm_score = score / np.log(len(resume_text.split()) + 1)
    if prediction == 'Average' and norm_score >= q_high:
        prediction = 'Strong'
    elif prediction == 'Average' and norm_score < q_low:
        prediction = 'Weak'
    return f"✅ Resume Strength: {prediction}"

# Job role
def predict_role(resume_text):
    roles = ["AI Engineer", "Data Scientist", "Project Manager", "Teacher", "Sales Executive"]
    prompt = f"""
You are a job role classification expert. Pick one best-fit role from the list: {', '.join(roles)}.
Resume:
{resume_text}
Only respond with a single job role.
"""
    response = ChatOpenAI(model="gpt-4o", openai_api_key=openai.api_key).invoke(prompt)
    return f"🧩 Predicted Role: {response.content.strip()}"

# Feedback logic
def gpt_resume_feedback(resume_text):
    prompt = f"""
You are an expert resume reviewer. Provide structured feedback.
Resume:
{resume_text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# STEP 3: Tools
strength_tool = Tool.from_function(predict_strength, name="Strength Tool", description="ML resume strength")
role_tool = Tool.from_function(predict_role, name="Role Tool", description="GPT role classifier")
feedback_tool = Tool.from_function(gpt_resume_feedback, name="Feedback Tool", description="GPT resume feedback")

llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai.api_key)
agent_executor = initialize_agent(
    tools=[strength_tool, role_tool, feedback_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ STEP 4: Main routing logic (with Career Guidance Tool)
def gpt_career_guidance(resume_text="", question=""):

    openai_api_key=openai.api_key
    if resume_text:
        prompt = f"""
You are a friendly AI career mentor. Based on the resume below, answer the user's question politely and clearly.
Use the resume to personalize your advice.
Resume:
{resume_text}
User Question:
{question}
"""
    else:
        prompt = f"""
You are a helpful AI career mentor. The user didn't upload a resume.
Provide a clear, friendly, and helpful response to this general career question:
User Question:
{question}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error in Career Guidance Agent: {str(e)}"

# ✅ Final decision logic
def agent_decision(resume_file=None, question=""):
    resume_text = ""
    outputs = []

    if resume_file:
        resume_text = extract_resume_text(resume_file)

    q_lower = question.lower()

    # If only resume is uploaded with no question
    if resume_text and not question.strip():
        return agent_executor.run(f"Analyze the resume and give both strength and role. Text: {resume_text}")

    # Collect outputs from matching keywords
    if resume_text and "strength" in q_lower:
        outputs.append(strength_tool.run(resume_text))

    if resume_text and "role" in q_lower:
        outputs.append(role_tool.run(resume_text))

    if resume_text and "feedback" in q_lower:
        outputs.append(feedback_tool.run(resume_text))

    if question.strip() and (
        "strength" not in q_lower and "role" not in q_lower and "feedback" not in q_lower
    ):
        outputs.append(gpt_career_guidance(resume_text, question))

    # Join responses or show a fallback message
    if outputs:
        return "\n\n".join(outputs)
    elif question.strip():
        return gpt_career_guidance(resume_text, question)
    else:
        return "⚠️ Please upload a resume or ask a question."

# ✅ Clear button logic
def clear_fields():
    return None, "", ""

with gr.Blocks(title="PathForge Agent App 🧠") as demo:
    # ✅ Add Title (so it's visible like in your first screenshot)
    gr.Markdown("<h1 style='text-align: center;'>PathForge Agent App 🧠</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload your resume or ask a question. This smart agent will decide which tool to use!</p>")

    # ✅ How to Use Accordion
    with gr.Accordion("🛠️ How to Use This App", open=False):
        gr.Markdown("""
**🔍 Use this app in 3 simple ways:**
1. **Upload your resume** to get:
   - Resume strength (Weak / Average / Strong)
   - Suitable job role prediction
2. **Ask a question** (optional), such as:
   - "What’s my resume strength?"
   - "Can you give resume feedback?"
   - "What role suits my profile?"
   - "How to grow my career in AI?"
3. **Use both together** to get personalized guidance.
If you only ask a general career question without a resume, the app will still respond with advice!
""")

    # ✅ Input Section
    with gr.Row():
        resume = gr.File(label="📄 Upload Resume", type="filepath", file_types=[".pdf"])
        question = gr.Textbox(
            label="💬 Ask something (optional)",
            placeholder="Ask about resume, role, feedback, or career growth...",
            lines=3
        )

    # ✅ Submit and Clear side-by-side
    with gr.Row():
        submit = gr.Button("🚀 Submit")
        clear = gr.Button("🧹 Clear", variant="secondary")

    output = gr.Textbox(label="📤 Response", lines=12)

    # Button logic
    submit.click(fn=agent_decision, inputs=[resume, question], outputs=output)
    clear.click(fn=lambda: (None, "", ""), inputs=[], outputs=[resume, question, output])


demo.launch()

if __name__ == "__main__":
    demo.launch()
