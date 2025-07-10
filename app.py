import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# ----------- Sample Training Data (Small Demo Model) ----------- #

data = {
    "Resume": [
        "Python developer skilled in Flask, REST API, SQL.",
        "Frontend developer with ReactJS, HTML, CSS.",
        "Data scientist with Python, machine learning, Pandas.",
        "Java backend engineer with Spring Boot, Microservices.",
        "AI/ML enthusiast with Python, TensorFlow, deep learning.",
        "ReactJS UI/UX engineer.",
        "Java backend developer with Spring.",
        "Graphic designer skilled in Photoshop.",
        "PHP Laravel web developer.",
        "Content writer with documentation skills."
    ],
    "Job_Description": [
        "Hiring Python backend developer with Flask and REST APIs.",
        "Looking for ReactJS frontend developer with UX skills.",
        "We need a data scientist with Python and ML knowledge.",
        "Opening for Java microservices engineer with Spring Boot.",
        "Hiring AI/ML engineer with TensorFlow and deep learning.",
        "Hiring Python backend engineer.",
        "Looking for data scientist.",
        "Hiring DevOps engineer.",
        "Need data analyst with Power BI.",
        "Looking for Python software developer."
    ],
    "Match": [1,1,1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

# ----------- Text Cleaning Function ----------- #

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['Clean_Resume'] = df['Resume'].apply(clean_text)
df['Clean_Job'] = df['Job_Description'].apply(clean_text)
df['Combined'] = df['Clean_Resume'] + ' ' + df['Clean_Job']

# ----------- Feature Extraction ----------- #

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Combined'])
y = df['Match']

# ----------- Model Training ----------- #

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------- Streamlit App UI ----------- #

st.set_page_config(page_title="AI JobFit: Resume & Job Match Predictor", page_icon="🤖")
st.title("🤖 AI JobFit: Resume & Job Match Predictor")
st.markdown("Paste your **Resume** and **Job Description** below to check suitability.")

resume_input = st.text_area("📄 Paste Resume Here", height=200)
job_input = st.text_area("📝 Paste Job Description Here", height=200)

if st.button("🚀 Check Match"):
    if resume_input.strip() == "" or job_input.strip() == "":
        st.warning("⚠ Please paste both Resume and Job Description.")
    else:
        # Clean input
        clean_resume = clean_text(resume_input)
        clean_job = clean_text(job_input)
        combined_input = clean_resume + " " + clean_job
        input_vector = vectorizer.transform([combined_input])

        # Prediction
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][1] * 100

        if prediction == 1:
            st.success(f"✅ Suitable Match! Confidence: {probability:.2f}%")
        else:
            st.error(f"❌ Not Suitable Match. Confidence: {probability:.2f}%")
