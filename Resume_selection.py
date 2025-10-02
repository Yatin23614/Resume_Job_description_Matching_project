# train_resume_model.py

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st

# Text cleaning 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Loading Data
df = pd.read_csv("Resume.csv")

#Category to JD mapping
category_to_jd = {
    "HR": "We are seeking an HR specialist experienced in recruitment, employee relations, and HR policies.",
    "Data Science": "Looking for a data scientist skilled in Python, ML algorithms, and data analysis.",
    "Operations": "Operations manager needed to handle logistics, supply chain, and team coordination.",
    "Advocate": "Hiring an advocate experienced in legal drafting, court proceedings, and case analysis.",
    "IT": "IT professional required for managing software systems, network security, and technical support.",
    "Finance": "We need a finance expert to handle accounting, budgeting, and financial reporting.",
    "Teaching": "Teacher needed with strong subject knowledge, communication, and classroom management skills.",
    "Healthcare": "Hiring a healthcare specialist skilled in patient care, diagnostics, and health management.",
    "Sales": "Sales executive needed to manage client relations, meet targets, and grow business.",
    "Engineering": "Engineer required with strong technical background in design, analysis, and implementation.",
    "DESIGNER": "Creative designer needed with expertise in graphic design, tools like Photoshop, and branding.",
    "INFORMATION-TECHNOLOGY": "IT professional required for system administration, network security, and application support.",
    "TEACHER": "Looking for an educator skilled in subject teaching, classroom management, and student engagement.",
    "ADVOCATE": "Hiring an advocate experienced in legal research, litigation, and client representation.",
    "BUSINESS-DEVELOPMENT": "Business development executive needed to grow client base and identify new market opportunities.",
    "HEALTHCARE": "Healthcare worker needed with experience in patient care, medical procedures, and clinical assistance.",
    "FITNESS": "Fitness trainer required to create workout plans, guide clients, and promote physical health.",
    "AGRICULTURE": "Agriculture specialist needed with expertise in crop management, irrigation, and sustainable farming.",
    "BPO": "BPO associate required with good communication skills to handle customer service and technical support.",
    "ENGINEERING": "Engineer needed for design, testing, and implementation of technical projects.",
    "SALES": "Salesperson required to meet sales targets, maintain client relationships, and identify new opportunities.",
    "CONSULTANT": "Consultant needed to analyze business problems and deliver strategic recommendations.",
    "DIGITAL-MEDIA": "Digital media expert required for online content creation, SEO, and social media management.",
    "FINANCE": "Financial analyst needed to work on budgeting, reporting, and forecasting.",
    "ACCOUNTANT": "Accountant required to manage ledgers, tax filings, and financial statements.",
    "NETWORKING": "Networking engineer required to maintain LAN/WAN infrastructure and troubleshoot connectivity issues.",
    "CHEF": "Chef needed with experience in preparing a variety of cuisines and managing kitchen operations.",
    "APPAREL": "Apparel specialist required with experience in textile design, garment production, and fashion trends.",
    "ARTS": "Artist or art teacher needed for teaching creative skills, organizing exhibitions, and student engagement."
}

# Creating JD_str column
df['JD_str'] = df['Category'].map(category_to_jd)

# Cleaning text
df['Cleaned_Resume'] = df['Resume_str'].astype(str).apply(clean_text)
df['Cleaned_JD'] = df['JD_str'].astype(str).apply(clean_text)

# Concatenating JD and Resume for matching
df['Text_Combined'] = df['Cleaned_JD'] + " " + df['Cleaned_Resume']

#Label for binary classification
df['Label'] = df['Category'].apply(lambda x: 1 if x in category_to_jd else 0)

# Filter for multiple classes (1 and 0)
if df['Label'].nunique() < 2:
    raise ValueError("Need at least 2 classes in the data for training.")

# Split data
X = df['Text_Combined']
y = df['Label']

# TF-IDF and model training
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

with open("resume_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
#Streamlit App
import pickle
import streamlit as st
import pandas as pd

# Loading model and vectorizer
with open("resume_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("📄 Resume-JD Matching (CSV Batch Prediction)")

st.write("Upload a CSV file with a column of resumes. Select a job category to evaluate match.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df.head())

    text_columns = df.select_dtypes(include='object').columns.tolist()
    resume_column = st.selectbox("Select Resume Text Column", text_columns)

    if isinstance(category_to_jd, dict):
        category_options = sorted(category_to_jd.keys())
    else:
        st.error("Job category mapping is not a dictionary.")
        category_options = []

    selected_category = st.selectbox("Select Job Category (JD)", category_options, index=0)

    if st.button("Predict Suitability"):
        jd_text = clean_text(category_to_jd[selected_category])
        df["Cleaned_Resume"] = df[resume_column].astype(str).apply(clean_text)
        df["Combined_Text"] = jd_text + " " + df["Cleaned_Resume"]

        X_vec = vectorizer.transform(df["Combined_Text"])
        probs = model.predict_proba(X_vec)[:, 1]

        df["Confidence"] = probs

        # Get top 5 candidates based on confidence
        top_5 = df[[resume_column, "Confidence"]].sort_values(by="Confidence", ascending=False).head(5).reset_index(drop=True)

        st.success("Top 5 best matching resumes:")
        st.dataframe(top_5)

        # Optionally download full predictions
        df_full = df[[resume_column, "Confidence"]]
        csv = df_full.to_csv(index=False)
        st.download_button("📥 Download All Predictions as CSV", csv, file_name="resume_predictions.csv", mime="text/csv")