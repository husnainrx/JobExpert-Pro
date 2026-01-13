import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import docx
import PyPDF2
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATASET_PATH = "Dataset/resume_dataset.csv"
EVAL_FOLDER = "Model_Evaluation"

# -------------------- TEXT EXTRACTION -------------------- #
def extract_text(file):
    extension = file.name.split(".")[-1].lower()
    if extension == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    elif extension == "docx":
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    elif extension == "txt":
        return file.read().decode("utf-8", errors="ignore")
    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------- MODEL TRAINING -------------------- #
@st.cache_resource
def train_model():
    data = pd.read_csv(DATASET_PATH)
    data["cleaned"] = data["Resume"].apply(clean_text)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data["Category"])

    X_train, X_test, y_train, y_test = train_test_split(
        data["cleaned"], y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression + Calibrated probabilities
    base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train_vec, y_train)

    # ---------------- SAVE MODEL EVALUATION ---------------- #
    if not os.path.exists(EVAL_FOLDER):
        os.makedirs(EVAL_FOLDER)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(EVAL_FOLDER, "accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\n")
    with open(os.path.join(EVAL_FOLDER, "classification_report.txt"), "w") as f:
        f.write(report)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_FOLDER, "confusion_matrix.png"))
    plt.close()

    return model, vectorizer, encoder

# -------------------- UI STYLING -------------------- #
def apply_custom_style():
    st.markdown("""
        <style>
        .stApp {background-color: #0f1117;}
        header, footer {visibility: hidden;}
        h1, h2, h3, p, label {color: #e6e6e6; font-family: Inter, system-ui, sans-serif;}
        .block-container {padding-top: 2rem;}
        div[data-testid="metric-container"] {background-color: #1a1c23; border-radius: 12px; padding: 20px; border: 1px solid #2a2d3a;}
        .stButton>button {background-color: #2563eb; color: white; border-radius: 10px; height: 48px; font-size: 16px; border: none;}
        .stButton>button:hover {background-color: #1e40af;}
        textarea, .stFileUploader {background-color: #1a1c23 !important; border-radius: 10px !important; color: white !important;}
        </style>
    """, unsafe_allow_html=True)

# -------------------- MAIN APPLICATION -------------------- #
def main():
    st.set_page_config(page_title="JobExpert Pro", layout="wide")
    apply_custom_style()

    st.title("JobExpert Pro")
    st.write("AI-powered resume screening and job role classification using Logistic Regression.")

    if not os.path.exists(DATASET_PATH):
        st.error("Dataset not found. Please verify the dataset path.")
        return

    model, vectorizer, encoder = train_model()
    st.info(f"Model training complete. Evaluation saved in '{EVAL_FOLDER}' folder.")

    st.divider()
    tab_upload, tab_paste = st.tabs(["Upload Resume", "Paste Resume Text"])
    uploaded_text = ""
    manual_text = ""

    with tab_upload:
        file = st.file_uploader("Upload resume file", type=["pdf","docx","txt"])
        if file:
            uploaded_text = extract_text(file)

    with tab_paste:
        manual_text = st.text_area("Paste resume content", height=260)

    resume_text = manual_text if manual_text.strip() else uploaded_text
    st.divider()

    if st.button("Analyze Resume"):
        if not resume_text.strip():
            st.warning("Please upload a resume file or paste resume text.")
            return

        cleaned = clean_text(resume_text)
        vec = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(vec)[0]
        sorted_indices = np.argsort(probabilities)[::-1]

        # ---------------- TOP-N ROLES ---------------- #
        results = []
        for idx in sorted_indices:
            score = probabilities[idx]*100
            if len(results) == 0:
                results.append((encoder.classes_[idx], score))
            elif score >= 35 and len(results) < 3:
                results.append((encoder.classes_[idx], score))
            else:
                break

        primary_role, primary_conf = results[0]
        primary_conf = min(primary_conf, 99.0)  # cap display only

        # ---------------- DISPLAY METRICS ---------------- #
        st.subheader("Screening Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Inferred Job Role", primary_role)
        with col2:
            st.metric("Confidence Score", f"{primary_conf:.2f}%")
            st.progress(primary_conf/100)

        if len(results) > 1:
            st.subheader("Other Relevant Roles")
            for role, score in results[1:]:
                st.write(f"{role}: {score:.2f}%")

        if primary_conf < 70:
            st.info("Low confidence prediction. Manual review recommended.")

if __name__ == "__main__":
    main()
