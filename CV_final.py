# Importing required libraries
import streamlit as st
import pdfplumber
import docx2txt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

# Function to extract user-defined skills from resume text
def extract_skills(text, user_skills):
    text = text.lower()
    extracted = [skill.strip().lower() for skill in user_skills if skill.strip().lower() in text]
    return list(set(extracted))  # remove duplicates

# Function to estimate years of experience from dates mentioned
def extract_experience(text):
    doc = nlp(text)
    years = []
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            try:
                if 'year' in ent.text.lower():
                    num = int(ent.text.split()[0])
                    years.append(num)
            except:
                continue
    return max(years, default=0)

# Function to compute a similarity score between resume and job description
def match_score(resume_text, job_description):
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(score[0][0]) * 100, 2)

# -------- Streamlit Frontend Starts Here -------- #

st.title("🔍 AI Resume Screening App")

# Text area for job description
job_description = st.text_area("📄 Paste the Job Description Below:", height=200)

# Text input for skills (comma-separated)
skills_input = st.text_input("🛠️ Enter Required Skills (comma-separated):", placeholder="e.g., Python, SQL, Machine Learning")

# File uploader for multiple resumes
uploaded_files = st.file_uploader("📂 Upload Resume Files (PDF/DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)

# Main logic to process resumes
if uploaded_files and job_description and skills_input:
    
    # Parse user-entered skills
    user_skills = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
    
    if not user_skills:
        st.warning("⚠️ Please enter at least one skill.")
    else:
        st.markdown("### 🔎 Screening Results")

        for resume in uploaded_files:
            # Extract text
            if resume.name.endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume)
            elif resume.name.endswith('.docx'):
                resume_text = extract_text_from_docx(resume)
            else:
                st.warning(f"Unsupported file type: {resume.name}")
                continue

            # Extract information
            skills = extract_skills(resume_text, user_skills)
            experience = extract_experience(resume_text)
            score = match_score(resume_text, job_description)

            # Display results
            st.subheader(f"👤 Candidate: {resume.name}")
            st.write(f"✅ **Skills Matched**: {', '.join(skills) if skills else 'None'}")
            st.write(f"🧠 **Estimated Experience**: {experience} year(s)")
            st.write(f"📊 **Match Score**: {score}%")
            st.markdown("---")
