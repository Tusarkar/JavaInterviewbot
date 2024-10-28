
import pandas as pd
import re

# Load the CSV file containing questions and answers
df = pd.read_csv('outputcsv.csv')  # Replace with your CSV file path
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Function to clean text
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
    else:
        text = ''  # Replace non-string entries (e.g., NaN) with an empty string
    return text

# Apply cleaning function to each question and answer
cleaned_questions = [preprocess_text(q) for q in questions]
cleaned_answers = [preprocess_text(a) for a in answers]
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('distilroberta-base')  # This model is fast and suitable for similarity tasks

# Encode each question into a vector
question_vectors = model.encode(cleaned_questions)
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance with API key and serverless specifications
pc = Pinecone(
    api_key='a517fc55-6b34-4cc2-8ac8-27eab0f9666b',
)

# Define index name and vector dimension
index_name = "javaqna"
dimension = 768  # 'all-MiniLM-L6-v2' model has a 384-dimensional output

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',  # Adjust the metric if needed
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index using Index() instead of index()
index = pc.Index(index_name)

# Upsert question vectors with metadata
for i, vector in enumerate(question_vectors):
    index.upsert([(str(i), vector.tolist(), {'question': cleaned_questions[i], 'answer': cleaned_answers[i]})])
import streamlit as st
def find_answer(user_question):
    cleaned_question = preprocess_text(user_question)
    user_vector = model.encode([cleaned_question])[0]
    
    # Use keyword arguments for the query method
    result = index.query(vector=user_vector.tolist(), top_k=1, include_metadata=True)
    
    # Check if there are matches before trying to access the answer
    if result['matches']:
        return result['matches'][0]['metadata']['answer']
    else:
        return "Sorry, I couldn't find an answer to that question."

# Function to format the answer for better readability
def format_answer(answer):
    # Capitalize the first letter of each sentence
    answer = re.sub(r'(^\w|\.\s+\w)', lambda x: x.group().upper(), answer)
    return answer

# Streamlit UI

st.title("Question Answer Bot")
user_question = st.text_input("Java interview questions:")

if user_question:
    answer = find_answer(user_question)
    formatted_answer = format_answer(answer)
    st.write(f"Answer: {formatted_answer}")
