import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
import nltk
from dotenv import load_dotenv
load_dotenv()

# Download necessary NLTK data
nltk.download('punkt')
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize Pinecone
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your data
data = {
    "Health Condition": ["Cold & Flu", "Headache", "Insomnia", "Indigestion", "Stress"],
    "Symptoms": [
        "Runny nose, fever, sore throat",
        "Throbbing pain, sensitivity",
        "Difficulty sleeping, restlessness",
        "Bloating, stomach pain",
        "Anxiety, irritability"
    ],
    "Suggested Herbal Solution": [
        "Ginger tea with honey, turmeric milk",
        "Peppermint oil massage, chamomile tea",
        "Lavender tea, Ashwagandha",
        "Fennel seeds, ginger tea",
        "Lemon balm tea, Tulsi (holy basil) tea"
    ],
    "Preparation Instructions": [
        "Boil fresh ginger slices, add honey; Mix turmeric in warm milk",
        "Massage temples with diluted peppermint oil; Brew chamomile tea",
        "Brew lavender flowers; Take Ashwagandha capsules",
        "Chew fennel seeds; Prepare ginger tea",
        "Brew lemon balm; Make Tulsi tea"
    ]
}

# Convert data into chunks
data_str = str(data)
sections = data_str.split('\n\n')
chunks = [section.strip() for section in sections if section.strip()]

# Generate embeddings for chunks
embeddings = model.encode(chunks)
PINE_API_KEY=os.getenv("PINECONE_API_KEY")
pc = Pinecone(
        api_key=PINE_API_KEY
    )

# Initialize Pinecone index
if 'chat' not in pc.list_indexes().names():
    pc.create_index(
        name='chat',
        dimension=384,  # Model dimension
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index('chat')

# Upsert data
for i, emb in enumerate(embeddings):
    index.upsert([(f"chunk{i}", emb, {"text": chunks[i]})])

# Load GPT model
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Streamlit app
st.title("Personalized Assistant")

# User input
query = st.text_input("Ask a question:", "")

if st.button("Get Answer"):
    if query:
        # Generate query embedding
        query_embedding = model.encode(query).tolist()

        # Query Pinecone
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Extract relevant context
        relevant_context = " ".join([match['metadata']['text'] for match in results['matches']])

        # Prepare prompt
        prompt = f"Context: {relevant_context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        response = generator(prompt, max_new_tokens=150, do_sample=True, truncation=True)

        # Display response
        st.text_area("Answer:", response[0]['generated_text'], height=200)
    else:
        st.warning("Please enter a question.")
