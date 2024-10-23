import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefine the path to the PDF you want to process
PDF_PATH = "CC.pdf"  # Specify your backend PDF file here

# Function to extract text from the predefined PDF file
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks for easier processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks using FAISS and embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up a conversational chain with a custom prompt
def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, "Sorry but I cant fetch the Information from the database" and don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# To track the last query time
last_query_time = 0
query_cooldown = 5  # seconds

@st.cache_data
def cached_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )
    
    return response["output_text"]

# Function to handle user input and fetch the response
def user_input(user_question):
    global last_query_time
    current_time = time.time()

    if current_time - last_query_time < query_cooldown:
        return "Please wait before asking another question."

    last_query_time = current_time
    return cached_user_input(user_question)

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Team ALPHA", layout="wide")
    st.header("VCT Hackathon Esports Manager Challenge - LLM Powered Chatbot 🤖 ")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Sidebar for additional information and a horizontal slider
    with st.sidebar:
        st.subheader("🚀 Project Submission for Hackathon: LLM-Powered Digital Assistant for VALORANT Esports")
        
        # Team Name and Members
        st.write("👥 **Team Name:** Team Alpha")
        st.write("**Team Members:**")
        st.write("- [Pratik Khose](https://github.com/Pratik-Khose)")
        st.write("- [Hariom Ingle](https://github.com/Hariom-Ingle)")  # Replace with actual links
        st.write("- [Chetan Rathod](https://github.com/ChetanRRathod)")  # Replace with actual links
        
        # Project Description
        st.write("We built an AI-powered digital assistant leveraging **Amazon Bedrock’s** capabilities to assist a VALORANT esports team in scouting and recruitment. The assistant is designed to streamline the process of team composition, role assignments, and provide detailed analysis based on the given data about players, roles, and team synergy.")
        
        st.write("### ⚡ Key Features:")
        st.write("- ✅ **Team Synergy Evaluation:** It analyzes player combinations to justify team effectiveness.")
        st.write("- 🎯 **Role Assignment:** Based on specific prompt, it can suggest ideal role for each player, ensuring balanced team composition.")
        st.write("- 💬 **Interactive Chat Interface:** A user-friendly chat interface, powered by **Amazon Bedrock**, allows seamless interaction.")

        # Links to GitHub Repo and Demo Video
        st.write("### 🔗 Links:")
        st.write("- [GitHub Submission Repo](https://github.com/Pratik-Khose/VCT-Hackathon-Esports-Manager-Challenge)")  # Replace with actual GitHub link
        st.write("- [Submission Video](https://youtu.be/demo-video)")  # Replace with actual YouTube link
        
        st.write("### 🙏 Acknowledgements:")
        st.write("We would like to extend our sincere thanks to the **VCT team** and the hackathon organizers for providing us with the opportunity to participate in this exciting challenge. Special thanks to **Amazon Web Services (AWS)** for granting us $100 in free credits 💸, which allowed us to fine-tune the large language models (LLMs) on **Amazon Bedrock** and leverage the full potential of their services in building our AI-powered assistant. Your support made it possible for us to innovate and create this 🎉.")


    # Process the predefined PDF when the app starts
    with st.spinner("Establishing Database Connection..."):
        raw_text = get_pdf_text(PDF_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database searched and configured Successfully!")

    # Display the conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input textbox at the bottom of the page
    if user_question := st.chat_input("Ask a question "):
        # Append user message to the chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Generate the response from the PDF
        with st.chat_message("assistant"):
            st.markdown("Thinking...")
            response = user_input(user_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

if __name__ == "__main__":
    main()  