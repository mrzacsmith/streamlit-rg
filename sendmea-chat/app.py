import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from constants import *

# Load environment variables
load_dotenv()

# Retrieve API keys
openai_api_key = os.getenv("OPENAI_API_KEY2")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI embeddings and Pinecone Vector Store
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings,
                                           pinecone_api_key=pinecone_api_key)


def get_response(user_input, temp):
    if user_input:
        try:
            # Retrieve context from the vector store based on the input
            retriever = document_vectorstore.as_retriever()
            context = retriever.get_relevant_documents(user_input)

            # Log the context for debugging
            print("Retrieved context:", context)

            # Create prompt with context
            prompt = f"{user_input} Context: {context}"
            llm = ChatOpenAI(api_key=openai_api_key, temperature=temp)
            results = llm.invoke(prompt)
            # Append temperature to the response
            return f"{results.content} ({temp})"
        except Exception as e:
            print(f"Error: {e}")
            return f"Sorry, I couldn't process your request due to: {e}"
    return 'i love you'


# Streamlit app setup
st.set_page_config(page_title="RAG HTML App", page_icon=":books:", layout="wide")
st.title('Sendmea Blog Chat!')

# Sidebar and user input
with st.sidebar:
    st.markdown("## ❤️ AKM")
    # Use session state to maintain temperature changes without affecting chat history
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.5
    st.session_state.temperature = st.slider("Choose the response temperature", min_value=0.0, max_value=1.0,
                                             value=st.session_state.temperature, step=0.05)

# Manage chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input('Enter your question here...')
if user_query:
    # Append human message immediately
    st.session_state.chat_history.append({"type": "Human", "content": user_query})

    # Display chat messages as they come
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if message["type"] == "AI" else "Human"):
            st.write(message["content"])

    # Get AI response, append temperature, and display
    response = get_response(user_query, st.session_state.temperature)
    st.session_state.chat_history.append({"type": "AI", "content": response})
    with st.chat_message("AI"):
        st.write(response)
