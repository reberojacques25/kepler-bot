
import os
import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Kepler AI Chatbot", layout="wide", page_icon="🎓")

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
    font-family: 'Segoe UI', sans-serif;
}
.question-box {
    background-color: #0C2340;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 10px;
}
.answer-box {
    background-color: #2ECC71;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Kepler AI Chatbot Assistant")
st.markdown("Ask any question from the Q&A dataset.")

df = pd.read_csv("kepler_qa.csv")
docs = [Document(page_content=f"Q: {row['Questions']}\nA: {row['Answers']}") for _, row in df.iterrows()]

openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OpenAI API key in Streamlit secrets or environment variables.")
    st.stop()

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔍 Type your question here:")
if query:
    result = qa_chain({"question": query})
    answer = result["answer"]
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", answer))

for speaker, msg in st.session_state.chat_history[-10:]:
    box = 'question-box' if speaker == "You" else 'answer-box'
    icon = "🧑" if speaker == "You" else "🤖"
    st.markdown(f"<div class='{box}'><b>{icon} {speaker}:</b> {msg}</div>", unsafe_allow_html=True)
