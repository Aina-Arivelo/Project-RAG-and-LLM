import os
os.environ["HF_HOME"] = "/goinfre/sarakoto/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/goinfre/sarakoto/.cache/huggingface/hub"

import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`")

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory        
from htmlTemplate import css, bot_template, user_template
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_txt_chunks(raw_txt):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_txt)


def get_vectorStore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorStore):
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ✅ léger ~600MB
        max_new_tokens=512,
        temperature=0.5,
        do_sample=True,
        return_full_text=False   # ✅ retourne seulement la réponse, pas le prompt
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload Your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_txt = get_pdf_text(pdf_docs)
                txt_chunks = get_txt_chunks(raw_txt)
                vectorStore = get_vectorStore(txt_chunks)
                st.session_state.conversation = get_conversation_chain(vectorStore)


if __name__ == '__main__':
    main()