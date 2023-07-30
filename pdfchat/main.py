import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import  css,bot_template,user_template
from langchain.llms import HuggingFaceHub

headers = {
    "authorization": st.secerts['OPENAI_API_KEY'],
}

# PDF into a string variable
def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# String broken into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Chunks -> OPENAI(embedding) -> vectorstore
def get_vectorstore(text_chuncks):
    embeddings = OpenAIEmbeddings()
    # local embeddings
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") 
    vectorstore = FAISS.from_texts(texts=text_chuncks, embedding=embeddings)
    return vectorstore

# Select llm, create converstaional memory
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Conversation dynamics
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}",message.content), unsafe_allow_html=True)
        else:

            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    # intialize session_state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with PDFs :books:")

    # create chat interface
    user_question = st.chat_input('Ask a question')
    if user_question:
        handle_userinput(user_question)

    # Where PDFs are uploaded
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here and click Analyze", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner('Processing'):
                # create the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # create the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation object
                st.session_state.conversation = get_conversation_chain(vectorstore)

               

if __name__ == '__main__':
    main()
