import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# OpenAI API Key 설정
OPENAI_API_KEY = 'sk-'
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 텍스트 파일 로드 함수
def load_text(uploaded_file):
    return uploaded_file.read().decode('utf-8')

# PDF에서 텍스트 추출
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 텍스트를 청크 단위로 분할 및 벡터 스토어 생성
def process_documents(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text("\n".join(documents))
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

# 대화 체인 생성
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, window_size=10)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Streamlit UI 설정
st.title("문서 Q&A 시스템")

# 파일 업로드 섹션
st.header("파일 업로드")
uploaded_files = st.file_uploader("텍스트 또는 PDF 파일을 업로드하세요", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Upload", key="upload"):
        with st.spinner("처리중.."):
            documents = []
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "text/plain":
                    documents.append(load_text(uploaded_file))
                elif uploaded_file.type == "application/pdf":
                    documents.append(get_pdf_text([uploaded_file]))

            embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
            vectorstore = process_documents(documents, embeddings)

            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.chat_history = []
            st.success("파일 처리가 완료되었습니다.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("대화형 챗봇")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("질문을 입력해주세요~"):
    if 'conversation' in st.session_state:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        try:
            result = st.session_state.conversation({
                "question": user_query,
                "chat_history": st.session_state.chat_history
            })
            response = result["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            response = f"오류가 발생했습니다: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.chat_message("assistant"):
            st.markdown("대화 체인이 아직 생성되지 않았습니다. 파일을 업로드하고 'Upload' 버튼을 클릭해주세요.")