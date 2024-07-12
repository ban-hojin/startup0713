import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = "sk-"  # 실제 API 키 입력

# PDF에서 텍스트 추출
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 텍스트를 청크 단위로 분할
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],  # 문자열 이스케이프 문제 수정
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 텍스트 청크를 벡터 스토어로 변환
def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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
st.title("PDF 문서 Q&A")
user_uploads = st.file_uploader("파일을 업로드해주세요~", accept_multiple_files=True)
if user_uploads:
    if st.button("Upload"):
        with st.spinner("처리중.."):
            raw_text = get_pdf_text(user_uploads)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.chat_history = []
            st.success("문서 처리가 완료되었습니다.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 기존 대화 내역 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("질문을 입력해주세요~"):
    if 'conversation' in st.session_state:
        # 사용자 질문을 대화 기록에 추가
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # 대화 체인 실행
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
            st.markdown("대화 체인이 아직 생성되지 않았습니다. 문서를 업로드하고 'Upload' 버튼을 클릭해주세요.")
