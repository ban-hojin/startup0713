import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# OpenAI API key 설정
os.environ["OPENAI_API_KEY"] = "sk-"  # 여기에 실제 OpenAI API 키를 입력하세요.

# Streamlit 페이지 설정
st.set_page_config(page_title="🦜🔗 뭐든지 질문하세요~ ")
st.title('🦜🔗 뭐든지 질문하세요~ ')

def generate_response(input_text):  # LLM이 답변 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 0으로 설정
        model_name='gpt-3.5-turbo',  # 모델명
    )
    messages = [HumanMessage(content=input_text)]  # 메시지 생성
    response = llm(messages)  # 모델 호출
    return response.content  # 응답 반환

# Streamlit 폼 사용
with st.form('Question'):
    text = st.text_area('질문 입력:', '')  # 첫 페이지가 실행될 때 보여줄 질문
    submitted = st.form_submit_button('보내기')
    if submitted:
        response = generate_response(text)  # 폼 제출 시 응답 생성
        st.info(response)  # 응답 출력
