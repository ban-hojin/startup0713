import openai
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import streamlit as st
from datetime import datetime
import io

# OpenAI API 키 설정
openai.api_key = 'sk-'

# OpenAI 모델 초기화
llm = OpenAI(api_key=openai.api_key)

# 대화 체인 초기화
conversation = ConversationChain(llm=llm)

# 챗봇 응답 함수
def chatbot_response(user_input):
    try:
        response = conversation.run(input=user_input)
    except Exception as e:
        response = f"오류 발생: {e}"
    return response

# Streamlit 앱 구성
st.title("💬 안녕하세요! 챗봇입니다.")
col1, col2 = st.columns([4, 1])
with col1:
    st.write("챗봇과 대화를 시작하세요.")
with col2:
    if st.button("종료"):
        st.session_state.terminated = True
        st.session_state.chat_history.append("**🤖 챗봇:** 챗봇을 종료합니다. 👋")

# 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'terminated' not in st.session_state:
    st.session_state.terminated = False

# 사용자 입력 폼 함수
def submit():
    if st.session_state.user_input:
        response = chatbot_response(st.session_state.user_input)
        timestamp = datetime.now().strftime('%H:%M')
        # 대화 기록에 추가
        st.session_state.chat_history.append(f"**👤 사용자:** {st.session_state.user_input} <span style='font-size: 50%;'>({timestamp})</span>")
        st.session_state.chat_history.append(f"**🤖 챗봇:** {response} <span style='font-size: 50%;'>({timestamp})</span>")
        st.session_state.user_input = ""  # 입력 필드 초기화

# 대화 기록 표시
st.write("### 대화 기록")
for chat in st.session_state.chat_history:
    st.markdown(chat, unsafe_allow_html=True)

# 사용자 입력 폼
if not st.session_state.terminated:
    st.text_input("사용자 입력:", key="user_input", on_change=submit)
else:
    if st.button("다시 시작하기"):
        st.session_state.chat_history = []
        st.session_state.terminated = False

# 대화 기록 다운로드 기능
def download_chat_history():
    chat_history_str = "\n".join(st.session_state.chat_history)
    buffer = io.BytesIO(chat_history_str.encode())
    return buffer

if st.session_state.chat_history:
    st.download_button(
        label="대화 기록 다운로드",
        data=download_chat_history(),
        file_name="chat_history.txt",
        mime="text/plain"
    )
