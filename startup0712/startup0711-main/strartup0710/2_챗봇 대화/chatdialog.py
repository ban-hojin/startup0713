import openai
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import streamlit as st

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
st.write("챗봇과 대화를 시작하세요. 종료하려면 **'종료'**라고 입력하세요.")

# 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'terminated' not in st.session_state:
    st.session_state.terminated = False

# 사용자 입력 폼
def submit():
    if st.session_state.user_input:
        if st.session_state.user_input.lower() == "종료":
            st.session_state.terminated = True
            st.session_state.chat_history.append("**🤖 챗봇:** 챗봇을 종료합니다. 👋")
        else:
            response = chatbot_response(st.session_state.user_input)
            # 대화 기록에 추가
            st.session_state.chat_history.append(f"**👤 사용자:** {st.session_state.user_input}")
            st.session_state.chat_history.append(f"**🤖 챗봇:** {response}")
        st.session_state.user_input = ""  # 입력 필드 초기화

if not st.session_state.terminated:
    st.text_input("사용자 입력:", key="user_input", on_change=submit)

# 대화 기록 표시
for chat in st.session_state.chat_history:
    st.write(chat)