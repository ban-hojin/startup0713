import streamlit as st
import openai

# OpenAI API Key 설정
openai.api_key = "sk-"

# 사용자가 선택할 수 있는 언어 목록
languages = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "Vietnamese": "vi"
}

def translate_text(text, target_language_code):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Translate the following text to {target_language_code}: {text}. Only provide the translated text without any additional information."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    translated_text = response.choices[0].message['content'].strip()
    
    return translated_text

# Streamlit 애플리케이션 구성
st.markdown("<h1 style='text-align: center;'>Multilingual Translator</h1>", unsafe_allow_html=True)

# 컬럼을 사용하여 균형감 있게 배치
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # 언어 선택을 위한 라디오 버튼
    target_language = st.radio("Select the target language:", list(languages.keys()), key="center_radio")

# 대화 기록을 위한 리스트
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Form을 사용하여 입력과 버튼을 한 번에 처리
with col2:
    with st.form(key="translate_form", clear_on_submit=True):
        source_text = st.text_area("Enter the text to translate:", key="input_text")
        submit_button = st.form_submit_button(label="Translate")

    # 대화 초기화 버튼
    st.markdown("""
        <style>
        .small-button button {
            width: 50%;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Reset", key="reset_button"):
        st.session_state.conversation = []

if submit_button:
    if source_text:
        target_language_code = languages[target_language]
        translated_text = translate_text(source_text, target_language_code)
        
        # 대화 기록 업데이트
        st.session_state.conversation.append(f"**User**: {source_text}")
        st.session_state.conversation.append(translated_text)

# 대화 기록 표시
with col2:
    for message in st.session_state.conversation:
        if message.startswith("**User**:"):
            st.markdown(message)
        else:
            # 번역된 텍스트를 2배 크기로 표시
            st.markdown(f"<div style='font-size: 2em;'>{message}</div>", unsafe_allow_html=True)
