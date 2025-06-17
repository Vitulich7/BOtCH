import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime
import os

# Настройка логирования
LOG_FILE = "chat_logs.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    encoding='utf-8'
)

# Список моделей DialoGPT
model_names = {
    "DialoGPT small": "microsoft/DialoGPT-small",
    "DialoGPT medium": "microsoft/DialoGPT-medium",
    "DialoGPT large": "microsoft/DialoGPT-large",
}

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_response(tokenizer, model, chat_history_ids, user_input):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_history_ids, response

# Заголовок приложения
st.title("BOtCH — чат-бот на DialoGPT")

# Выбор модели
model_choice = st.selectbox("Выберите модель DialoGPT", list(model_names.keys()))
tokenizer, model = load_model(model_names[model_choice])

# Инициализация состояния сессии
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ввод сообщения пользователем
user_input = st.text_input("Введите сообщение", "")

# Кнопки управления
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Отправить") and user_input.strip():
        st.session_state.chat_history, bot_response = generate_response(tokenizer, model, st.session_state.chat_history, user_input)
        st.session_state.messages.append(("Пользователь", user_input))
        st.session_state.messages.append(("БОТ", bot_response))
        
        # Логирование
        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {bot_response}")
with col2:
    if st.button("Очистить историю"):
        st.session_state.chat_history = None
        st.session_state.messages = []

# Отображение истории сообщений
for speaker, msg in st.session_state.messages:
    if speaker == "Пользователь":
        st.markdown(f"**{speaker}:** {msg}")
    else:
        st.markdown(f"<div style='color: green'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)

streamlit run app.py
