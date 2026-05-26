import streamlit as st
from transformers import pipeline
import logging

# Hide transformer warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("🤖 My AI Chatbot")


@st.cache_resource
def load_model():

    chatbot = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

    return chatbot


chatbot = load_model()


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input
user_input = st.chat_input("Type your message")


if user_input:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Create proper prompt
    prompt = "You are a helpful AI assistant.\n\n"

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"

        else:
            prompt += f"Assistant: {msg['content']}\n"

    prompt += "Assistant:"

    # Generate response
    with st.spinner("Thinking..."):

        response = chatbot(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            truncation=True,
            clean_up_tokenization_spaces=False
        )

        full_output = response[0]["generated_text"]

        bot_reply = full_output.split("Assistant:")[-1].strip()

    # Save assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    # Display assistant reply
    with st.chat_message("assistant"):
        st.write(bot_reply)