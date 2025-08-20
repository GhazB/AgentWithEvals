import streamlit as st
from langchain_openai import ChatOpenAI
from app_chat import create_msg_history

def start_chat():
    st.title("Local Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message['role'] != 'system' and message['role'] != 'developer':
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    if user_input := st.chat_input("How can I help you - running locally? "):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        message_history = create_msg_history(st.session_state.messages)

        llm=ChatOpenAI(model="openai/gpt-oss-20b", api_key="XXX",base_url="http://localhost:8080/v1")
        #response = llm.invoke(message_history)
        #with st.chat_message("assistant"):
        #    st.markdown(response.content)
        full_response = ""
        placeholder = st.empty()
        for response in llm.stream(message_history):
            print(f"DEBUG: Response chunk: {response}")
            if response.content:
                full_response += response.content
                with placeholder.chat_message("assistant"):
                    st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    st.set_page_config(page_title="Helpful Chatbot", page_icon=":robot_face:")
    start_chat()
