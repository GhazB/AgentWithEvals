import streamlit as st
import os
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from graph import ChatbotAgent

import random

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="ArrowStreet"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

def create_msg_history(messages):
    message_history = [] 
    for message in st.session_state.messages:
        if message['role'] == 'user':
            message_history.append(HumanMessage(message['content']))
        elif message['role'] == 'assistant':
            message_history.append(AIMessage(message['content']))
        elif message['role'] == 'system':
            message_history.append(SystemMessage(message['content']))
    return message_history


def start_chat():
    st.title("Helpful Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread-id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 9999)
    thread_id = st.session_state.thread_id
    
    for message in st.session_state.messages:
        if message['role'] != 'system' and message['role'] != 'developer':
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    if user_input := st.chat_input("How can I help you? "):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        message_history = create_msg_history(st.session_state.messages)

        agent = ChatbotAgent(os.environ['OPENAI_API_KEY'])
        thread = {"configurable":{"thread_id": thread_id}}
        parameters = {"messages": message_history}
        full_response = ""
        for s in agent.graph.stream(parameters,thread):
            for k,v in s.items():
                print(f"DEBUG {k=}: {v=}")
                if response := v.get("response"):
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        for r in response:
                            full_response += r.content
                            placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
            print("**********")






if __name__ == '__main__':
    st.set_page_config(page_title="Helpful Chatbot", page_icon=":robot_face:")
    start_chat()