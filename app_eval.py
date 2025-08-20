import streamlit as st
import os
from graph import ChatbotAgent
from app_chat import create_msg_history

import random
import pandas as pd
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["OPENAI_MODEL"] = st.secrets['OPENAI_MODEL']
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]="ArrowStreet"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"

EVAL_PROMPT = """
You are a helpful assistant that evaluates the performance of an agent based on its responses.
Given a query, an expected response, and an actual response from the agent, determine if the agent's response is correct.
You will receive a list of queries, an expected response, and the actual response from the agent.
Your task is to evaluate the correctness of the agent's response in terms of both category and content.
You will return a structured response with the following fields:
- matches: A boolean indicating if the agent's response matches the expected response.
- reasoning: A string explaining your reasoning for the evaluation.
"""

def get_evals_golden_data():
    """Get the golden data for evals."""
    examples=[
        {
          "input":[
              { "question": "Hi"}, 
          ],
          "output":{
             "category":"smalltalk_agent",
             "response": "I am good. How are you doing?"
          }
        },
        {
          "input":[
              { "question": "Hi"}, 
              { "answer": "How are you?"},
              { "question": "I have a problem"},
          ],
          "output":{
             "category":"complaint_agent",
             "response": "I am sorry to hear. Can you please tell me more about the issue so I can help you better?"
          }
        },        
    ]
    return examples

def convertListToMessages(message_list):
    output_list=[]
    print(f"DEBUG: Converting message_list: {message_list=}")
    for message in message_list:
      if message.get("question"):
        output_list.append(HumanMessage(content=message["question"]))
      elif message.get("answer"):
        output_list.append(AIMessage(content=message["answer"]))
      else:
        print("ERROR. Unexpected dictionary key for {message=}")
    return output_list

def convert_inputlist_to_string(input_list):
    formatted_list=[]
    for item in input_list:
        for k,v in item.items():
            if k == "question":
                formatted_list.append(f"Q: {v}")
            elif k == "answer":
                formatted_list.append(f"A: {v}")
            else:
                print(f"ERROR: Unexpected key {k} in input item {item}")
    rlist= "  \n\n".join(formatted_list)
    return rlist

def run_graph(conversation):
   print(f"**** DEBUG: Running graph with conversation: {conversation}")
   agent= ChatbotAgent(os.environ['OPENAI_API_KEY'])
   thread = {"configurable":{"thread_id": random.randint(1000, 9999)}}
   parameters = {"messages": conversation}
   actual_response = agent.graph.invoke(parameters, thread)
   print(f"---- DEBUG: Actual response from graph: {actual_response}")
   return actual_response

class EvalMatch(BaseModel):
    matches: bool
    reasoning: str

def llm_judge_compare(query_list,expected_response,actual_response):
    llm = ChatOpenAI(model=os.environ['OPENAI_MODEL'], api_key=os.environ['OPENAI_API_KEY'])
    messages = [SystemMessage(content=EVAL_PROMPT)]
    m = HumanMessage(content=f"QUERY: {query_list}\nEXPECTED RESPONSE: {expected_response}\nACTUAL RESPONSE: {actual_response}")
    messages.append(m)
    resp=llm.with_structured_output(EvalMatch).invoke(messages)
    print(f"LLM Judge compare. EVAL Response: {resp}\n\n")
    return resp.matches, resp.reasoning

def final_answer_correct(query, expected_category,expected_response,actual_category,actual_response):
    #print(f" FINAL-ANSWER-CORRECT: {expected_category=}, {expected_response=}, {actual_category=},{actual_response=}")
    category_answer_correct = expected_category == actual_category
    response_answer_correct, reason = llm_judge_compare(query, expected_response,actual_response)
    #print(f" FINAL-ANSWER-CORRECT: {category_answer_correct=}, {response_answer_correct=}")
    return category_answer_correct, response_answer_correct, reason

def main_run():
    examples = get_evals_golden_data()
    results=[]
    for example in examples:
#      try:
        inputs=example["input"]
        llm_input=convertListToMessages(inputs)
        expected_response = example["output"]["response"]
        expected_category = example["output"]["category"]
        resp = run_graph(llm_input)
        actual_category = resp['category']
        actual_response_generator = resp['response']
        actual_response=""
        for step in actual_response_generator:
            print(f"DEBUG: Step in response: {step}")
            actual_response += step.content
        cat_result,resp_result, reason = final_answer_correct(inputs,expected_category,expected_response,actual_category,actual_response)
 
        inputstr="  \n\n".join([f"{i+1}. {msg['question'] if 'question' in msg else msg['answer']}" for i, msg in enumerate(inputs)])
        expected = example["output"]
        response_content={'category': actual_category,'response': actual_response}

        result_dict = {

            "input": convert_inputlist_to_string(inputs),
            "expected": expected,
            "actual": response_content,
            "cat_result": cat_result,
            "resp_result": resp_result,
            "resp_reason": reason,
            "input_list": inputs,  # Store the original input list for reference
        }
        results.append(result_dict)
#      except Exception as e:
#        print(f"Error processing example: {str(e)}")
#        continue
    df=pd.DataFrame(results)
    st.write("### Evaluation Results")
    st.dataframe(df, use_container_width=True)
   

if __name__ == '__main__':
    st.set_page_config(page_title="Agent Evaluation", page_icon=":robot:")
    st.title("Agent Evaluation")
    st.write("This app evaluates the agent's performance based on predefined examples.")
    
    if st.button("Run Evaluation"):
        main_run()

 