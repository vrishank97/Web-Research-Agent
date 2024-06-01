import os
import json
import datetime
import time
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Set page configuration
st.set_page_config(layout="wide", page_title="Local Web Research Agent", page_icon="🤖")
st.sidebar.title("Web Research Agent 🤖📝")

# Defining LLM
model_choices = ['llama3', 'llama-3-8b-ultra-instruct', 'phi3:14b-medium-128k-instruct-q3_K_M']  # Add your model choices here
local_llm = st.sidebar.selectbox("Select Model", model_choices) ## Model object created later

# Web Search Tool
wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

# System Prompts
default_generate_prompt = """
system 
You are an AI assistant for Research Question Tasks, that synthesizes web search results. 
    Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know. 
    keep the answer concise, but provide all of the details you can in the form of a research report. 
    Only make direct references to material if provided in the context. Use only the high quality reports for your response and provide th sources at the end 
    of the report with the corresponding url (link). Provide links only from the web search context
    Also consider the previous conversation if available 
user
Question: {question}
Here is the previous conversation: {history}
Web Search Context: {context} 
Answer: 
assistant
"""

default_router_prompt = """
system
You are an expert at routing a user question to either the generation stage or web search.
Use the web search for questions that require more context for a better answer, or recent events.
Otherwise, you can skip and go straight to the generation phase to respond. 
If you have previous conversation, use that to decide if a web search is needed.
Give a binary choice 'web_search' or 'generate' based on the question. 
Return the JSON with a single key 'choice' with no preamble or explanation.
Question to route: {question} 
Here is the previous conversation for context: {history}
assistant
"""

default_query_prompt = """
system
You are an expert at crafting web search queries for research questions.
More often than not, a user will give you a task or ask a basic question that they wish to learn more about, however it might not be in the best format.
Reword their query to be the most effective web search string possible. Keep the query simple and relevant.
If you have a previous conversation then use it form a coherent search query.
Return the JSON with a single key 'query' with no preamble or explanation.
Question to transform: {question} 
Here is the previous conversation for context: {history}
assistant
"""

# Advanced Features
with st.sidebar.expander("Advanced Options"):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    generate_prompt = st.text_area("Generator Prompt", default_generate_prompt)
    router_prompt = st.text_area("Router Prompt", default_router_prompt)
    query_prompt = st.text_area("Query Prompt", default_query_prompt)

llama3 = ChatOllama(model=local_llm, temperature=temperature)
llama3_json = ChatOllama(model=local_llm, format='json', temperature=temperature)

# Prompt Templates
generate_prompt_template = PromptTemplate(template=generate_prompt, input_variables=["question", "context", "history"])
router_prompt_template = PromptTemplate(template=router_prompt, input_variables=["question", "history"])
query_prompt_template = PromptTemplate(template=query_prompt, input_variables=["question", "history"])

# Chains
generate_chain = generate_prompt_template | llama3 | StrOutputParser()
question_router = router_prompt_template | llama3_json | JsonOutputParser()
query_chain = query_prompt_template | llama3_json | JsonOutputParser()

# Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    search_query: str
    context: str
    history: str

# Node - Generate
def generate(state):
    question = state["question"]
    context = state["context"]
    history = state["history"]
    generation = generate_chain.invoke({"context": context, "question": question, "history": history})
    return {"generation": generation}

# Node - Query Transformation
def transform_query(state):
    question = state['question']
    history = state["history"]
    gen_query = query_chain.invoke({"question": question, "history": history})
    search_query = gen_query["query"]
    return {"search_query": search_query}

# Node - Web Search
def web_search(state):
    search_query = state['search_query']
    search_result = web_search_tool.invoke(search_query)
    return {"context": search_result}

# Conditional Edge, Routing
def route_question(state):
    question = state['question']
    history = state['history']
    output = question_router.invoke({"question": question, "history": history})
    if output['choice'] == "web_search":
        return "websearch"
    elif output['choice'] == 'generate':
        return "generate"

# Build the nodes
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

# Build the edges
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

# Compile the workflow
local_agent = workflow.compile()

def run_agent(query, response_container):
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state[conversation_key]])
    state = {"question": query, "context": "", "history": history}
    
    with st.spinner("Step: Routing Query..."):
        next_step = route_question(state)
    
    if next_step == "websearch":
        with st.spinner("Step: Optimizing Query for Web Search..."):
            state.update(transform_query(state))
        with st.spinner(f"Step: Searching the Web for: '{state['search_query']}'..."):
            state.update(web_search(state))
    
    with st.spinner("Step: Generating Final Response..."):
        state.update(generate(state))
    
    response = state["generation"]
    response_container.markdown(response)
    return response

conversation_key = "chat_history"
if conversation_key not in st.session_state:
    st.session_state[conversation_key] = []

prompt = st.chat_input(f"Ask a question to '{local_llm}' ...")

def st_ollama(model_name, user_question, conversation_key):
    if conversation_key not in st.session_state:
        st.session_state[conversation_key] = []

    print_chat_history_timeline(conversation_key)

    if user_question:
        st.session_state[conversation_key].append({"content": f"{user_question}", "role": "user"})
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.write(user_question)

        response_container = st.empty()
        response = run_agent(user_question, response_container)

        st.session_state[conversation_key].append({"content": response, "role": "assistant"})
        
        return response_container

def print_chat_history_timeline(conversation_key):
    for message in st.session_state[conversation_key]:
        role = message["role"]
        if role == "user":
            with st.chat_message("user", avatar="🧑‍🚀"): 
                question = message["content"]
                st.markdown(f"{question}", unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"], unsafe_allow_html=True)

if prompt:
    st_ollama(local_llm, prompt, conversation_key)

# Save conversation to file
def save_conversation(llm_name, conversation_key):
    OUTPUT_DIR = "llm_conversations"
    OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{OUTPUT_DIR}/{timestamp}_{llm_name.replace(':', '-')}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if st.session_state[conversation_key]:
        if st.sidebar.button("Save conversation"):
            with open(f"{filename}.json", "w") as f:
                json.dump(st.session_state[conversation_key], f, indent=4)
            st.success(f"Conversation saved to {filename}.json")

save_conversation(local_llm, conversation_key)

# Load conversation from file
def load_conversation():
    OUTPUT_DIR = "llm_conversations"
    OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR)
    if os.path.exists(OUTPUT_DIR):
        files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
        selected_file = st.sidebar.selectbox("Select a conversation to load", files)
        if selected_file and st.sidebar.button("Load conversation"):
            with open(os.path.join(OUTPUT_DIR, selected_file), "r") as f:
                st.session_state[conversation_key] = json.load(f)
            print_chat_history_timeline(conversation_key)
            st.success(f"Conversation loaded from {selected_file}")
            #st.rerun()

load_conversation()
if st.session_state[conversation_key]:
    clear_conversation = st.sidebar.button("Clear chat")
    if clear_conversation:
        st.session_state[conversation_key] = []
        st.rerun()
