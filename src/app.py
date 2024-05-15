import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

load_dotenv()

def update_progress_bar(progress_bar, percentage):
    progress_bar.progress(percentage)

def get_response(user_input):
    progress_bar = st.progress(0)
    update_progress_bar(progress_bar, 0)
    retriever_chain = get_context_retrieval_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    for i in range(10):
        time.sleep(0.5)
        update_progress_bar(progress_bar, (i + 1) * 10)
    response = conversation_rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        }) 
    update_progress_bar(progress_bar, 100)
    return response['answer']


def get_vectorstore_from_url(url):
    #get text from the website in document form
    loader = WebBaseLoader(url)
    documents = loader.load()
    #splitting docs in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents_chunks = text_splitter.split_documents(documents)
    #create a vector store
    vectorstore = Chroma.from_documents(documents_chunks, OpenAIEmbeddings())

    return vectorstore

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions as best you can using the given context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_context_retrieval_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain      


#config
st.set_page_config(page_title="Talk with Web", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
custom_css = f"""
<style>
    .st-au {{
    background-color: white;
    color: black;
    }}
    
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Talk with your website")
#sidebar
with st.sidebar:    
    website_url = st.text_input("Your URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, how can I help you?"),
    ] 
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

#user input
    user_query = st.chat_input("Type your message here")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


#conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)