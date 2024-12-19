import streamlit as st  
import ollama  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import WebBaseLoader  
from langchain_community.vectorstores import Chroma  
from langchain_community.embeddings import OllamaEmbeddings  
  
st.title("Chat with Webpage 🌐")  
st.caption("This app allows you to chat with a webpage using local Llama-3 and RAG")  
  
# Get the webpage URL from the user  
webpage_url = st.text_input("Enter Webpage URL", type="default")  

if webpage_url:  
    # 1. Load the data  
    loader = WebBaseLoader(webpage_url)  
    docs = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
    splits = text_splitter.split_documents(docs)  

    # 2. Create Ollama embeddings and vector store  
    embeddings = OllamaEmbeddings(model="Lora_16")  
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # 3. Call Ollama Llama3 model  
    def ollama_llm(question, context):  
        formatted_prompt = f"Question: {question}\n\nContext: {context}"  
        response = ollama.chat(model='Lora_16', messages=[{'role': 'user', 'content': formatted_prompt}])  
        return response['message']['content']

    # 4. RAG Setup  
    # retriever = vectorstore.as_retriever(search_kwargs={'k': 1})  
    retriever = vectorstore.as_retriever()  
  
    def combine_docs(docs):  
        return "\n\n".join(doc.page_content for doc in docs)  
  
    def rag_chain(question):  
    #     retrieved_paragraphs = retriever.retrieve(question)
    
    # # 将检索到的段落合并为一个上下文字符串
    #     formatted_context = "\n\n".join(paragraph.page_content for paragraph in retrieved_paragraphs)
        retrieved_docs = retriever.invoke(question)  
        retrieved_content = []
        for doc in retrieved_docs:
        # 假设每个文档对象都有一个属性 page_content 来获取文档内容
            retrieved_content.append(doc.page_content)
        st.write("Retrieved Content:")
        # st.write(retrieved_docs)
        for content in retrieved_content:
            st.write(content)
            # break
        formatted_context = combine_docs(retrieved_docs) 

        return ollama_llm(question, formatted_context) 
  
    st.success(f"Loaded {webpage_url} successfully!")

    # Ask a question about the webpage  
    prompt = st.text_input("Ask any question about the webpage")  
  
    # Chat with the webpage  
    if prompt:  
        result = rag_chain(prompt)  
        st.write(result)
