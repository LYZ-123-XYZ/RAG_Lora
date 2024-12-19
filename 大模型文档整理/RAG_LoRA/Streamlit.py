# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from llama_index.core.query_engine import CitationQueryEngine
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
# from dotenv import load_dotenv
# import os
# import time
# import ollama
# # Load environment variables
# load_dotenv()
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Set device
# device = 'cuda' 

# # Load models and pipelines
# model_path = './opus-mt-zh-en/'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# translation_pipeline = pipeline("translation", model = model, tokenizer = tokenizer, device = device)

# # Load Llama Index components
# embed_model = HuggingFaceEmbedding(model_name = "bge-m3")
# Settings.llm = None
# Settings.embed_model = embed_model
# documents = SimpleDirectoryReader("archinfo/").load_data()
# index = VectorStoreIndex.from_documents(documents)
# query_engine = CitationQueryEngine.from_args(index, similarity_top_k = 3, citation_chunk_size = 512)
# query_engine = index.as_query_engine()

# # Streamlit app
# def main():
#     st.title("大模型问答系统")

#     action_key = "action_selectbox"
#     user_question_key = "user_question_textarea"
#     button_key = "get_answer_button"

#     # 使用列来布局界面
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         action = st.selectbox("请选择操作：", ["提问", "退出"], key = action_key)
#     with col2:
#         user_question = st.text_area("请输入您的问题：", key = user_question_key)

#     if st.button("获取答案", key = button_key):
#         if user_question:
#             # 添加加载动画
#             with st.spinner('正在处理问题，请稍等...'):
#                 time.sleep(1)  # 模拟处理时间
#                 # Translate question using translation pipeline
#                 translated_question = translation_pipeline(user_question)[0]['translation_text']
#                 response1 = ollama.chat(model = '2739', messages = [{'role': 'user', 'content': user_question}])
#                 # Retrieve answer using Llama Index
#                 response = query_engine.retrieve(translated_question)
#                 st.write(response1['message']['content'])
#                 # 显示结果
#                 if response:
#                     for i, node_with_score in enumerate(response):
#                         node = node_with_score.node
#                         file_name = node.metadata.get('file_name')
#                         text = node.text
#                         # 使用折叠式显示框
#                         with st.expander(f"Source{i + 1}: http://git.enflame.cn/jingming.guo/archinfo/-/blob/main/{file_name}"):
#                             st.write(f"Reference: {text[:500]}")
#                 else:
#                     st.write("没有找到相关答案。")
#         else:
#             st.write("请输入一个问题。")

# if __name__ == "__main__":
#     main()
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
import os
import time
import ollama
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import base64


# Load environment variables
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set device
device = 'cuda'

# Load models and pipelines
model_path = './opus-mt-zh-en/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

# Load Llama Index components
embed_model = HuggingFaceEmbedding(model_name="bge-m3")
Settings.llm = None
Settings.embed_model = embed_model
documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = CitationQueryEngine.from_args(index, similarity_top_k=3, citation_chunk_size=512)
query_engine = index.as_query_engine()


# 设置页面背景颜色和字体
st.markdown(
    """
    <style>
    body {
        color: #333;
        background-color: #f8f9fa;
        font-family: Arial, sans - serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Streamlit app
def main():
    # 优化标题样式
    st.markdown("<h1 style='text - align: center; color: #007BFF;'>大模型问答系统</h1>", unsafe_allow_html=True)

    action_key = "action_selectbox"
    user_question_key = "user_question_textarea"
    button_key = "get_answer_button"

    # 使用列来布局界面，并优化选择框和文本输入框样式
    col1, col2 = st.columns([1, 3])
    with col1:
        action = st.selectbox("请选择操作：", ["提问", "退出"], key=action_key, help="选择你想要执行的操作"
                              )
    with col2:
        user_question = st.text_area("请输入您的问题：", key=user_question_key, help="在这里输入你的问题"
                                     )

    # 优化按钮样式
    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font - size: 16px;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    if st.button("获取答案", key=button_key):
        if user_question:
            # 添加加载动画并优化样式
            with st.spinner('正在处理问题，请稍等...'):
                time.sleep(1)  # 模拟处理时间
                st.markdown("<div style='text - align: center; color: #666;'>正在处理，请耐心等待...</div>", unsafe_allow_html=True)
                # Translate question using translation pipeline
                translated_question = translation_pipeline(user_question)[0]['translation_text']
                response1 = ollama.chat(model='2739', messages=[{'role': 'user', 'content': user_question}])
                # Retrieve answer using Llama Index
                response = query_engine.retrieve(translated_question)
                st.write(response1['message']['content'])
                # 优化结果展示样式
                if response:
                    for i, node_with_score in enumerate(response):
                        node = node_with_score.node
                        file_name = node.metadata.get('file_name')
                        text = node.text
                        with st.expander(f"Source{i + 1}: http://git.enflame.cn/jingming.guo/archinfo/-/blob/main/{file_name}", expanded=False):
                            st.markdown(f"<p style='color: #333;'>Reference: {text[:500]}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='text - align: center; color: #999;'>没有找到相关答案。</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text - align: center; color: #999;'>请输入一个问题。</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()