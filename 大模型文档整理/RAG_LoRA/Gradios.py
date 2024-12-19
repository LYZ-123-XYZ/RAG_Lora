import gradio as gr
import ollama 
import torch
import re
import csv
from filelock import FileLock
from llama_index.llms.huggingface import HuggingFaceLLM
import os
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    PromptTemplate,
)
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import time
from vllm import LLM
from datetime import datetime, timedelta

model_path = './opus-mt-zh-en/'  
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForSeq2SeqLM.from_pretrained(model_path) 
pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device='cuda')

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

embed_model = HuggingFaceEmbedding(model_name="bge-m3")

Settings.llm = None
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)
query_engine = index.as_query_engine()

def process_question(question):
    response0 = pipeline(question)
    start_time = time.time()
    response = query_engine.retrieve(response0[0]['translation_text'])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"RAG用时: {execution_time} seconds")
    formatted_prompt = question
    start_time = time.time()
    response1 = ollama.chat(model='2739', messages=[{'role': 'user', 'content': formatted_prompt}])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Ollama 微调用时: {execution_time} seconds")
    response1_content = response1['message']['content']
    extracted_data = []
    for node_with_score in response:
        node = node_with_score.node
        file_name = node.metadata.get('file_name')
        text = node.text
        extracted_data.append({'file_name': file_name, 'text': text})
    formatted_extracted_data = []
    formatted_extracted_data2 = []
    if len(extracted_data) >= 2:
        for i, data in enumerate(extracted_data[:2]):
            if i == 0:
                data['file_name'] = "http://git.enflame.cn/jingming.guo/archinfo/-/blob/main/" + data['file_name']
                data['file_name'] = f'<a href="{data["file_name"]}" target="_blank">{data["file_name"]}</a>'
                formatted_extracted_data.append([data['file_name'] + '<br>' + data['text'] + '<br>'])
            else:
                data['file_name'] = "http://git.enflame.cn/jingming.guo/archinfo/-/blob/main/" + data['file_name']
                data['file_name'] = f'<a href="{data["file_name"]}" target="_blank">{data["file_name"]}</a>'
                formatted_extracted_data2.append([data['file_name'] + '<br>' + data['text']])
    else:
        for _ in range(2):
            formatted_extracted_data.append(['', ''])
    current_time = datetime.now() + timedelta(hours=8)
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    with FileLock('user_input_output.csv.lock'):
        with open('user_input_output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([current_time_str, question, response1_content, formatted_extracted_data])
    return response1_content, formatted_extracted_data, formatted_extracted_data2

def custom_css():
    return """
  .input-container {
        display: flex;
        justify-content: space-between;
    }
  .answer-box {
        margin-top: 20px;
    }
  .reference-box {
        margin-top: 20px;
        border-top: 1px solid #ccc;
        padding-top: 10px;
    }
    """

iface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(label="Question:", elem_id="question_box"),
    outputs=[
        gr.Textbox(label="Answer:", elem_id="answer_box"),  
        gr.HTML(label="Reference1 about this question", show_label=True, elem_id="reference_box"),
        gr.HTML(label="Reference2 about this question", show_label=True, elem_id="reference_box")
    ],
    title="量子芯核",
    description="请输入您想提问的问题，获取对应回答以及问题相关文本。",
    submit_btn='发送',
    clear_btn='清空',
    css=custom_css()
)

iface.launch(share = True)