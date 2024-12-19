
import ollama 
import torch
import json
import csv
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
jsonl_file = 'TEST.jsonl'
result = 'new__result.jsonl'
device = 'cuda'
# 生成JSONL文件
messages = []

model_path = './opus-mt-zh-en/'  
#创建tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path) 
#创建模型 
model = AutoModelForSeq2SeqLM.from_pretrained(model_path) 
#创建pipeline
pipeline = pipeline("translation", model=model, tokenizer=tokenizer, device='cuda')
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


Llama3_LoRA = "llama-3-chinese-8b-instruct"

 
selected_model = Llama3_LoRA
 
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. """
 
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)





llm = HuggingFaceLLM(context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto"
)



embed_model = HuggingFaceEmbedding(model_name="bge-large-zh")
 
Settings.llm = llm
Settings.embed_model = embed_model
 
# load documents
documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 此处我们可以控制引用来源的粒度，默认值为 512
    citation_chunk_size=256,
)
query_engine = index.as_query_engine()
with open('output3.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile) 
    # 写入表头
    csvwriter.writerow(['row1', 'row2', 'row3', 'row4'])
    # 读取jsonl文件
    with open(jsonl_file, 'r') as file:
        for i,line in enumerate(file):
            # 解析每一行的json数据
            row = [i+1]
            data = json.loads(line)
            context = data["input"]
            response0 = pipeline(context)
            response = query_engine.query(response0[0]['translation_text'])
            formatted_prompt = context  
            response1 = ollama.chat(model='Lora_16', messages=[{'role': 'user', 'content': formatted_prompt}])  
            file_name = response1['message']['content'].split('/')[-1]
            print(context)
            for j in range(2):
                if response.source_nodes[j].node.metadata.get('file_name') == file_name:
                    row.append(1)
                else:
                    row.append(0)
            csvwriter.writerow(row)
            print(response0[0]['translation_text'])
            print("------------------------------")
            print(response1['message']['content'])
            print("------------------------------")
            print(response)
            print("------------------------------")
            print(response.source_nodes)
            print("------------------------------")
            print(response.source_nodes[0].node.metadata.get('file_name'))
            print("------------------------------")
            print(response.source_nodes[0].node.get_text())
            print("######################################")
            # print(response.source_nodes[0].get_text())
