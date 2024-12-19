import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Model names (make sure you have access on HF)
LLAMA2_7B = "qwen/Qwen1___5-7B-Chat"

LLAMA2_13B_CHAT = "qwen/Qwen1___5-7B-Chat"

 
selected_model = LLAMA2_13B_CHAT
 
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""
 
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
 
embed_model = HuggingFaceEmbedding(model_name="bge-large-zh")
from llama_index.core import Settings
 
Settings.llm = llm
Settings.embed_model = embed_model
from llama_index.core import SimpleDirectoryReader
 
# load documents
documents = SimpleDirectoryReader("context/").load_data()
# print(documents)
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query('芯片的简介')
print('芯片的简介')
print("------------------------------")
print(response)
response = query_engine.query('芯片的基本结构')
print('芯片的基本结构')
print("------------------------------")
print(response)
response = query_engine.query('芯片的主要制造流程')
print('芯片的主要制造流程')
print("------------------------------")
print(response)
response = query_engine.query('芯片的主要类型')
print('芯片的主要类型')
print("------------------------------")
print(response)
response = query_engine.query('芯片的应用领域')
print('芯片的应用领域')
print("------------------------------")
print(response)