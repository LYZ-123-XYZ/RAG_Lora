import torch
from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core import PromptTemplate
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
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

LLAMA2_7B = "qwen/Qwen1___5-7B-Chat"

LLAMA2_13B_CHAT = "qwen/Qwen1___5-7B-Chat"

 
selected_model = LLAMA2_13B_CHAT
 
SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. """
# - Generate human readable output, avoid creating output with gibberish text.
# - Generate only the requested output, don't include any other language before or after the requested output.
# - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
# - Generate professional language typically used in business documents in North America.
# - Never generate offensive or foul language.
# """
 
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
documents = SimpleDirectoryReader("datasets/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 此处我们可以控制引用来源的粒度，默认值为 512
    citation_chunk_size=256,
)
query_engine = index.as_query_engine()
response = query_engine.query("What are the drawbacks of using TSMC's SoIC for HBM packaging?")
print('What are the drawbacks of using "TSMC"s SoIC for HBM packaging?')
print("------------------------------")
print("Answer：" + str(response))
print("------------------------------")
print(response.source_nodes[0].node.get_text())
print("######################################")
print(response.source_nodes[0].get_text())
print("######################################")
response = query_engine.query('芯片的基本结构')
print('芯片的基本结构')
print("------------------------------")
print("回答："+ str(response))
print("------------------------------")
for source in response.source_nodes:
    print(source.node.get_text())
    print("######################################")
response = query_engine.query('芯片的主要制造流程')
print('芯片的主要制造流程')
print("------------------------------")
print("回答："+ str(response))
print("------------------------------")
for source in response.source_nodes:
    print(source.node.get_text())
    print("######################################")
response = query_engine.query('芯片的主要类型')
print('芯片的主要类型')
print("------------------------------")
print("回答："+ str(response))
print("------------------------------")
for source in response.source_nodes:
    print(source.node.get_text())
    print("######################################")
response = query_engine.query('芯片的应用领域')
print('芯片的应用领域')
print("------------------------------")
print("回答："+ str(response))
print("------------------------------")
for source in response.source_nodes:
    print(source.node.get_text())
    print("######################################")