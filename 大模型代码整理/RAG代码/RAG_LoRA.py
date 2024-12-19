# from googletrans import Translator
import ollama 
# 创建一个Translator对象
# translator = Translator()
# 要翻译的中文文本
# text_to_translate = "为什么公司急于将生成式AI部署到其内部工作流程或面向客户的应用程序中？"

# # 翻译文本，目标语言为英语
# translated_text = translator.translate(text_to_translate, dest='en')
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


Llama3_LoRA = "llama-3-chinese-8b-instruct"

 
selected_model = Llama3_LoRA
 
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
documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 此处我们可以控制引用来源的粒度，默认值为 512
    citation_chunk_size=256,
)
query_engine = index.as_query_engine()
response = query_engine.query(f"Why are companies rushing to deploy generative AI into their internal workflows or customer-facing applications?")
formatted_prompt = f"为什么公司急于将生成式AI部署到其内部工作流程或面向客户的应用程序中？"  
response1 = ollama.chat(model='Lora_16', messages=[{'role': 'user', 'content': formatted_prompt}])  
print('为什么公司急于将生成式AI部署到其内部工作流程或面向客户的应用程序中？')
print("------------------------------")
print(response1['message']['content'])
print("------------------------------")
print(response.source_nodes[0].node.get_text())
print("######################################")
print(response.source_nodes[0].get_text())
