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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import time
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer




translator = pipeline("translation", model="./NLLB/facebook/nllb-200-distilled-600M/", truncation=True, max_length=600)

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = 'cuda'
embed_model = HuggingFaceEmbedding(model_name="bge-m3")

Settings.llm = None
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("archinfo/").load_data()
index = VectorStoreIndex.from_documents(documents)

new_tokenizer = AutoTokenizer.from_pretrained('bge-reranker-base')
new_model = AutoModelForSequenceClassification.from_pretrained('bge-reranker-base')
new_model.eval()
mode_path = './Llama3.1/'

local_tokenizer = AutoTokenizer.from_pretrained(mode_path)
local_model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)


def rewrite_question(question):
    """
    使用本地模型对输入问题进行改写
    """
    input_ids = local_tokenizer.encode(question, return_tensors="pt").to(device)
    generated_ids = local_model.generate(
        input_ids,
        max_new_tokens=50,  
        do_sample=True,
        temperature=0.7  
    )
    rewritten_question = local_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return rewritten_question


def process_question(rewritten_question, similarity_top_k):
    translation_en = translator(rewritten_question, src_lang="zho_Hans", tgt_lang="eng_Latn")
    translated_question = translation_en[0]['translation_text']
    start_time = time.time()
    query_engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=similarity_top_k,
        citation_chunk_size=512,
    )
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    response = query_engine.retrieve(translated_question)
    query = translated_question
    scored_data = []
    with torch.no_grad():
        pairs = [[query, node_with_score.node.text] for node_with_score in response]
        inputs = new_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = new_model(**inputs, return_dict=True).logits.view(-1, ).float()
        for node_with_score, score in zip(response, scores):
            node = node_with_score.node
            file_name = node.metadata.get('file_name')
            text = node.text
            scored_data.append((file_name, text, score.item()))


    top_two = sorted(scored_data, key=lambda x: x[2], reverse=True)[:2]
    top_two_data = [{'file_name': item[0], 'text': item[1]} for item in top_two]
    end_time = time.time()
    execution_time_rag = end_time - start_time
    source1 = scored_data[0][1]
    source2 = scored_data[1][1]
    # 确保来源文本不超过剩余长度
    start_time = time.time()
    prompt = "You are an expert in software and hardware co-design in the field of chip design, familiar with software, hardware, interconnection, process, packaging, etc. Please provide an answer based solely on the the provided sources. Source 1:" + source1 + ",Source 2:" + source2
    local_messages = [
        {"role": "System", "content": prompt},
        {"role": "User", "content": query}
    ]
    local_text = local_tokenizer.apply_chat_template(local_messages, tokenize=False, add_generation_prompt=True)
    local_model_inputs = local_tokenizer([local_text], return_tensors="pt", max_length=8192).to(device)
    local_generated_ids = local_model.generate(
        local_model_inputs.input_ids,
        max_new_tokens=1024
    )
    local_generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(local_model_inputs.input_ids, local_generated_ids)
    ]
    local_response = local_tokenizer.batch_decode(local_generated_ids, skip_special_tokens=True)[0]
    response1_content = local_response
    translation_zh = translator(response1_content, src_lang="eng_Latn", tgt_lang="zho_Hans")
    response1_content = translation_zh[0]['translation_text']
    end_time = time.time()
    execution_time_llama3 = end_time - start_time
    # response1_content = convert_links_to_html(response1_content)
    extracted_data = []
    for node_with_score in response:
        node = node_with_score.node
        file_name = node.metadata.get('file_name')
        text = node.text
        extracted_data.append({'file_name': file_name, 'text': text})
    formatted_extracted_data = []
    formatted_extracted_data2 = []
    if len(top_two_data) >= 2:
        for i, data in enumerate(top_two_data[:2]):
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
            writer.writerow([current_time_str, rewritten_question, response1_content, formatted_extracted_data])

    # 清空相关缓存
    torch.cuda.empty_cache()  # 清空CUDA缓存
    if hasattr(local_model, 'reset_cache'):  # 如果模型有reset_cache方法，调用它来清空模型内部缓存
        local_model.reset_cache()

    return response1_content, formatted_extracted_data, formatted_extracted_data2, f"RAG用时: {execution_time_rag} seconds", f"Llama3用时: {execution_time_llama3} seconds"


def custom_css():
    return """
.input-container {
        display: flex;
        justify-content: space-between;
    }
.answer-box {
        margin_top: 20px;
    }
.reference-box {
        margin_top: 20px
        border_top: 1px  solid #ccc;
        padding_top: 10px;
    }
    """


iface = gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(label="Question:", elem_id="question_box"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Similarity Top K", value=3)
    ],
    outputs=[
        gr.HTML(label="Answer:", show_label=True, elem_id="answer_box"),
        gr.HTML(label="Reference1 about this question", show_label=True, elem_id="reference_box"),
        gr.HTML(label="Reference2 about this question", show_label=True, elem_id="reference_box"),
        gr.Textbox(label="RAG Time", show_label=True),
        gr.Textbox(label="Llama3 Time", show_label=True)
    ],
    title="量子芯核",
    description="请输入您想提问的问题，获取对应回答以及您的问题相关文本。",
    submit_btn='发送',
    clear_btn='清空',
    css=custom_css()
)

iface.launch()