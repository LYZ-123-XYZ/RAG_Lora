# import argparse
 
# import torch
# from peft import PeftModel, PeftConfig
# from transformers import (
#     AutoModel,
#     AutoTokenizer,
#     BloomForCausalLM,
#     BloomTokenizerFast,
#     AutoModelForCausalLM,
#     LlamaTokenizer,
#     LlamaForCausalLM,
#     AutoModelForSequenceClassification,
# )
 
# MODEL_CLASSES = {
#     "bloom": (BloomForCausalLM, BloomTokenizerFast),
#     "chatglm": (AutoModel, AutoTokenizer),
#     "llama": (LlamaForCausalLM, LlamaTokenizer),
#     "baichuan": (AutoModelForCausalLM, AutoTokenizer),
#     "auto": (AutoModelForCausalLM, AutoTokenizer),
# }
 
 
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_type', default="auto", type=str, required=False)
#     parser.add_argument('--tokenizer_path', default=None, type=str,
#                         help="Please specify tokenization path.")
 
#     parser.add_argument('--output_dir', default='./merged', type=str)
#     args = parser.parse_args()
 
 
#     base_model_path = "./qwen/Qwen1___5-7B-Chat"
#     lora_model_path = "output/Qwen1.5/checkpoint-250"
#     output_dir = args.output_dir
#     peft_config = PeftConfig.from_pretrained(lora_model_path)
#     model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
 
#     # 模型加载
#     if peft_config.task_type == "SEQ_CLS":
#         if args.model_type == "qwen":
#             raise ValueError("qwen does not support sequence classification")
#         base_model = AutoModelForSequenceClassification.from_pretrained(
#             base_model_path,
#             num_labels=1,
#             load_in_8bit=False,
#             torch_dtype=torch.float32,
#             trust_remote_code=True,
#             device_map="auto",
#         )
#     else:
#         base_model = model_class.from_pretrained(
#             base_model_path,
#             load_in_8bit=False,
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#             device_map="auto",
#         )
    
#     # 分词器加载
#     if args.tokenizer_path:
#         tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
#     else:
#         tokenizer = tokenizer_class.from_pretrained(base_model_path, trust_remote_code=True)
 
#     # 修改词表大小
#     # if args.resize_emb:
#     #     base_model_token_size = base_model.get_input_embeddings().weight.size(0)
#     #     if base_model_token_size != len(tokenizer):
#     #         base_model.resize_token_embeddings(len(tokenizer))
 
#     # 初始化Peft新模型
#     new_model = PeftModel.from_pretrained(
#         base_model,
#         lora_model_path,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
#     new_model.eval()
#     new_base_model = new_model.merge_and_unload()
 
#     tokenizer.save_pretrained(output_dir)
#     new_base_model.save_pretrained(output_dir, safe_serialization=False, max_shard_size='10GB')
 
# if __name__ == '__main__':
#     main()

# /* 将checkpoint转换为LoRA格式 */
# from transformers import AutoModelForSequenceClassification,AutoTokenizer
# import os
 
# # 需要保存的lora路径
# lora_path= "./qwen/Qwen1___5-7B-Chat-LoRA"
# # 模型路径
# model_path = './qwen/Qwen1___5-7B-Chat'
# # 检查点路径
# checkpoint_dir = './output/Qwen1.5_600/checkpoint-1850'
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
# # 保存模型
# model.save_pretrained(lora_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# # 保存tokenizer
# tokenizer.save_pretrained(lora_path)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
 
model_path = './qwen/Qwen1___5-7B-Chat'
lora_path = "./qwen/Qwen1___5-7B-Chat-LoRA"
device = 'cuda' 
# 合并后的模型路径
output_path = './qwen/Qwen1___5-7B-Chat-Merged'
 
# 等于训练时的config参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
 
base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
base_tokenizer = AutoTokenizer.from_pretrained(model_path)
lora_model = PeftModel.from_pretrained(
    base,
    lora_path,
    torch_dtype=torch.float16,
    config=config
)
model = lora_model.merge_and_unload()
model.save_pretrained(output_path)
base_tokenizer.save_pretrained(output_path)