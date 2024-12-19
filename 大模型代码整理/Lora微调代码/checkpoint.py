from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = './qwen/Qwen1___5-7B-Chat/'
lora_path = 'output/Qwen1.5/checkpoint-250'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)


import csv
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
jsonl_file = 'new_test1.jsonl'
result = 'result.jsonl'
# 生成JSONL文件
messages = []


# 读取jsonl文件
with open(jsonl_file, 'r') as file:
    for line in file:
        # 解析每一行的json数据
        data = json.loads(line)
        context = data["input"]
        instruction = data["instruction"]
        # prompt = f'{context},{catagory}'
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": context}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        message = response
        print(response)
        messages.append(message)

# 保存为JSONL文件
with open(result, 'w', encoding='utf-8') as file:
    for message in messages:
        file.write(json.dumps(message, ensure_ascii=False) + '\n')