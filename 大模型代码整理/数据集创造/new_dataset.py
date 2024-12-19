import os
import json
def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["instruction"]
            catagory = data["input"]
            label = data["output"]
            message = {
                "instruction": "You are an expert in the field of chips and GPUs. Please answer the following questions based on your given knowledge base. Please note that if the question is not in the knowledge base or is not related to chips, GPUs, etc., please do not give any links in your answer, as this will lead to serious errors.",
                "input": f"{context}",
                "output": label,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")



# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "test.jsonl"

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"

dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)