import json

# 输入的JSONL文件路径
input_file_path = 'train.jsonl'
# 输出的JSONL文件路径
output_file_path = 'test.jsonl'

# 读取原始JSONL文件并修改output字段
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析JSON对象
        obj = json.loads(line)
        # 将output字段置空
        obj['output'] = ""
        # 将修改后的JSON对象转换为字符串并写入新文件
        outfile.write(json.dumps(obj) + '\n')

print("JSONL文件中的output字段已置空。")