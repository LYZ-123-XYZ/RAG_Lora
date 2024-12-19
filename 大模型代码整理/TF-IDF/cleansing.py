import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取JSONL文件并解析每一行
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

# 加载JSONL数据并进行中文分词
def load_data(jsonl_file):
    data = read_jsonl(jsonl_file)
    segmented_data = []
    for item in data:
        # 确保item['input']是字符串类型
        if isinstance(item['input'], bytes):
            item['input'] = item['input'].decode('utf-8')
        # 使用jieba进行中文分词
        segmented_input = ' '.join(jieba.cut(item['input']))
        segmented_data.append({'input': segmented_input, 'output': item['output']})
    return segmented_data

# 计算输入之间的相似度并过滤数据
def calculate_similarity_and_filter(data, threshold):
    inputs = [item['input'] for item in data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(inputs)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    to_delete = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                # 比较输出长度，删除较短的输出对应的输入
                if len(data[i]['output']) < len(data[j]['output']):
                    to_delete.append(i)
                else:
                    to_delete.append(j)
    to_delete = set(to_delete)
    filtered_data = [data[i] for i in range(len(data)) if i not in to_delete]
    return filtered_data


def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        new_lines = [json.dumps(item, ensure_ascii=False) + '\n' for item in data]
        file.writelines(new_lines)


def main(jsonl_file, threshold, output_file):
    data = load_data(jsonl_file)
    filtered_data = calculate_similarity_and_filter(data, threshold)
    write_jsonl(filtered_data, output_file)
    print(f"Filtered data has been written to {output_file}")

# 运行主函数
if __name__ == "__main__":
    threshold = 0.7  # 设置相似度阈值
    input_jsonl_file = 'new_train.jsonl'  
    output_jsonl_file = 'filter7_data.jsonl'  
    main(input_jsonl_file, threshold, output_jsonl_file)