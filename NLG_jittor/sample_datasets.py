import json
import os
import random
import numpy as np
from collections import defaultdict
from sklearn import model_selection  

# 设置随机种子，确保结果可重现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 抽样比例
SAMPLE_RATIO = 0.18

# 数据集路径
DATASET_PATHS = {
    'e2e': '/root/LoRA/NLG_pytorch/data/e2e',
    'webnlg': '/root/LoRA/NLG_pytorch/data/webnlg_challenge_2017',
    'dart': '/root/LoRA/NLG_pytorch/data/dart'
}

# 文件后缀
FILE_SUFFIXES = ['train_formatted.jsonl', 'valid_formatted.jsonl', 'test_formatted.jsonl']

# 创建输出目录
def create_output_dirs():
    for dataset in DATASET_PATHS.keys():
        output_dir = os.path.join(DATASET_PATHS[dataset], 'sampled')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

# 从JSONL文件中读取数据
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 将数据写入JSONL文件
def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 从WebNLG数据中提取类别
def extract_category_webnlg(item):
    # WebNLG数据集在context中包含类别信息
    context = item.get('context', '')
    # 尝试从context中提取类别
    if 'category:' in context:
        parts = context.split('category:')
        if len(parts) > 1:
            category = parts[1].strip().split()[0]
            return category
    # 如果没有找到类别，使用'cate'字段
    return 'cate_' + str(item.get('cate', False))

# 从DART数据中提取类别
def extract_category_dart(item):
    # DART数据集在context中包含三元组信息
    context = item.get('context', '')
    # 尝试从第一个三元组中提取谓词作为类别
    if '[' in context and ']' in context:
        triples = context.split('\n')
        for triple in triples:
            if '[' in triple and ']' in triple:
                parts = triple.strip().split('[')[1].split(']')[0].split(', ')
                if len(parts) >= 2:
                    return parts[1]  # 使用谓词作为类别
    return 'unknown'

# 从E2E数据中提取类别
def extract_category_e2e(item):
    # E2E数据集在context中包含属性信息
    context = item.get('context', '')
    # 尝试提取name属性作为类别
    if 'name[' in context:
        parts = context.split('name[')
        if len(parts) > 1:
            name_part = parts[1].split(']')[0]
            return name_part
    return 'unknown'

# 按类别进行分层抽样
def stratified_sample(data, extract_category_fn, ratio=SAMPLE_RATIO):
    # 按类别分组
    category_data = defaultdict(list)
    for i, item in enumerate(data):
        category = extract_category_fn(item)
        category_data[category].append((i, item))
    
    # 对每个类别进行抽样
    sampled_indices = []
    for category, items in category_data.items():
        indices = [i for i, _ in items]
        sample_size = max(1, int(len(indices) * ratio))
        sampled_indices.extend(random.sample(indices, sample_size))
    
    # 排序以保持原始顺序
    sampled_indices.sort()
    return [data[i] for i in sampled_indices]

# 处理数据集
def process_dataset(dataset_name):
    dataset_path = DATASET_PATHS[dataset_name]
    output_dir = os.path.join(dataset_path, 'sampled')
    
    # 选择合适的类别提取函数
    if dataset_name == 'webnlg':
        extract_category_fn = extract_category_webnlg
    elif dataset_name == 'dart':
        extract_category_fn = extract_category_dart
    else:  # e2e
        extract_category_fn = extract_category_e2e
    
    # 处理每个文件
    for suffix in FILE_SUFFIXES:
        input_file = os.path.join(dataset_path, suffix)
        output_file = os.path.join(output_dir, suffix)
        
        if os.path.exists(input_file):
            print(f"处理 {input_file}...")
            data = read_jsonl(input_file)
            sampled_data = stratified_sample(data, extract_category_fn)
            write_jsonl(sampled_data, output_file)
            print(f"  原始数据: {len(data)}条, 抽样后: {len(sampled_data)}条")

# 主函数
def main():
    create_output_dirs()
    
    for dataset_name in DATASET_PATHS.keys():
        print(f"\n处理 {dataset_name} 数据集...")
        process_dataset(dataset_name)

if __name__ == "__main__":
    main()