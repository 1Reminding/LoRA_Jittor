import os
import json
import re
from collections import defaultdict

def count_e2e_types(file_path):
    """统计E2E数据集中的类型"""
    type_counts = {}
    total_entries = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('||')
            if len(parts) < 1:
                continue
            
            total_entries += 1
            # 提取Type字段
            type_match = re.search(r'Type\s*:\s*([^|]+)', parts[0])
            if type_match:
                item_type = type_match.group(1).strip()
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            else:
                # 如果没有Type字段，归类为"无类型"
                type_counts["无类型"] = type_counts.get("无类型", 0) + 1
    
    return type_counts, total_entries

def count_webnlg_types(file_path):
    """统计WebNLG数据集中的类别"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        category_counts = {}
        total_entries = 0
        for i, entry in enumerate(data['entries']):
            total_entries += 1
            category = entry[str(i+1)]['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts, total_entries
    except Exception as e:
        print(f"处理WebNLG数据时出错: {e}")
        return {}, 0

def count_dart_triples(file_path):
    """统计DART数据集中的三元组类型"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        relation_counts = {}
        unique_sources = set()
        total_entries = len(data)
        
        for example in data:
            source_key = "".join(["||".join(triple) for triple in example['tripleset']])
            unique_sources.add(source_key)
            
            for triple in example['tripleset']:
                relation = triple[1].lower()
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        return relation_counts, len(unique_sources), total_entries
    except Exception as e:
        print(f"处理DART数据时出错: {e}")
        return {}, 0, 0

def format_counts_result(name, counts, total_entries, sort=True, include_unique_sources=None):
    """格式化统计结果为字符串"""
    result = f"\n{name} 数据集统计:\n"
    result += "-" * 40 + "\n"
    
    if sort and counts:
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    else:
        items = counts.items()
    
    for category, count in items:
        result += f"{category}: {count}\n"
    
    if counts:
        result += f"总计: {len(counts)}类型, {total_entries}条目\n"
    
    if include_unique_sources is not None:
        result += f"{name}唯一源: {include_unique_sources}\n"
    
    return result

# 在文件开头的import部分后添加这个新函数

def count_jsonl_lines(file_path):
    """统计jsonl文件的行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        return count
    except Exception as e:
        print(f"处理{file_path}时出错: {e}")
        return 0

# 然后在main函数中修改数据集路径和统计方法
def main():
    # 数据集路径
    e2e_train = "/root/LoRA/NLG_pytorch/data/e2e/train.jsonl"
    e2e_valid = "/root/LoRA/NLG_pytorch/data/e2e/valid.jsonl"
    e2e_test = "/root/LoRA/NLG_pytorch/data/e2e/test.jsonl"
    
    webnlg_train = "/root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/train.jsonl"
    webnlg_valid = "/root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/valid.jsonl"
    webnlg_test = "/root/LoRA/NLG_pytorch/data/webnlg_challenge_2017/test.jsonl"
    
    dart_train = "/root/LoRA/NLG_pytorch/data/dart/train.jsonl"
    dart_valid = "/root/LoRA/NLG_pytorch/data/dart/valid.jsonl"
    dart_test = "/root/LoRA/NLG_pytorch/data/dart/test.jsonl"
    
    # 收集所有结果
    all_results = []
    
    # 统计数据集总条目数
    dataset_totals = {
        "E2E": 0,
        "WebNLG": 0,
        "DART": 0
    }
    
    # 按类别统计各数据集的分布
    e2e_category_stats = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0, 'total': 0})
    webnlg_category_stats = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0, 'total': 0})
    dart_category_stats = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0, 'total': 0})
    
    # 记录每个数据集的训练、验证、测试集总数
    e2e_split_totals = {'train': 0, 'valid': 0, 'test': 0}
    webnlg_split_totals = {'train': 0, 'valid': 0, 'test': 0}
    dart_split_totals = {'train': 0, 'valid': 0, 'test': 0}
    
    # 统计E2E数据集
    if os.path.exists(e2e_train):
        e2e_train_entries = count_jsonl_lines(e2e_train)
        dataset_totals["E2E"] += e2e_train_entries
        e2e_split_totals['train'] = e2e_train_entries
        all_results.append(f"\nE2E训练集: {e2e_train_entries}条目")
        
        # 由于jsonl文件不包含类型信息，我们将所有条目归为"无类型"
        e2e_category_stats["无类型"]['train'] = e2e_train_entries
        e2e_category_stats["无类型"]['total'] += e2e_train_entries
    else:
        all_results.append(f"文件不存在: {e2e_train}")
    
    if os.path.exists(e2e_valid):
        e2e_valid_entries = count_jsonl_lines(e2e_valid)
        dataset_totals["E2E"] += e2e_valid_entries
        e2e_split_totals['valid'] = e2e_valid_entries
        all_results.append(f"E2E验证集: {e2e_valid_entries}条目")
        
        e2e_category_stats["无类型"]['valid'] = e2e_valid_entries
        e2e_category_stats["无类型"]['total'] += e2e_valid_entries
    
    if os.path.exists(e2e_test):
        e2e_test_entries = count_jsonl_lines(e2e_test)
        dataset_totals["E2E"] += e2e_test_entries
        e2e_split_totals['test'] = e2e_test_entries
        all_results.append(f"E2E测试集: {e2e_test_entries}条目")
        
        e2e_category_stats["无类型"]['test'] = e2e_test_entries
        e2e_category_stats["无类型"]['total'] += e2e_test_entries
    
    # 统计WebNLG数据集
    if os.path.exists(webnlg_train):
        webnlg_train_entries = count_jsonl_lines(webnlg_train)
        dataset_totals["WebNLG"] += webnlg_train_entries
        webnlg_split_totals['train'] = webnlg_train_entries
        all_results.append(f"\nWebNLG训练集: {webnlg_train_entries}条目")
        
        webnlg_category_stats["无类型"]['train'] = webnlg_train_entries
        webnlg_category_stats["无类型"]['total'] += webnlg_train_entries
    else:
        all_results.append(f"文件不存在: {webnlg_train}")
    
    if os.path.exists(webnlg_valid):
        webnlg_valid_entries = count_jsonl_lines(webnlg_valid)
        dataset_totals["WebNLG"] += webnlg_valid_entries
        webnlg_split_totals['valid'] = webnlg_valid_entries
        all_results.append(f"WebNLG验证集: {webnlg_valid_entries}条目")
        
        webnlg_category_stats["无类型"]['valid'] = webnlg_valid_entries
        webnlg_category_stats["无类型"]['total'] += webnlg_valid_entries
    
    if os.path.exists(webnlg_test):
        webnlg_test_entries = count_jsonl_lines(webnlg_test)
        dataset_totals["WebNLG"] += webnlg_test_entries
        webnlg_split_totals['test'] = webnlg_test_entries
        all_results.append(f"WebNLG测试集: {webnlg_test_entries}条目")
        
        webnlg_category_stats["无类型"]['test'] = webnlg_test_entries
        webnlg_category_stats["无类型"]['total'] += webnlg_test_entries
    
    # 统计DART数据集
    if os.path.exists(dart_train):
        dart_train_entries = count_jsonl_lines(dart_train)
        dataset_totals["DART"] += dart_train_entries
        dart_split_totals['train'] = dart_train_entries
        all_results.append(f"\nDART训练集: {dart_train_entries}条目")
        
        dart_category_stats["无类型"]['train'] = dart_train_entries
        dart_category_stats["无类型"]['total'] += dart_train_entries
    else:
        all_results.append(f"文件不存在: {dart_train}")
    
    if os.path.exists(dart_valid):
        dart_valid_entries = count_jsonl_lines(dart_valid)
        dataset_totals["DART"] += dart_valid_entries
        dart_split_totals['valid'] = dart_valid_entries
        all_results.append(f"DART验证集: {dart_valid_entries}条目")
        
        dart_category_stats["无类型"]['valid'] = dart_valid_entries
        dart_category_stats["无类型"]['total'] += dart_valid_entries
    
    if os.path.exists(dart_test):
        dart_test_entries = count_jsonl_lines(dart_test)
        dataset_totals["DART"] += dart_test_entries
        dart_split_totals['test'] = dart_test_entries
        all_results.append(f"DART测试集: {dart_test_entries}条目")
        
        dart_category_stats["无类型"]['test'] = dart_test_entries
        dart_category_stats["无类型"]['total'] += dart_test_entries
    
    # 在最后统一输出所有结果
    print("\n数据集统计结果汇总")
    print("=" * 50)
    for result in all_results:
        print(result)
    
    # 打印三个数据集的总条目数
    print("\n三个数据集总条目数汇总")
    print("=" * 50)
    for dataset, total in dataset_totals.items():
        print(f"{dataset}数据集总条目数: {total}")
    print(f"所有数据集总条目数: {sum(dataset_totals.values())}")
    
    # 打印按类别统计的详细分布
    print("\nE2E数据集按类别统计")
    print("=" * 50)
    print(f"{'类别':<20} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
    for category, stats in sorted(e2e_category_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{category:<20} {stats['train']:<10} {stats['valid']:<10} {stats['test']:<10} {stats['total']:<10}")
    # 添加总计行
    print(f"{'总计':<20} {e2e_split_totals['train']:<10} {e2e_split_totals['valid']:<10} {e2e_split_totals['test']:<10} {sum(e2e_split_totals.values()):<10}")
    
    print("\nWebNLG数据集按类别统计")
    print("=" * 50)
    print(f"{'类别':<20} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
    for category, stats in sorted(webnlg_category_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{category:<20} {stats['train']:<10} {stats['valid']:<10} {stats['test']:<10} {stats['total']:<10}")
    # 添加总计行
    print(f"{'总计':<20} {webnlg_split_totals['train']:<10} {webnlg_split_totals['valid']:<10} {webnlg_split_totals['test']:<10} {sum(webnlg_split_totals.values()):<10}")
    
    print("\nDART数据集按类别统计 (仅显示前20个类别)")
    print("=" * 50)
    print(f"{'类别':<20} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
    for category, stats in sorted(dart_category_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20]:
        print(f"{category:<20} {stats['train']:<10} {stats['valid']:<10} {stats['test']:<10} {stats['total']:<10}")
    # 添加总计行
    print(f"{'总计':<20} {dart_split_totals['train']:<10} {dart_split_totals['valid']:<10} {dart_split_totals['test']:<10} {sum(dart_split_totals.values()):<10}")

if __name__ == "__main__":
    main()