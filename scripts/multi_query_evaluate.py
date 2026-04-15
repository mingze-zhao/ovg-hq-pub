import os
import csv
import argparse
from datetime import datetime

def extract_header_and_second_line(file_path):
    """读取CSV文件的表头和第二行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # 获取表头
            second_line = next(reader)  # 获取第二行
            return header, second_line
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None
    
def create_timestamped_dir(base_dir):
    """创建带时间戳的目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, f"evaluation_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir

def aggregate_results(input_dir, output_file):
    """聚合多查询类型的评估结果"""
    # 定义要处理的文件列表
    files = [
        "metrics_text.csv",
        "metrics_image_r.csv",
        "metrics_image_r_text.csv",
        "metrics_image_c_text_c.csv",
        "metrics_image_g.csv",
        "metrics_image_g_text.csv",
        "metrics_segment_g.csv",
        "metrics_segment_g_text.csv",
    ]

    # 打开输出文件
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        header_written = False
        
        # 遍历每个文件并提取表头和数据
        for file_name in files:
            file_path = os.path.join(input_dir, file_name)
            header, second_line = extract_header_and_second_line(file_path)
            
            if header is not None and second_line is not None:
                # 第一次写入表头（添加query_type列）
                if not header_written:
                    full_header = ["query_type"] + header
                    writer.writerow(full_header)
                    header_written = True
                
                # 添加查询类型标识到数据行
                query_type = file_name.replace("metrics_", "").replace(".csv", "")
                row_with_type = [query_type] + second_line
                writer.writerow(row_with_type)

def run_multi_query_evaluation(model_path, config_path, eval_path, results_base_dir, 
                                 eval_split_name="val", 
                                 eval_query_types=None):
    """
    运行多查询类型评估的核心函数
    
    Args:
        model_path: 模型checkpoint路径
        config_path: 配置文件路径
        eval_path: 评估数据路径
        results_base_dir: 结果保存基础目录
        eval_split_name: 评估数据集名称
        eval_query_types: 查询类型列表
    
    Returns:
        timestamped_results_dir: 带时间戳的结果目录路径
    """
    if eval_query_types is None:
        eval_query_types = ["image_r", "image_c", "image_g", "segment_g"]
    
    # 创建带时间戳的结果目录
    timestamped_results_dir = create_timestamped_dir(results_base_dir)
    print(f"创建时间戳目录: {timestamped_results_dir}")
    
    # 循环执行评估
    for eval_query_type in eval_query_types:
        command = f"python training/evaluate.py --config {config_path} " \
                    f"--model_path {model_path} " \
                    f"--eval_split_name {eval_split_name} " \
                    f"--eval_path {eval_path} " \
                    f"--results_dir {timestamped_results_dir} " \
                    f"--eval_query_type {eval_query_type}"
        print(f"Executing: {command}")
        os.system(command)
    
    # 聚合结果
    aggregated_output = os.path.join(timestamped_results_dir, 'aggregated_results.csv')
    aggregate_results(timestamped_results_dir, aggregated_output)
    print(f"聚合结果已保存到: {aggregated_output}")
    
    return timestamped_results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多查询类型评估脚本')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型checkpoint路径')
    parser.add_argument('--config_path', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--eval_path', type=str, required=True,
                        help='评估数据路径')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='结果保存目录')
    parser.add_argument('--eval_split_name', type=str, default='val',
                        help='评估数据集名称')
    parser.add_argument('--eval_query_types', type=str, nargs='+',
                        default=["image_r", "image_c", "image_g", "segment_g"],
                        help='查询类型列表')
    
    args = parser.parse_args()
    
    run_multi_query_evaluation(
        model_path=args.model_path,
        config_path=args.config_path,
        eval_path=args.eval_path,
        results_base_dir=args.results_dir,
        eval_split_name=args.eval_split_name,
        eval_query_types=args.eval_query_types
    )

