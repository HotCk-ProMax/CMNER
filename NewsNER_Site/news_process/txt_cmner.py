import os
import json
from datetime import datetime
from transformers import BertTokenizer


def run(day):
    # 定义输入数据的根目录和输出目录
    input_root_dir = f'data/classified_data/{day}/'

    output_root_dir = f'data/ner_txt/{day}/'
    # 创建输出根目录
    os.makedirs(output_root_dir, exist_ok=True)

    # 定义要处理的子目录名称
    subdirs = ['business', 'entertainment', 'politics', 'sport', 'tech']

    # 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # 处理每个子目录
    for subdir in subdirs:
        input_dir = os.path.join(input_root_dir, subdir)
        subdir_output_root = os.path.join(output_root_dir, subdir)
        os.makedirs(subdir_output_root, exist_ok=True)

        # 将所有内容放到 test 文件夹中
        output_file_dir = os.path.join(subdir_output_root, 'test')
        os.makedirs(output_file_dir, exist_ok=True)

        file_counter = 0  # 文件序号计数器
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        for json_file in input_files:
            input_file_path = os.path.join(input_dir, json_file)

            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 假设新闻文本在json文件中的字段名为 'content'
            news_text = data.get('text', '')
            current_index = data.get('index')
            # 分词
            tokens = tokenizer.tokenize(news_text)

            # 标签，这里我们假设所有单词的标签都是 'O'（无关标签），你可以根据需要进行修改
            labels = ['O'] * len(tokens)

            # 构建 s_lines 和 l_lines
            s_lines = [f"{token}\t" for token in tokens]
            l_lines = [f"{label}\t" for label in labels]

            # 保存到文件
            s_file_path = os.path.join(output_file_dir, f'{file_counter}_s.txt')
            l_file_path = os.path.join(output_file_dir, f'{file_counter}_l.txt')
            p_file_path = os.path.join(output_file_dir, f'{file_counter}_p.txt')

            with open(s_file_path, 'w', encoding='utf-8') as s_file:
                s_file.write(''.join(s_lines))
            with open(l_file_path, 'w', encoding='utf-8') as l_file:
                l_file.write(''.join(l_lines))
            with open(p_file_path, 'w', encoding='utf-8') as p_file:
                p_file.write(f"{current_index}")

            # 增加文件序号
            file_counter += 1

        print(f"Processed files for {subdir} and saved in {output_file_dir}")

    print("All files processed successfully!")


# run(datetime.today().strftime('%Y-%m-%d'))
