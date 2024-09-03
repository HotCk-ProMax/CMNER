import os
import json
from transformers import pipeline

def run(day):
    classified_dir = f'data/classified_data/{day}/'
    ner_dir = f'data/ner_result/{day}/'
    os.makedirs(ner_dir, exist_ok=True)
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=0
    )

    for category in os.listdir(classified_dir):
        category_path = os.path.join(classified_dir, category)

        if os.path.isdir(category_path):
            ner_category_dir = os.path.join(ner_dir, category)
            os.makedirs(ner_category_dir, exist_ok=True)

            news_files = sorted(os.listdir(category_path),
                                key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

            for news_file in news_files:
                news_path = os.path.join(category_path, news_file)
                with open(news_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)

                text = news_data['text']
                entities = ner_pipeline(text)

                tokens = text.split()

                # 生成BIO标签
                bio_tags = generate_bio_tags(tokens, entities)

                indexed_result = []
                for token, label in zip(tokens, bio_tags):
                    indexed_result.append(f"{token}\t{label}")
                indexed_result.append("")

                # 将结果保存到单独的文件
                result_content = "\n".join(indexed_result)
                output_file_path = os.path.join(ner_category_dir, f"{os.path.splitext(news_file)[0]}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(result_content)

                print(f"Processed file {news_file} and saved NER result to {output_file_path}.")

    print("NER processing completed for all news files.")


# 合并子词并生成BIO标签
def generate_bio_tags(tokens, entities):
    bio_tags = ["O"] * len(tokens)  # 初始化所有标记为 "O"

    token_idx = 0  # 用于跟踪tokens中的位置
    for entity in entities:
        entity_tokens = entity['word'].split()  # 获取实体的子词
        label = entity['entity_group']

        # 找到实体的起始位置
        while token_idx < len(tokens) and not tokens[token_idx].startswith(entity_tokens[0]):
            token_idx += 1

        if token_idx < len(tokens):
            bio_tags[token_idx] = f"B-{label}"  # 实体开头标记为 "B-"
            for i in range(1, len(entity_tokens)):
                if token_idx + i < len(tokens):
                    bio_tags[token_idx + i] = f"I-{label}"  # 实体内部标记为 "I-"
            token_idx += len(entity_tokens)  # 更新索引

    return bio_tags

