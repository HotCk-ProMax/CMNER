import os
import json
import joblib
from textblob import Word
import re
import glob

# 加载模型和向量化器
model_filename = 'news_process/news_classifier.pkl'
vect_filename = 'news_process/vectorizer.pkl'

# 定义文本预处理函数
def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess_text(text):
    return ' '.join([Word(word).lemmatize() for word in clean_str(text).split()])

def run(day):
    # 定义分类目录
    base_dir = f'data/classified_data/{day}/'
    model = joblib.load(model_filename)
    vect = joblib.load(vect_filename)
    os.makedirs(base_dir, exist_ok=True)
    # 读取 ./news_data 目录下的所有 *_articles.json 文件
    json_files = glob.glob(f'data/news_data/{day}/*_articles.json')

    # 逐个处理 JSON 文件
    for file in json_files:
        with open(file, 'r') as f:
            news_articles = json.load(f)

        # 逐条处理新闻并分类
        for article in news_articles:
            text = preprocess_text(article['text'])
            article_vector = vect.transform([text])
            predicted_category = model.predict(article_vector)[0]

            # 定义分类目录
            category_dir = os.path.join(base_dir, predicted_category)
            os.makedirs(category_dir, exist_ok=True)

            # 保存文章到对应的分类目录中
            article_filename = f"{article['index']}.json"
            with open(os.path.join(category_dir, article_filename), 'w') as outfile:
                json.dump(article, outfile, indent=4)

            print(f"Article {article['index']} classified as {predicted_category} and saved to {category_dir}")

    print("Classification and saving completed.")