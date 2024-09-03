import newspaper
import json
import os
import requests

# 定义要爬取的新闻页面URL及每个页面要爬取的新闻数量
url_dic = {
    # 'https://www.cbsnews.com/': 350,
    'https://www.bbc.com/': 110,
    # 'https://www.nytimes.com/': 100,
    # 'https://edition.cnn.com/': 100,
    # 'https://www.washingtonpost.com/': 200,
    # 'https://abcnews.go.com/': 100,
    # 'https://www.npr.org/': 100
}

# 获取已保存文件的最大索引值
def get_max_global_index(save_dir):
    max_index = 0
    for filename in os.listdir(save_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'r') as f:
                articles = json.load(f)
                if articles:
                    max_index = max(max_index, max(article['index'] for article in articles))
    return max_index

# 定义过滤条件
def is_relevant_article(article, today):
    min_length = 100  # 文章最短字数
    ignored_titles = ["Latest News & Updates", "Breaking News", "US election 2024",
                      "US & Canada", "Northern Ireland Politics", "BBC Live & Breaking World and U.S. News",
                      "CNN Business", "Travel News", "Breaking News, Latest News and Videos", "ABC News"]  # 忽略的标题
    if len(article.title) < 10 or article.title in ignored_titles:
        return False
    if len(article.text) < min_length:
        return False
    if article.publish_date and article.publish_date.date() != today:
        return False
    return True

# 计算Jaccard相似度
def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# 检查标题是否重复
def is_duplicate_title(new_title, existing_titles, threshold=0.8):
    for title in existing_titles:
        if jaccard_similarity(new_title, title) > threshold:
            return True
    return False

# 保存数据到文件
def save_articles_to_file(file_path, articles_data):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_articles = json.load(f)
        articles_data.extend(existing_articles)

    with open(file_path, 'w') as f:
        json.dump(articles_data, f, indent=4)
    print(f"Saved {len(articles_data)} articles to {file_path}")

def run(day):
    save_dir = f'./data/news_data/{day}/'
    picture_dir = f'./data/pictures/{day}/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)
    # 初始化全局索引变量
    global_index = get_max_global_index(save_dir) + 1
    for url, articles_to_save in url_dic.items():
        cbs_paper = newspaper.build(url, language='en', memoize_articles=False)

        saved_articles = 0
        articles_data = []
        existing_titles = []
        fail_count = 0  # 初始化失败计数器

        # 获取域名并构建文件路径
        domain = url.split('/')[-2]  # 从URL中提取域名
        file_path = os.path.join(save_dir, f'{domain}_articles.json')

        # 如果文件存在，读取已有数据
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_articles = json.load(f)
                existing_titles.extend([article['title'] for article in existing_articles])

        for article in cbs_paper.articles:
            if saved_articles >= articles_to_save:
                break
            try:
                article.download()
                article.parse()
                article.nlp()

                # 过滤不相关或重复的文章
                if not is_relevant_article(article, day):
                    continue

                # 检查标题是否重复
                if is_duplicate_title(article.title, existing_titles):
                    print(f"Skipped duplicate title: {article.title}")
                    continue

                # 检查是否有文本和图片
                if article.text and article.top_image:
                    # 下载图片
                    image_url = article.top_image
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image_filename = f"{global_index}.jpg"  # 使用索引作为图片文件名
                        image_path = os.path.join(picture_dir, image_filename)

                        # 保存图片
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_response.content)

                        article_info = {
                            'index': global_index,  # 添加索引字段
                            'title': article.title,
                            'text': article.text,
                            'url': article.url,
                            'image': image_filename,  # 保存图片文件名
                            'publish_date': str(article.publish_date) if article.publish_date else None
                        }
                        articles_data.append(article_info)
                        existing_titles.append(article.title)  # 将标题加入已存在的标题列表
                        saved_articles += 1
                        global_index += 1
                        fail_count = 0  # 成功后重置失败计数器
                        print(f"Saved article {saved_articles}: {article.title} with image {image_filename}")

                # 每50条保存一次数据
                if saved_articles % 50 == 0:
                    save_articles_to_file(file_path, articles_data)
                    articles_data = []  # 清空临时数据，避免重复保存

            except Exception as e:
                fail_count += 1
                print(f"Error processing article {article.url}: {e}")
                if fail_count >= 20:
                    print(f"Skipping {url} after {fail_count} consecutive failures.")
                    break

        # 保存剩余数据
        if articles_data:
            save_articles_to_file(file_path, articles_data)

    print("Completed!")
