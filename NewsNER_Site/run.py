from news_process.news_spider import run as spr
from news_process.classify import run as clsr
from news_process.txt_cmner import run as txtr
from news_process.ner import run as nerr
from datetime import datetime
import subprocess

def main():
    today = datetime.today().strftime('%Y-%m-%d')
    print("Current date: ", today)
    spr(today)
    clsr(today)
    txtr(today)
    imgr_path = 'news_process/img_detect/my_grounding.py'
    subprocess.run(['python', imgr_path, today])
    nerr(today)


if __name__ == "__main__":
    main()
