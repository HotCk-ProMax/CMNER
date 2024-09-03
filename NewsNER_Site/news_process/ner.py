import subprocess
def run(day):
    # 定义命令和参数
    command = [
        "python", "../cmner.py",
        "--do_test",
        f"--txtdir=./data/ner_txt/{day}/",
        f"--imgdir=./data/ner_img/{day}/",
        f"--testoutdir=./data/ner_result/{day}/",
        "--ckpt_path=../v1_model.pt",
    ]

    # 执行命令
    subprocess.run(command, text=True)
