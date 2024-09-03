# CMNER

CNMER (Cross-Domain Multi-Modal Named Entity Recognition) is based on the [UMGF](https://github.com/TransformersWsz/UMGF) model, with improved loss functions and datasets, including improved models and supporting web applications. This project is only used for course practice and learning, not for other purposes.
CMNER contains the sub-project [NewsNER](https://github.com/HotCk-ProMax/NewsNER) for extracting cross-domain multimodal NER datasets from English websites.

Train:
```
python cmner.py --do_train --txtdir=./data/ner_txt --imgdir=./data/ner_img --ckpt_path=./v1_model.pt --num_train_epoch=30 --train_batch_size=16 --lr=0.0001 --seed=2024
```
Website:
Enter in the command line:
```
python -m http.server PORT
```
If you use the conda virtual environment, you may need to start the web page in the corresponding environment.
Start the ```CMNER\NewsNER_Site\templates``` access page in localhost:PORT.