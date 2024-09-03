from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
from torchcrf import CRF
from PIL import Image
import os
import glob
import time
import random
import argparse
from tqdm import tqdm
import warnings

from model.utils import *
from metric import evaluate_pred_file
from config import tag2idx, idx2tag, max_len, max_node, log_fre

warnings.filterwarnings("ignore")
predict_file = "./output/v1/{}/epoch_{}.txt"
device = torch.device("cuda:0")


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MMNerDataset(Dataset):
    def __init__(self, textdir, imgdir="./data/ner_img"):
        self.X_files = sorted(glob.glob(os.path.join(textdir, "*_s.txt")))
        self.Y_files = sorted(glob.glob(os.path.join(textdir, "*_l.txt")))
        self.P_files = sorted(glob.glob(os.path.join(textdir, "*_p.txt")))
        self._imgdir = imgdir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __len__(self):
        return len(self.X_files)

    def construct_inter_matrix(self, word_num, pic_num=max_node):
        mat = np.zeros((max_len, pic_num), dtype=np.float32)
        mat[:word_num, :pic_num] = 1.0
        return mat

    def __getitem__(self, idx):
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:
            s = fr.readline().split("\t")

        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:
            l = fr.readline().split("\t")

        with open(self.P_files[idx], "r", encoding="utf-8") as fr:
            imgid = fr.readline().strip()
            picpaths = [os.path.join(self._imgdir, "{}/{}_{}.jpg".format(imgid, entity, imgid))
                        for entity in ["crop_person", "crop_miscellaneous", "crop_location", "crop_organization"]]

        ntokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(s, l):  # iterate every word
            tokens = self.tokenizer._tokenize(word)  # one word may be split into several tokens
            ntokens.extend(tokens)
            for i, _ in enumerate(tokens):
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])
        ntokens = ntokens[:max_len - 1]
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len - 1]
        label_ids.append(tag2idx["SEP"])

        matrix = self.construct_inter_matrix(len(label_ids), len(picpaths))

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        rest_pad = [0] * pad_len  # pad to max_len
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)

        # pad ntokens
        ntokens.extend(["pad"] * pad_len)

        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids,
            "picpaths": picpaths,
            "matrix": matrix,
            "file_path": self.X_files[idx]
        }


def collate_fn(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []

    b_ntokens = []
    b_matrix = []
    b_img = torch.zeros(len(batch) * max_node, 3, 224, 224)
    b_filepath = []
    domain_labels = []

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    for idx, example in enumerate(batch):
        b_ntokens.append(example["ntokens"])
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
        b_matrix.append(example["matrix"])
        b_filepath.append(example["file_path"])

        # 提取domain label
        domain_label = get_domain_label(example["file_path"])
        domain_labels.append(domain_label)

        for i, picpath in enumerate(example["picpaths"]):
            try:
                b_img[idx * max_node + i] = preprocess(Image.open(picpath).convert('RGB'))
            except:
                print("========={} error!===============".format(picpath))
                exit(1)

    return {
        "b_ntokens": b_ntokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device)
        },
        "b_img": torch.tensor(b_img).to(device),
        "b_matrix": torch.tensor(b_matrix).to(device),
        "y": torch.tensor(label_ids).to(device),
        # "file_path": torch.tensor(b_filepath).to(device),
        "domain_labels": torch.tensor(domain_labels).to(device)
    }


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_grl
        return output, None


def get_domain_label(filepath):

    domain_to_label = {
        "business": 0,
        "entertainment": 1,
        "politics": 2,
        "sport": 3,
        "tech": 4
    }
    domain_name = filepath.split(os.sep)[-3]

    return domain_to_label.get(domain_name, -1)


class MMNerModel(nn.Module):

    def __init__(self, d_model=512, d_hidden=256, n_heads=8, dropout=0.4, layer=6, tag2idx=tag2idx, num_domains=5):
        super(MMNerModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.resnet = models.resnet152(pretrained=True)
        self.crf = CRF(len(tag2idx), batch_first=True)
        # self.hidden2tag = nn.Linear(2*d_model, len(tag2idx))
        self.hidden2tag = nn.Linear(768 + 512, len(tag2idx))

        objcnndim = 2048
        fc_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features=fc_feats, out_features=objcnndim, bias=True)

        self.layer = layer
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden

        self.trans_txt = nn.Linear(768, d_model)
        self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

        # text
        self.mhatt_x = clone(MultiHeadedAttention(
            n_heads, d_model, dropout), layer)
        self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), layer)

        # img
        self.mhatt_o = clone(MultiHeadedAttention(
            n_heads, d_model, dropout, v=0, output=0), layer)
        self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), layer)

        self.mhatt_x2o = clone(Linear(d_model * 2, d_model), layer)
        self.mhatt_o2x = clone(Linear(d_model * 2, d_model), layer)
        self.xgate = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ogate = clone(SublayerConnectionv2(d_model, dropout), layer)
        # 增加领域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_domains),
            nn.LogSoftmax(dim=1)
        )

    def log_likelihood(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()  # reserve origin bert output (9, 48, 768)
        x = self.trans_txt(x)  # 9, 48, 512

        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            # 9, 48, 512
            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))
        # print("Tag to index mapping:", tag2idx)

        x = torch.cat((bert_x, x), dim=2)
        logits = self.hidden2tag(x)  # 将输出转换为logits

        # Define class weights for Cross-Entropy
        class_weights = torch.tensor([
                                        0.1,  # PAD
                                        1.0,  # B-PER
                                        1.0,  # I-PER
                                        1.0,  # B-LOC
                                        1.0,  # I-LOC
                                        2.0,  # B-ORG
                                        2.0,  # I-ORG
                                        3.0,  # B-OTHER
                                        3.0,  # I-OTHER
                                        1.0,  # O
                                        0.0,  # X
                                        0.1,  # CLS
                                        0.1   # SEP
                                    ], device=device)

        loss_fct = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

        # Calculate weighted Cross-Entropy loss
        active_loss = crf_mask.view(-1) == 1  # 获取有效的标签位置
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_tags = tags.view(-1)[active_loss]
        ce_loss = loss_fct(active_logits, active_tags).mean()

        # Calculate CRF loss
        crf_loss = -self.crf(logits, tags, mask=crf_mask, reduction='mean')

        # Combine losses
        total_loss = ce_loss + crf_loss
        return total_loss

    def forward(self, x, b_img=None, inter_matrix=None, text_mask=None, tags=None, domain_labels=None, lambda_grl=1.0):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()
        x = self.trans_txt(x)

        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))

        # Concatenate
        x = torch.cat((bert_x, x), dim=2)
        # fully connected layer
        logits = self.hidden2tag(x)

        # CRF decode
        decoded_tags = self.crf.decode(logits, mask=crf_mask)

        if domain_labels is not None:
            # GRL layer
            reverse_x = ReverseLayerF.apply(bert_x, lambda_grl)
            domain_preds = self.domain_classifier(reverse_x[:, 0, :])
            return decoded_tags, domain_preds
        else:
            return decoded_tags


def save_model(model, model_path="./model.pt"):
    torch.save(model.state_dict(), model_path)
    print("Current Best mmner model has beed saved!")


def predict(epoch, model, dataloader, mode="val", res=None):
    model.eval()
    with torch.no_grad():
        filepath = predict_file.format(mode, epoch)
        os.makedirs(filepath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fw:
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx[
                            "CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            true_tag = idx2tag[ground_idx.data.item()]
                            line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                            fw.write(line)
        print("=============={} -> {} epoch eval done=================".format(mode, epoch))
        cur_f1 = evaluate_pred_file(filepath)
        to_save = False
        if mode == "val":
            if res["best_f1"] < cur_f1:
                res["best_f1"] = cur_f1
                res["epoch"] = epoch
                to_save = True
            print("current best f1: {}, epoch: {}".format(res["best_f1"], res["epoch"]))
        return to_save


class_weights = torch.tensor([1.0, 1.0, 2.0, 3.0], device=device)  # 假设ORG和OTHER类别较少，给予更高的权重
loss_fct = nn.CrossEntropyLoss(weight=class_weights)

def train(args):
    seed_torch(args.seed)

    # 遍历所有领域目录
    domain_dirs = [os.path.join(args.txtdir, d) for d in os.listdir(args.txtdir) if
                   os.path.isdir(os.path.join(args.txtdir, d))]

    if not domain_dirs:
        raise ValueError(f"No domain subdirectories found in the specified directory: {args.txtdir}")

    domain_dirs = [d for d in domain_dirs if os.path.basename(d) != os.path.basename(args.txtdir)]

    if not domain_dirs:
        raise ValueError(f"No valid domain directories found in the specified path: {args.txtdir}")

    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    for domain_dir in domain_dirs:
        train_textdir = os.path.join(domain_dir, "train")
        val_textdir = os.path.join(domain_dir, "valid")
        test_textdir = os.path.join(domain_dir, "test")

        all_train_datasets.append(MMNerDataset(textdir=train_textdir, imgdir=args.imgdir))
        all_val_datasets.append(MMNerDataset(textdir=val_textdir, imgdir=args.imgdir))
        all_test_datasets.append(MMNerDataset(textdir=test_textdir, imgdir=args.imgdir))

    train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(all_val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(all_test_datasets)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=False)

    model = MMNerModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    res = {"best_f1": 0.0, "epoch": -1}
    start = time.time()

    for epoch in range(args.num_train_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch["x"]
            b_img = batch["b_img"]
            inter_matrix = batch["b_matrix"]
            text_mask = x["attention_mask"]
            y = batch["y"]
            domain_labels = batch["domain_labels"]

            decoded_tags, domain_preds = model(x, b_img, inter_matrix, text_mask, y, domain_labels=domain_labels)

            # 序列标注损失
            loss_seq = model.log_likelihood(x, b_img, inter_matrix, text_mask, y)
            # 领域对抗损失
            loss_domain = criterion(domain_preds, domain_labels)

            # 总损失
            loss = loss_seq + args.lambda_grl * loss_domain
            loss.backward()
            optimizer.step()

            if i % log_fre == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch, i, loss.data))

        scheduler.step()
        to_save = predict(epoch, model, val_dataloader, mode="val", res=res)
        predict(epoch, model, test_dataloader, mode="test", res=res)
        if to_save:
            save_model(model, args.ckpt_path)

    print("================== train done! ================")
    end = time.time()
    hour = int((end - start) // 3600)
    minute = int((end - start) % 3600 // 60)
    print("total time: {} hour - {} minute".format(hour, minute))

def test(args):
    model = MMNerModel().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cuda:0'))
    print("Absolute path of the input directory:", os.path.abspath(args.txtdir))

    # 获取所有域文件夹
    domain_dirs = sorted([d for d in os.listdir(args.txtdir) if os.path.isdir(os.path.join(args.txtdir, d))])
    output_root = args.testoutdir
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for domain_dir in domain_dirs:
        domain_path = os.path.join(args.txtdir, domain_dir)
        test_dir = os.path.join(domain_path, 'test')
        if not os.path.exists(test_dir):
            print(f"Test directory not found for domain: {domain_dir}")
            continue

        # 在输出文件夹中创建对应的域文件夹
        domain_output_dir = os.path.join(output_root, domain_dir)
        if not os.path.exists(domain_output_dir):
            os.makedirs(domain_output_dir)

        test_dataset = MMNerDataset(textdir=test_dir, imgdir=args.imgdir)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn)

        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Testing {domain_dir}"):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)

                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]

                    # 获取当前样本对应的 _p.txt 文件的路径
                    p_file_path = os.path.join(test_dir, f"{idx}_p.txt")
                    with open(p_file_path, 'r', encoding='utf-8') as p_file:
                        index = p_file.readline().strip()  # 读取 _p.txt 文件中的 index 值

                    # 定义输出文件路径
                    output_file_path = os.path.join(domain_output_dir, f"{index}.txt")

                    # 将结果写入文件
                    with open(output_file_path, "w", encoding="utf-8") as fw:
                        for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                            if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx[
                                "CLS"] or ground_idx == tag2idx["SEP"]:
                                continue
                            else:
                                predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                    "PAD", "X", "CLS", "SEP"] else "O"
                                true_tag = idx2tag[ground_idx.data.item()]
                                line = "{}\t{}\t{}\n".format(b_ntokens[idx][pos], predict_tag, true_tag)
                                fw.write(line)

    print("Testing complete. Outputs saved in:", os.path.abspath(output_root))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run testing.")
    parser.add_argument('--txtdir',
                        type=str,
                        default="./my_data/twitter2015/",
                        help="Path to the parent directory containing all domain subdirectories.")
    parser.add_argument('--imgdir',
                        type=str,
                        default="./data/twitter2015/image/",
                        help="Path to the image directory.")
    parser.add_argument('--ckpt_path',
                        type=str,
                        default="./model.pt",
                        help="Path to save or load the model.")
    parser.add_argument("--num_train_epoch",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for testing.")
    parser.add_argument("--lr",
                        default=0.0001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=2021,
                        help="Random seed for initialization.")
    parser.add_argument("--lambda_grl",
                        default=0.001,
                        type=float,
                        help="The trade-off parameter for domain adversarial loss.")
    parser.add_argument("--testoutdir",
                        default="./data/ner_result/",
                        type=str,
                        help="The trade-off parameter for domain adversarial loss.")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_test` must be True.')

# python cmner.py --do_train --txtdir=./data/ner_txt --imgdir=./data/ner_img --ckpt_path=./v1_model.pt --num_train_epoch=30 --train_batch_size=16 --lr=0.0001 --seed=2024
if __name__ == "__main__":
    main()

