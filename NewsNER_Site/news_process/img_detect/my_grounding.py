import matplotlib as mpl
mpl.use('Agg')
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
from torchvision.transforms import Compose, ToTensor, Normalize
from model.grounding_model import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.word_utils import Corpus
from utils.transforms import *
from check import check


def main(day):
    ########### path settings #########
    corpus_pth = 'news_process/img_detect/data/referit/corpus.pth'
    resume_path = 'news_process/img_detect/saved_models/lstm_referit_model.pth.tar'
    test_img_root = f'data/pictures/{day}/'
    output_root = f'data/ner_img/{day}/'
    valid_img_ext = ['jpg', 'png']

    ########### running settings ##############
    gpu_ids = '0'
    random_seed = 13
    dataset_name = 'refeit'
    lstm = True # False means bert
    light = False
    emb_size = 512
    imsize = 256
    anchor_imsize = 416
    query_word_list = ['location', 'organization', 'person', 'miscellaneous']
    ###########################################

    global anchors_full
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(random_seed)
    np.random.seed(random_seed + 1)
    torch.manual_seed(random_seed + 2)
    torch.cuda.manual_seed_all(random_seed + 3)

    ## following anchor sizes calculated by kmeans under args.anchor_imsize=416
    if dataset_name == 'refeit':
        anchors = '30,36,  78,46,  48,86,  149,79,  82,148,  331,93,  156,207,  381,163,  329,285'
    elif dataset_name == 'flickr':
        anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    else:
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


    if lstm:
        corpus = Corpus()
        corpus = torch.load(corpus_pth)
    else:
        corpus = None

    model = grounding_model(corpus=corpus, light=light, emb_size=emb_size, coordmap=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if resume_path:
        if os.path.isfile(resume_path):
            print(("=> loading checkpoint '{}'".format(resume_path)))
            logging.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    visu_param = list(model.module.visumodel.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    for folder, dirs, fnames in os.walk(test_img_root):
        if len(fnames) > 0:
            for fname in fnames:
                if fname.split('.')[-1] not in valid_img_ext:
                    continue
                print('grounding', fname)
                ipath = os.path.join(folder, fname)
                save_path = os.path.join(output_root, fname.split('.')[0])
                for word in query_word_list:
                    try:
                        img, word_id, word_mask, ratio, dw, dh = get_item(ipath, word, corpus, transform=input_transform)
                        imgs = img.cuda()
                        word_id = torch.from_numpy(word_id).cuda()
                        word_mask = torch.from_numpy(word_mask).cuda()
                        dw = torch.from_numpy(dw)
                        dh = torch.from_numpy(dh)
                        ratio = torch.from_numpy(ratio)
                        imgs = torch.unsqueeze(imgs, 0)
                        word_id = torch.unsqueeze(word_id, 0)
                        word_mask = torch.unsqueeze(word_mask, 0)
                        image = Variable(imgs)
                        word_id = Variable(word_id)
                        word_mask = Variable(word_mask)

                        with torch.no_grad():
                            ## Note LSTM does not use word_mask
                            pred_anchor = model(image, word_id, word_mask)
                        for ii in range(len(pred_anchor)):
                            pred_anchor[ii] = pred_anchor[ii].view(pred_anchor[ii].size(0), 3, 5, pred_anchor[ii].size(2),
                                                                   pred_anchor[ii].size(3))
                        # gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor)

                        ## test: convert center+offset to box prediction
                        pred_conf_list = []
                        for ii in range(len(pred_anchor)):
                            pred_conf_list.append(pred_anchor[ii][:, :, 4, :, :].contiguous().view(1, -1))

                        pred_conf = torch.cat(pred_conf_list, dim=1)
                        max_conf, max_loc = torch.max(pred_conf, dim=1)

                        pred_bbox = torch.zeros(1, 4)

                        pred_gi, pred_gj, pred_best_n = [], [], []
                        for ii in range(1):
                            if max_loc[ii] < 3 * (imsize // 32) ** 2:
                                best_scale = 0
                            elif max_loc[ii] < 3 * (imsize // 32) ** 2 + 3 * (imsize // 16) ** 2:
                                best_scale = 1
                            else:
                                best_scale = 2

                            grid, grid_size = imsize // (32 // (2 ** best_scale)), 32 // (2 ** best_scale)
                            anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
                            anchors = [anchors_full[i] for i in anchor_idxs]
                            scaled_anchors = [(x[0] / (anchor_imsize / grid),
                                               x[1] / (anchor_imsize / grid)) for x in anchors]

                            pred_conf = pred_conf_list[best_scale].view(1, 3, grid, grid).data.cpu().numpy()
                            max_conf_ii = max_conf.data.cpu().numpy()

                            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
                            (best_n, gj, gi) = np.where(pred_conf[ii, :, :, :] == max_conf_ii[ii])
                            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
                            pred_gi.append(gi)
                            pred_gj.append(gj)
                            pred_best_n.append(best_n + best_scale * 3)

                            pred_bbox[ii, 0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
                            pred_bbox[ii, 1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
                            pred_bbox[ii, 2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * \
                                               scaled_anchors[best_n][0]
                            pred_bbox[ii, 3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * \
                                               scaled_anchors[best_n][1]
                            pred_bbox[ii, :] = pred_bbox[ii, :] * grid_size
                        pred_bbox = xywh2xyxy(pred_bbox)

                        # pred_bbox[:, 0], pred_bbox[:, 2] = (pred_bbox[:, 0] - dw[0]) / ratio[0], (pred_bbox[:, 2] - dw[0]) / ratio[0]
                        # pred_bbox[:, 1], pred_bbox[:, 3] = (pred_bbox[:, 1] - dh[0]) / ratio[0], (pred_bbox[:, 3] - dh[0]) / ratio[0]

                        pred_bbox[:, 0], pred_bbox[:, 2] = (pred_bbox[:, 0] - dw) / ratio, (pred_bbox[:, 2] - dw) / ratio
                        pred_bbox[:, 1], pred_bbox[:, 3] = (pred_bbox[:, 1] - dh) / ratio, (pred_bbox[:, 3] - dh) / ratio

                        ## convert pred, gt box to original scale with meta-info
                        top, bottom = round(float(dh[0]) - 0.1), imsize - round(float(dh[0]) + 0.1)
                        left, right = round(float(dw[0]) - 0.1), imsize - round(float(dw[0]) + 0.1)
                        img_np = imgs[0, :, top:bottom, left:right].data.cpu().numpy().transpose(1, 2, 0)

                        ratio = float(ratio)
                        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
                        ## also revert image for visualization
                        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
                        img_np = Variable(torch.from_numpy(img_np.transpose(2, 0, 1)).cuda().unsqueeze(0))

                        pred_bbox[:, :2], pred_bbox[:, 2], pred_bbox[:, 3] = \
                            torch.clamp(pred_bbox[:, :2], min=0), torch.clamp(pred_bbox[:, 2],
                                                                              max=img_np.shape[3]), torch.clamp(
                                pred_bbox[:, 3], max=img_np.shape[2])

                        n = img_np.shape[0]

                        input = img_np.data.cpu().numpy()
                        input = input.transpose(0, 2, 3, 1)
                        for ii in range(n):
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            imgs = input[ii, :, :, :].copy()
                            imgs = (imgs * np.array([0.299, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.
                            # imgs = imgs.transpose(2,0,1)
                            imgs = np.array(imgs, dtype=np.float32)
                            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
                            crop = imgs[int(pred_bbox[ii, 1]): int(pred_bbox[ii, 3]), int(pred_bbox[ii, 0]): int(pred_bbox[ii, 2])]
                            if crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0:
                                cv2.imwrite(save_path + '/crop_' + word + '_' + fname, crop)
                            else:
                                print('crop error for', fname, word)
                    except:
                        print('ground error for', fname)
    check(test_img_root)


def get_item(img_path, phrase, corpus, imsize=256, transform=None, lstm=True):
    img = cv2.imread(img_path)
    ## duplicate channel if gray image
    if img.shape[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.stack([img] * 3)

    phrase = phrase.lower()
    img, _, ratio, dw, dh = letterbox(img, None, imsize)
    if transform is not None:
        img = transform(np.array(img))
    if lstm:
        word_id = corpus.tokenize(phrase, 128)
        word_mask = np.zeros(word_id.shape)
    else:
        ## encode phrase to bert input
        pass

    return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
           np.array([ratio], dtype=np.float32), np.array([dw], dtype=np.float32), np.array([dh], dtype=np.float32)


if __name__ == "__main__":
    import sys
    day = sys.argv[1]
    main(day)
    print("Detect done")
