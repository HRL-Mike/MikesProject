
import os
import sys
import torch
import argparse
import torch.utils.data
import torch.nn.functional as F

from torch import nn
from utils import save_clf_checkpoint, adjust_learning_rate, calc_acc, calc_precision_recall_fscore, calc_classwise_acc
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def seed_everything(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')

    # VB Model parameters
    # parser.add_argument('--emb_dim',        type=int,   default=300,  help='dimension of word embeddings.')
    # parser.add_argument('--n_heads',        type=int,   default=8,    help='Multi-head attention.')
    # parser.add_argument('--dropout',        type=float, default=0.1,  help='dropout')
    # parser.add_argument('--encoder_layers', type=int,   default=6,    help='the number of layers of encoder in Transformer.')
    # VB is not used in this case

    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=2,    help='number of epochs to train for (if early stopping is not triggered).')  # 80, 26
    parser.add_argument('--batch_size',     type=int,   default=20,   help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,    help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--print_freq',     type=int,   default=100,  help='print training/validation stats every __ batches.')

    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,             help='path to checkpoint, None if none.')

    parser.add_argument('--lr',             type=float, default=0.00001,  help='0.000005, 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default='checkpoints/CLIPGPT2/m18_v1_z_qf_',  help='med_vqa_c$version$/m18/c80/m18_vid$temporal_size$/c80_vid$temporal_size$') #clf_v1_2_1x1/med_vqa_c3
    parser.add_argument('--dataset_type',   default='m18',            help='med_vqa/m18/c80/m18_vid/c80_vid')
    parser.add_argument('--dataset_cat',    default='cat1',           help='cat1/cat2/cat3')
    parser.add_argument('--tokenizer_ver',  default='gpt2v1',         help='btv2/btv3/gpt2v1')
    parser.add_argument('--question_len',   default=25,               help='25')
    parser.add_argument('--model_ver',      default='CLIPGPT2',       help='vb/vbrm/efvlegpt2rs18/efvlegpt2Swin/"')  # vrvb/gpt2rs18/gpt2ViT/gpt2Swin/biogpt2rs18/vilgpt2vqa/efgpt2rs18gr/efvlegpt2Swingr
    parser.add_argument('--model_subver',   default='v1',             help='V0,v1/v2/v3/v4')
    parser.add_argument('--vis_pos_emb',    default='zeroes',         help='None, zeroes, pos')  # 这个有用
    parser.add_argument('--patch_size',     default=5,                help='1/2/3/4/5')
    # 留意一下, clip-vit-base-patch32

    parser.add_argument('--num_class',      default=18,               help='25')
    parser.add_argument('--validate',       default=False,            help='When only validation required False/True')

    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):

    model.train()
    total_loss = 0.0
    label_true = None
    label_pred = None
    label_score = None

    for i, (_, v_f, q, labels) in enumerate(train_dataloader, 0):  # visual features, questions and labels (answers)
        questions = []
        for question in q:
            questions.append(question)

        inputs = tokenizer(questions, padding="max_length", max_length=args.question_len,
                           return_tensors="pt")

        # Visual features
        visual_features = v_f
        visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'], 1)

        # labels
        labels = labels.to(device)
        outputs = model(inputs.to(device), visual_features.to(device))
        loss = criterion(outputs, labels)  # calculate loss
        optimizer.zero_grad()
        loss.backward()  # calculate gradient
        optimizer.step()  # update parameters

        # print statistics
        total_loss += loss.item()

        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        if label_true is None:  # accumulate true labels of the entire training set
            label_true = labels.data.cpu()
        else:
            label_true = torch.cat((label_true, labels.data.cpu()), 0)
        if label_pred is None:  # # accumulate pred labels of the entire training set
            label_pred = predicted.data.cpu()
        else:
            label_pred = torch.cat((label_pred, predicted.data.cpu()), 0)
        if label_score is None:
            label_score = scores.data.cpu()
        else:
            label_score = torch.cat((label_score, scores.data.cpu()), 0)

    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %
          (epoch, total_loss, acc, precision, recall, f_score))
    return acc


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output=False):

    model.eval()
    total_loss = 0.0
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()

    with torch.no_grad():
        for i, (file_name, v_f, q, labels) in enumerate(val_loader, 0):
            # prepare questions
            questions = []
            for question in q:
                questions.append(question)

            inputs = tokenizer(questions, padding="max_length", max_length=args.question_len, return_tensors="pt")

            # Visual features
            visual_features = v_f
            visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'], 1)

            # label
            labels = labels.to(device)

            # model forward pass
            outputs = model(inputs.to(device), visual_features.to(device))

            # loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            label_true = labels.data.cpu() if label_true is None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred is None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score is None else torch.cat((label_score, scores.data.cpu()), 0)
            for f in file_name:
                file_names.append(f)  # not used

    acc = calc_acc(label_true, label_pred)
    c_acc = 0.0
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %
          (epoch, total_loss, acc, precision, recall, f_score))

    return acc, c_acc, precision, recall, f_score


if __name__ == '__main__':

    args = get_arg()  # 只用默认值可以吗
    os.makedirs('checkpoints/CLIPGPT2', exist_ok=True)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0
    # final_args = {"emb_dim": args.emb_dim, "n_heads": args.n_heads,
    #               "dropout": args.dropout, "encoder_layers": args.encoder_layers}  # for VB model, no use in our case

    if args.dataset_type == 'm18':  # m18 = EndoVis18
        # tokenizer
        if args.tokenizer_ver == 'gpt2v1':  # should we use CLIP tokenizer?
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]

        folder_head = 'D:/1-硕士研究项目/1-数据集/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Classification/*.txt'

        # 差一个dataloader
        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            train_dataset = EndoVis18VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = EndoVis18VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 18

        if args.model_ver == 'efvlegpt2Swin':
            model = EFVLEGPT2SwinClassification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in range(start_epoch, args.epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)

            # train
            train_acc = train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

            # validation
            test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)

            if test_acc >= best_results[0]:
                print('Best Epoch:', epoch)
                epochs_since_improvement = 0
                best_results[0] = test_acc
                best_epoch[0] = epoch
                save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, best_results[0], final_args)

