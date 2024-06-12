import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import random
import utils
from collections import defaultdict
from transformers import BertForSequenceClassification, BertTokenizer
# from torchsummary import summary
import pickle
from os.path import join
import settings
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=True, type=str2bool, help="whether train the model")
parser.add_argument("--valid", default=True, type=str2bool, help="whether apply model to the valid data")
parser.add_argument("--test", default=True, type=str2bool, help="whether apply the model to the test data")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NCF(nn.Module):
    def __init__(self, dropout, model):
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.model1 = BertForSequenceClassification.from_pretrained(model_name)
        self.model2 = BertForSequenceClassification.from_pretrained(model_name)

        self.embed_user_MLP = self.model1.base_model.embeddings
        self.user_ln_MLP = self.model1.base_model.encoder         # aobo: 这里替换成LLM模型
        self.embed_item_MLP = self.model2.base_model.embeddings         # # aobo: 这里替换成LLM模型
        self.item_ln_MLP = self.model2.base_model.encoder

        MLP_modules = []

        input_size = 768 * 2

        MLP_modules.append(nn.Dropout(p=self.dropout))
        MLP_modules.append(nn.Linear(input_size, input_size))
        MLP_modules.append(nn.ReLU())
        MLP_modules.append(nn.Dropout(p=self.dropout))
        MLP_modules.append(nn.Linear(input_size, input_size//2))
        MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = input_size//2

        self.predict_layer = nn.Linear(predict_size, 1)

    def forward(self, user, item):

        embed_user_MLP = self.embed_user_MLP(user)
        embed_user_MLP = self.user_ln_MLP(embed_user_MLP)       # scibert batch_size=20, 加上encoder层，速度慢了5倍
        embed_item_MLP = self.embed_item_MLP(item)
        embed_item_MLP = self.item_ln_MLP(embed_item_MLP)
        interaction = torch.cat((embed_user_MLP.last_hidden_state, embed_item_MLP.last_hidden_state), -1)

        output_MLP = self.MLP_layers(interaction)       # shape:(batch_size, max_len, 768)

        concat = output_MLP

        cls = concat[:, 0, :]       # shape:(batch_size, 768)
        prediction = self.predict_layer(cls)        # 用token [CLS]的表示进行分类，predictioin shape:(batch_size, 1)

        return prediction.view(-1)


def evaluation(data_name="valid"):
    """
    :param data_name: "valid" or "test"
    :return: nothing but save the submission file
    """
    BATCH_SIZE = 16

    if data_name == "valid":
        test_data = np.load(join(data_dir, "validset_text.npy"), allow_pickle=True)
    else:
        test_data = np.load(join(data_dir, "testset_text.npy"), allow_pickle=True)
    test = []

    score_dict = defaultdict(list)
    ncf.load_state_dict(torch.load(save_path))
    ncf.eval()
    # ncf.cpu()
    for i in tqdm(range(len(test_data))):
        paper_text = tokenizer.encode("[SEP]".join(test_data[i][0]), padding="max_length", truncation=True,
                                      max_length=MAX_LEN)
        ref_text = tokenizer.encode("[SEP]".join(test_data[i][1]), padding="max_length", truncation=True,
                                    max_length=MAX_LEN)
        key = test_data[i][2]
        test.append([paper_text, ref_text, key])  # 训练数据, 文本拼接
    for i in tqdm(range(len(test_data) // BATCH_SIZE + 1)):
        batch_test = test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        batch_test_usr = torch.tensor([x[0] for x in batch_test])
        batch_test_item = torch.tensor([x[1] for x in batch_test])

        batch_test_usr = batch_test_usr.cuda() if torch.cuda.is_available() else batch_test_usr
        batch_test_item = batch_test_item.cuda() if torch.cuda.is_available() else batch_test_item

        prediction = ncf(batch_test_usr, batch_test_item)
        prediction = torch.sigmoid(prediction)

        key_batch = [x[2] for x in batch_test]
        for j in range(len(prediction)):
            score_dict[key_batch[j]].append(float(prediction[j].data.cpu().numpy()))
    utils.dump_json(score_dict, wfdir=join(out_dir, "result"),
                    wfname="{}_submission_ncf_text_lr.json".format(data_name))


def train_func():
    all = []
    global_loss = 1
    for i in tqdm(range(len(train_data))):         # len(train_data)

        paper_text = tokenizer.encode("[SEP]".join(train_data[i][0]), padding="max_length", truncation=True,
                                      max_length=MAX_LEN)
        ref_text = tokenizer.encode("[SEP]".join(train_data[i][1]), padding="max_length", truncation=True,
                                    max_length=MAX_LEN)
        tmp_label = train_data[i][2][0]
        all.append([paper_text, ref_text, tmp_label])     # 训练数据, 文本拼接
    random.shuffle(all)

    train = all[:int(len(all)*0.8)]
    valid = all[int(len(all)*0.8):]

    test_result = []
    loss_ls = []
    # train
    for ep in tqdm(range(EPOCH)):
        ncf.train()
        ncf.cuda() if torch.cuda.is_available() else ncf
        for i in tqdm(range(len(train)//BATCH_SIZE+1)):
            ncf.train()
            ncf.cuda() if torch.cuda.is_available() else ncf
            train_batch = train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            tmp = [x[0] for x in train_batch]
            train_batch_usr = torch.tensor(tmp)         # shape (256,)
            tmp = [x[1] for x in train_batch]
            train_batch_item = torch.tensor(tmp)        # shape (256,)
            tmp = [x[2] for x in train_batch]
            train_batch_label = torch.tensor(tmp)       # shape (256,)

            train_batch_usr = train_batch_usr.cuda() if torch.cuda.is_available() else train_batch_usr
            train_batch_item = train_batch_item.cuda() if torch.cuda.is_available() else train_batch_item
            train_batch_label = train_batch_label.float().cuda() if torch.cuda.is_available() else train_batch_label.float()

            ncf.zero_grad()
            prediction = ncf(train_batch_usr, train_batch_item)  # 前向传播
            loss = loss_function(prediction, train_batch_label)
            loss.backward()
            try:
                optimizer.step()
                loss_ls.append(loss.data.cpu())
            except:
                print("error")
                print("epoch: ", ep)
                print("batch: ", i)

        avg_loss = np.mean(loss_ls)
        print("train avg loss: ", avg_loss)

        valid_BATCH_SIZE = 16
        ncf.eval()
        # ncf.cpu()
        val_loss_ls = []
        for i in tqdm(range(len(valid) // valid_BATCH_SIZE + 1)):
            train_batch = valid[i * valid_BATCH_SIZE: (i + 1) * valid_BATCH_SIZE]
            tmp = [x[0] for x in train_batch]
            train_batch_usr = torch.tensor(tmp)  # shape (256,)
            tmp = [x[1] for x in train_batch]
            train_batch_item = torch.tensor(tmp)  # shape (256,)
            tmp = [x[2] for x in train_batch]
            train_batch_label = torch.tensor(tmp)  # shape (256,)

            train_batch_usr = train_batch_usr.cuda() if torch.cuda.is_available() else train_batch_usr
            train_batch_item = train_batch_item.cuda() if torch.cuda.is_available() else train_batch_item
            train_batch_label = train_batch_label.float().cuda() if torch.cuda.is_available() else train_batch_label.float()

            prediction = ncf(train_batch_usr, train_batch_item)  # 前向传播
            loss = loss_function(prediction, train_batch_label.float())
            val_loss_ls.append(loss.data.cpu())

        val_avg_loss = np.mean(val_loss_ls)
        print("valid avg loss: ", val_avg_loss)

        # save model
        if val_avg_loss < global_loss:
            global_loss = val_avg_loss
            torch.save(ncf.state_dict(), save_path)
        # torch.save(ncf.state_dict(), join(out_dir, "ncf_text_model/torch_model_lr.bin"))
    parameters_num = count_parameters(ncf)
    print("Model parameters: ", parameters_num)
    # 打印当前GPU上分配给Tensor的显存量（以字节为单位）
    print("Memory allocated: {}GB".format(torch.cuda.memory_allocated() / (1024 ** 3)))
    print("Max Memory allocated: {}GB".format(torch.cuda.max_memory_allocated() / (1024 ** 3)))


loss_function = nn.BCEWithLogitsLoss()
BATCH_SIZE = 16
EPOCH = 20
LR = 1e-5
DROP = 0.2
MODEL = "MLP"
MAX_LEN = 512

data_dir = join(settings.DATA_TRACE_DIR, "PST")


if __name__ == "__main__":
    args = parser.parse_args()

    model_name = join(settings.PROJ_DIR, "pretrain_models/bertmodel")
    out_dir = settings.OUT_DIR
    save_path = join(out_dir, "ncf_text_model/torch_model_lr.bin")
    model_dir = os.path.dirname(save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_data = np.load(join(data_dir, "trainset_text.npy"), allow_pickle=True)

    usr_dim = 1          # user feature dimension

    # model
    ncf = NCF(dropout=DROP, model=MODEL)
    optimizer = optim.Adam(ncf.parameters(), lr=LR)
    ncf.cuda() if torch.cuda.is_available() else ncf

    # shuffle
    np.random.seed(1)           # NumPy 是 Pandas 的底层依赖库
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_func()      # training function
    # if args.valid:
    #     evaluation("valid")
    # if args.test:
    #     evaluation("test")

    # 打印模型的参数量
    parameters_num = count_parameters(ncf)
    print(parameters_num)
    # 打印当前GPU上分配给Tensor的显存量（以字节为单位）
    print("Memory allocated: {}GB".format(torch.cuda.memory_allocated()/(1024**3)))
    print("Max Memory allocated: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))


