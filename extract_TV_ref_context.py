import json
import os
from os.path import join

import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup

import utils
import settings

x_train = []
y_train = []
x_valid = []
y_valid = []

data_dir = join(settings.DATA_TRACE_DIR, "PST")
papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
n_papers = len(papers)
papers = sorted(papers, key=lambda x: x["_id"])
n_train = int(n_papers * 2 / 3)
# n_valid = n_papers - n_train

papers_train = papers[:n_train]
papers_valid = papers[n_train:]
pids_train = {p["_id"] for p in papers_train}
pids_valid = {p["_id"] for p in papers_valid}

in_dir = join(data_dir, "paper-xml")
files = []
for f in os.listdir(in_dir):
    if f.endswith(".xml"):
        files.append(f)

pid_to_source_titles = dd(list)
for paper in tqdm(papers):
    pid = paper["_id"]
    for ref in paper["refs_trace"]:
        pid_to_source_titles[pid].append(ref["title"].lower())


def train_bib_to_contexts_labels():
    bib_to_contexts_labels = dd(dict)
    for cur_pid in tqdm(pids_train | pids_valid):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        flag = False

        cur_pos_bib = set()

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib


        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        bib_to_contexts = utils.find_bib_context(xml)

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10
        # cur_neg_bib_sample = list(cur_neg_bib)          # 这里不采样试一下
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)           # 这里负采样

        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
        elif cur_pid in pids_valid:
            cur_x = x_valid
            cur_y = y_valid
        else:
            continue
            # raise Exception("cur_pid not in train/valid/test")

        for bib in cur_pos_bib:
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(1)
            bib_to_contexts_labels[cur_pid][bib] = (cur_context, 1)

        for bib in cur_neg_bib_sample:
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(0)
            bib_to_contexts_labels[cur_pid][bib] = (cur_context, 0)

    print("len(bib_to_contexts_labels)", len(bib_to_contexts_labels))
    json.dump(bib_to_contexts_labels, open(join(data_dir, "bib_to_contexts_labels.json"), "w", encoding="utf-8"))

    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))       # 15260, 8502； 7645, 4037


def bib_to_contexts(data_name):
    if data_name == "valid":
        papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
        sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")
    else:
        papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
        sub_example_dict = utils.load_json(data_dir,
                                           "submission_example_test.json")

    bib_to_contexts_dic = dd(dict)
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        assert len(sub_example_dict[cur_pid]) == n_refs
        bib_to_contexts = utils.find_bib_context(xml)
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for bib in bib_sorted:
            cur_context = " ".join(bib_to_contexts[bib])
            bib_to_contexts_dic[cur_pid][bib] = (cur_context)

    print("len(bib_to_contexts_labels)", len(bib_to_contexts_dic))
    json.dump(bib_to_contexts_dic, open(join(data_dir, "bib_to_contexts_{}.json".format(data_name)), "w", encoding="utf-8"))


if __name__ == "__main__":
    train_bib_to_contexts_labels()      # train data: train 15260, val 8502; neg sampling:7645, 4037
    bib_to_contexts("valid")            # valid data: 8502
    bib_to_contexts("test")             # test data: 8502

