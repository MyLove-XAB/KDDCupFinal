import json
from os.path import abspath, dirname, join
from collections import defaultdict as dd
from tqdm import tqdm
import numpy as np
import settings
import utils

print("------------------------------------bingyu data load ------------------------------------")

data_dir = join(settings.DATA_TRACE_DIR, "PST")


with open(join(data_dir, "TVT_paper_info_from_xml.json"), 'r', encoding="utf-8") as file:
    xml_info = json.load(file)
print("xml info length: ", len(xml_info))
# print("xml info keys: ", xml_info["5db80dc83a55acd5c14a24b9"].keys())


with open(join(data_dir, "Train_Valid_Test_paper_more_info_from_dblp.json"), 'r', encoding="utf-8") as file:
    paper_dblp_info = json.load(file)

print("paper dblp info length: ", len(paper_dblp_info))
# print("paper dblp info keys: ", paper_dblp_info["61dbf1dcd18a2b6e00d9f311"].keys())


# prepare data for NCF model

# load train and valid data
with open(join(data_dir, "paper_source_trace_train_ans.json"), 'r', encoding="utf-8") as file:
    train_data = json.load(file)

with open(join(data_dir, "Valid_papers_refs.json"), 'r', encoding="utf-8") as file:
    valid_data = json.load(file)


def get_paper_info(pid, num_ref=True, paper_title=True, abstract=True, content=True,
                   keywords=True, authors=True, n_citation=True, venue=True):
    """
    extract paper info from xml file and dblp file
    :param pid: paper id
    :param num_ref: number of references
    :param paper_title: title of the paper
    :param abstract: abstract of the paper
    :param content: body content of the paper
    :param keywords: keywords of the paper
    :param authors: authors of the paper
    :param n_citation: number of the citation
    :param venue: venue of the paper
    :return: a list of paper info
    """
    info_ls = list()
    if num_ref:
        try:
            info_ls.append(xml_info[pid]["num_ref"])
        except:
            info_ls.append(0)
    if paper_title:
        info_ls.append(xml_info[pid]["paper_title"])
    if abstract:
        info_ls.append(xml_info[pid]["abstract"])
    if content:
        info_ls.append(xml_info[pid]["content"])
    if keywords:
        info_ls.append(paper_dblp_info[pid]["keywords"])
    if authors:
        info_ls.append(paper_dblp_info[pid]["authors"])
    if n_citation:
        info_ls.append(paper_dblp_info[pid]["n_citation"])
    if venue:
        info_ls.append(paper_dblp_info[pid]["venue"])
    return info_ls


# def paper2id(data):
#     paperset = set()
#     for i in tqdm(data):
#         paperset.add(i[0])
#         paperset.add(i[1])
#     with open(join(data_dir, "paper2id.txt"), 'w', encoding="utf-8") as file:
#         id = 1
#         for paper in paperset:
#             file.write(paper)
#             file.write("\t")
#             file.write(str(id))
#             id += 1
#             file.write("\n")


def main2():
    """构建包含文本信息的训练数据"""
    bib_context_labels = json.load(
        open(join(data_dir, "bib_to_contexts_labels.json"), 'r', encoding="utf-8"))
    strID2title = json.load(
        open(join(data_dir, "strID_to_title.json"), 'r', encoding="utf-8"))
    paper_keys = bib_context_labels.keys()
    print("len papers: ", len(paper_keys))
    train_tmp = []
    for key in tqdm(paper_keys):
        for bib, label in bib_context_labels[key].items():
            # ([paper title, paper abstract], [ref title, ref context], [label])
            abs_info = xml_info[key]["abstract"] if key in xml_info.keys() else ""
            kw1_ls = paper_dblp_info[key]["keywords"] if key in paper_dblp_info.keys() else ["[PAD]"]
            kw1 = ";".join(kw1_ls)
            body = xml_info[key]["content"] if key in xml_info.keys() else "[PAD]"
            kw2_ls = paper_dblp_info[bib]["keywords"] if bib in paper_dblp_info.keys() else ["[PAD]"]
            kw2 = ";".join(kw2_ls)
            train_tmp.append(([strID2title.get(key, ""), abs_info, kw1, body[:1000]], [strID2title.get(bib, bib), label[0], kw2], [label[1]]))
    print("training set number:", len(train_tmp))  # 23762
    np.save(join(data_dir, "trainset_text.npy"), train_tmp)


def main3(data_name):
    """包含文本信息的验证集和测试集"""
    if data_name == "valid":
        bib_context = json.load(
            open(join(data_dir, "bib_to_contexts_valid.json"), 'r', encoding="utf-8"))
    else:
        bib_context = json.load(
            open(join(data_dir, "bib_to_contexts_test.json"), 'r', encoding="utf-8"))
    strID2title = json.load(
            open(join(data_dir, "strID_to_title.json"), 'r', encoding="utf-8"))

    paper_keys = bib_context.keys()
    print("len papers: ", len(paper_keys))
    tmp_ls = []
    for key in tqdm(paper_keys):
        for bib, label in bib_context[key].items():
            # ([paper title, paper abstract], [ref title, ref context], paper id)
            abs_info = xml_info[key]["abstract"] if key in xml_info.keys() else "[PAD]"     # scibert pad，空格的无法编码
            kw1_ls = paper_dblp_info[key]["keywords"] if key in paper_dblp_info.keys() else ["[PAD]"]
            kw1 = ";".join(kw1_ls)
            body = xml_info[key]["content"] if key in xml_info.keys() else "[PAD]"
            kw2_ls = paper_dblp_info[bib]["keywords"] if bib in paper_dblp_info.keys() else ["[PAD]"]
            kw2 = ";".join(kw2_ls)
            tmp_ls.append(([strID2title.get(key, "[PAD]"), abs_info, kw1, body[:500]], [strID2title.get(bib, bib), label, kw2], key))
    print("{} set number:".format(data_name), len(tmp_ls))
    np.save(join(data_dir, "{}set_text.npy".format(data_name)), tmp_ls)      # npy自动把tuple转成了ndarray (like list)


if __name__ == "__main__":
    main2()             # train data 23762; neg sampling: around 8891
    main3("valid")      # valid data 19917
    main3("test")       # test data  21003


