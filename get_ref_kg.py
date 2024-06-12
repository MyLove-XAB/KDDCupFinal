from fuzzywuzzy import fuzz
import json
from os.path import join
from lxml import etree
from tqdm import tqdm
import logging
import settings
import utils
from bs4 import BeautifulSoup


def is_fuzzy_match(title_text, title_set, threshold=80):
    """检查 title_text 是否与 title_set 中的任何标题模糊匹配"""
    for existing_title in title_set:
        if fuzz.ratio(title_text, existing_title) >= threshold:
            return True, existing_title
    return False, None


def get_paper_reference():      # valid json
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    in_dir = join(data_dir, "paper-xml")
    dblp_fname = "DBLP-Citation-network-V15.1.json"

    paper_dict_open = {}
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")
    papers_test = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    sub_example_dict_test = utils.load_json(data_dir, "submission_example_test.json")

    title_to_id = {}
    id_to_title = {}

    # 初始化一个计数器 parse_err_cnt，用于记录解析错误的总数
    parse_err_cnt = 0

    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            # 如果当前行数是 100000 的倍数，则输出当前行的论文数量和解析错误的总数
            if i % 100000 == 0:
                logging.info("reading papers %d, parse err cnt %d", i, parse_err_cnt)
            try:
                paper_tmp = json.loads(line.strip())

                # 将论文的引文信息添加到 paper_dict_open 字典中
                paper_dict_open[paper_tmp["id"]] = paper_tmp.get("references", [])
                # 将标题和 ID 添加到 title_to_id 字典中
                title_to_id[paper_tmp["title"].lower()] = paper_tmp["id"]
                # 将 ID 和标题添加到 id_to_title 字典中
                id_to_title[paper_tmp["id"]] = paper_tmp["title"]

            except Exception as e:
                parse_err_cnt += 1
                logging.error("Parsing error occurred: %s", str(e))

    # 从 papers_train 和 papers_valid 中获取论文 ID 对应的引文信息
    TV_paper_id_dict = {}
    # articles_with_more_xml_refs = []
    in_dir = join(data_dir, "paper-xml")
    strid_to_title = dict()

    for paper in tqdm(papers_train):        # train json data
        pid = paper["_id"]
        try:
            strid_to_title[pid] = id_to_title[pid].lower()
        except KeyError:
            print("paper_id not found in dblp: ", pid)
        cur_refs = paper.get("references", [])
        for tmp_ref in cur_refs:
            try:
                strid_to_title[tmp_ref] = id_to_title[tmp_ref].lower()
            except KeyError:
                print("ref_id not found in dblp: ", tmp_ref)
                print("paper_id: ", pid)

    for paper in tqdm(papers_valid):        # valid json data
        pid = paper["_id"]
        try:
            strid_to_title[pid] = id_to_title[pid].lower()
        except KeyError:
            print("paper_id not found in dblp: ", pid)
        cur_refs = paper.get("references", [])              # list of references from valid data
        cur_refs_title_to_id = dict()
        for tmp_ref in cur_refs:
            try:
                cur_refs_title_to_id[id_to_title[tmp_ref].lower()] = tmp_ref
                strid_to_title[tmp_ref] = id_to_title[tmp_ref].lower()
            except KeyError:
                # 存在某些ref_id在dblp中找不到的情况: v15.1: 5e8d8e6d9fced0a24b5d669e  v15: 62376b725aee126c0f0a7412
                print("ref_id not found in dblp: ", tmp_ref)
                print("paper_id: ", pid)
        if len(cur_refs) == 0:
            continue

        refs_open = paper_dict_open.get(pid, [])            # list of references from dblp
        refs_update = list(set(cur_refs + refs_open))
        id_set = set(refs_update)
        title_set = {id_to_title[ref_id].lower() for ref_id in refs_update if ref_id in id_to_title}
        # TV_paper_id_dict[pid] = refs_update
        final_refs = []     # 用来存最终的结果

        f = open(join(in_dir, pid + ".xml"), encoding='utf-8')
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
            if ref.analytic.title.text.lower() != "":
                bid_to_title[bid] = ref.analytic.title.text.lower()
            else:
                bid_to_title[bid] = ref.monogr.title.text.lower()

            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        assert len(sub_example_dict[pid]) == n_refs     # 确保reference的数量和submission_example中的数量一致，所以他通过这种方式和valid data中的数据匹配起来了，没有通过ref xml的id进行匹配的
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for i in range(n_refs):
            title_text = bid_to_title.get(bib_sorted[i], "")
            if title_text == "":
                print("current paper: ", pid)
                print("bib: ", bib_sorted[i])
                print("not found bid: ", i)
                final_refs.append(bib_sorted[i])            # 没有title的话，就直接用bid
                continue
            signal, existing_title = is_fuzzy_match(title_text, title_set)
            if not signal:  # 在dblp中匹配不上
                final_refs.append(title_text)
            else:
                # try:
                final_refs.append(title_to_id[existing_title])      # 大小写的原因会导致匹配不上
                # except KeyError:        # dblp里面找不到的话，就用valid里面的refs的id
                #     final_refs.append(cur_refs_title_to_id[existing_title])

        TV_paper_id_dict[pid] = final_refs

    # # 输出 ref_num_xml 大于 ref_num 的文章 ID
    # with open(join(data_dir, "articles_with_more_xml_refs.txt"), 'w') as f:
    #     for pid in articles_with_more_xml_refs:
    #         f.write("%s\n" % pid)

    # test data
    TV_paper_id_dict_test = {}
    # articles_test_with_more_xml_refs = []
    for paper in tqdm(papers_test):        # valid json data
        pid = paper["_id"]
        try:
            strid_to_title[pid] = id_to_title[pid].lower()
        except KeyError:
            print("test paper_id not found in dblp: ", pid)
        cur_refs = paper.get("references", [])  # list of references from valid data
        cur_refs_title_to_id = dict()
        for tmp_ref in cur_refs:
            try:
                cur_refs_title_to_id[id_to_title[tmp_ref].lower()] = tmp_ref
                strid_to_title[tmp_ref] = id_to_title[tmp_ref].lower()
            except KeyError:
                # 存在某些ref_id在dblp中找不到的情况: v15.1: 5e8d8e6d9fced0a24b5d669e  v15: 62376b725aee126c0f0a7412
                print("ref_id not found in dblp: ", tmp_ref)
                print("test paper_id: ", pid)
        if len(cur_refs) == 0:
            continue

        refs_open = paper_dict_open.get(pid, [])  # list of references from dblp
        refs_update = list(set(cur_refs + refs_open))
        id_set = set(refs_update)
        title_set = {id_to_title[ref_id].lower() for ref_id in refs_update if ref_id in id_to_title}
        # TV_paper_id_dict_test[pid] = refs_update
        final_refs = []  # 用来存最终的结果

        f = open(join(in_dir, pid + ".xml"), encoding='utf-8')
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
            if ref.analytic.title.text.lower() != "":
                bid_to_title[bid] = ref.analytic.title.text.lower()
            else:
                bid_to_title[bid] = ref.monogr.title.text.lower()

            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        assert len(sub_example_dict_test[pid]) == n_refs  # 确保reference的数量和submission_example中的数量一致，所以他通过这种方式和valid data中的数据匹配起来了，没有通过ref xml的id进行匹配的
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for i in range(n_refs):
            title_text = bid_to_title.get(bib_sorted[i], "")
            if title_text == "":
                print("current paper: ", pid)
                print("bib: ", bib_sorted[i])
                print("not found bid: ", i)
                final_refs.append(bib_sorted[i])  # 没有title的话，就直接用bid
                continue
            signal, existing_title = is_fuzzy_match(title_text, title_set)
            if not signal:  # 在dblp中匹配不上
                final_refs.append(title_text)
            else:
                # try:
                final_refs.append(title_to_id[existing_title])  # 大小写的原因会导致匹配不上
                # except KeyError:        # dblp里面找不到的话，就用valid里面的refs的id
                #     final_refs.append(cur_refs_title_to_id[existing_title])

        TV_paper_id_dict_test[pid] = final_refs

    utils.dump_json(strid_to_title, data_dir, "strID_to_title.json")        # NCF中的重点
    utils.dump_json(TV_paper_id_dict, data_dir, "Valid_papers_refs.json")
    utils.dump_json(TV_paper_id_dict_test, data_dir, "Test_papers_refs.json")
    print("验证集总数：", len(TV_paper_id_dict))
    print("测试集总数：", len(TV_paper_id_dict_test))


if __name__ == "__main__":
    get_paper_reference()
