import json
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
import logging
import settings
import utils
from bs4 import BeautifulSoup
from lxml import etree

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_dblp_data(data_dir, dblp_fname):
    parse_err_cnt = 0
    paper_dict_dblp = {}

    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            if i % 10000 == 0:
                logger.info("reading papers %d, parse err cnt %d", i, parse_err_cnt)
            try:
                paper_tmp = json.loads(line.strip())
                paper_dict_dblp[paper_tmp["id"]] = paper_tmp
            except Exception as e:
                logger.error(f"Error parsing line {i}: {e}")
                parse_err_cnt += 1

    logger.info("number of papers after loading %d", len(paper_dict_dblp))
    return paper_dict_dblp


def get_paper_reference(data_dir, papers_train, papers_valid, papers_test, paper_dict_dblp):
    TV_paper_id_dict = {}
    for paper in tqdm(papers_train + papers_valid + papers_test):
        pid = paper["_id"]
        cur_refs = paper.get("references", [])
        if len(cur_refs) == 0:
            continue
        refs_open = paper_dict_dblp.get(pid, {}).get("references", [])
        refs_update = list(set(cur_refs + refs_open))
        TV_paper_id_dict[pid] = refs_update

    with open(join(data_dir, "TV_pid_list.txt"), 'w') as f:
        for pid in TV_paper_id_dict:
            f.write("%s\n" % pid)

    utils.dump_json(TV_paper_id_dict, data_dir, "Train_Valid_Test_papers_refs.json")
    print("训练集+验证集总数", len(TV_paper_id_dict))


def get_more_info_from_dblp(data_dir, papers_train, papers_valid, papers_test, paper_dict_dblp):
    paper_more_info_from_dblp = dd(dict)
    for paper in tqdm(papers_train + papers_valid + papers_test):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = [cur_pid] + ref_ids
        for pid in pids:
            if pid not in paper_dict_dblp:
                continue
            cur_paper_info = paper_dict_dblp[pid]
            title = cur_paper_info.get("title", "")
            abstract = cur_paper_info.get("abstract", "")
            keywords = cur_paper_info.get("keywords", "")
            cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
            n_citation = cur_paper_info.get("n_citation", 0)
            venue = cur_paper_info.get("venue", "")

            paper_more_info_from_dblp[pid] = {
                "title": title, "abstract": abstract, "keywords": keywords,
                "authors": cur_authors, "n_citation": n_citation, "venue": venue
            }
    print("number of papers after filtering", len(paper_more_info_from_dblp))
    utils.dump_json(paper_more_info_from_dblp, data_dir, "Train_Valid_Test_paper_more_info_from_dblp.json")


def get_info_from_XML(data_dir, in_dir, papers_train, papers_valid, papers_test):
    all_paperinfo = dd(dict)
    for paper in tqdm(papers_train + papers_valid + papers_test):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        ref_num = len(ref_ids)

        paperinfo = dd(dict)

        try:
            path = join(in_dir, cur_pid + ".xml")
            tree = etree.parse(path)
            root = tree.getroot()
            listBibl = root.xpath("//*[local-name()='listBibl']")[0]
            biblStruct = listBibl.getchildren()
            num_ref_xml = len(biblStruct)
        except OSError:
            tree = None
            num_ref_xml = 0
            print('not exits xml ' + cur_pid)

        paperinfo['num_ref'] = num_ref_xml

        with open(path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        bs = BeautifulSoup(xml_content, 'xml')

        title_tag = bs.fileDesc.titleStmt.title
        paper_title = title_tag.text if title_tag else ""
        paperinfo['paper_title'] = paper_title

        abstract = bs.profileDesc.abstract.text.strip() if bs.profileDesc.abstract else ""
        paperinfo['abstract'] = abstract

        body_tag = bs.find('body')
        div_content_list = []
        if body_tag:
            # 在<body>标签内部查找所有的<div>标签
            div_tags = body_tag.find_all('div')
            # 初始化一个空列表，用于存储提取的内容
            div_content_list = []
            # 遍历每个<div>标签，提取其文本内容，并添加到列表中
            for div_tag in div_tags:
                div_content_list.append(div_tag.get_text())
            # 将列表中的内容连接成一个字符串
            div_content = '\n'.join(div_content_list)
            paperinfo['content'] = div_content
        all_paperinfo[cur_pid] = paperinfo

    print("总数量：", len(all_paperinfo))
    utils.dump_json(all_paperinfo, data_dir, "TVT_paper_info_from_xml.json")


def main():
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    in_dir = join(data_dir, "paper-xml")
    dblp_fname = "DBLP-Citation-network-V15.1.json"

    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers_test = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    paper_dict_dblp = load_dblp_data(data_dir, dblp_fname)

    # get_paper_reference(data_dir, papers_train, papers_valid, papers_test, paper_dict_dblp)
    get_more_info_from_dblp(data_dir, papers_train, papers_valid, papers_test, paper_dict_dblp)
    get_info_from_XML(data_dir, in_dir, papers_train, papers_valid, papers_test)


if __name__ == "__main__":
    main()
