# KDDCupFinal
## Introduction
### KDD Cup 2024 OAG PST
KDD CUP PST, Team AoboSama, Rank 12th

## Prerequisites
- Python 3.8
- PyTorch 1.12.0

## Getting Started
### Installation

Clone or download this repo.

Please install dependencies by

```bash
pip install -r requirements.txt
```

## PST Dataset
The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1I_HZXBx7U0UsRHJL5JJagw?pwd=bft3) with password bft3, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/PST/PST.zip) or [DropBox](https://www.dropbox.com/scl/fi/namx1n55xzqil4zbkd5sv/PST.zip?rlkey=impcbm2acqmqhurv2oj0xxysx&dl=1).
The paper XML files are generated by [Grobid](https://grobid.readthedocs.io/en/latest/Introduction/) APIs from paper pdfs.

### Running Steps

- First, download DBLP dataset from [AMiner](https://opendata.aminer.cn/dataset/DBLP-Citation-network-V16.zip).
Put the unzipped PST directory into ``data/`` and unzipped DBLP dataset into ``data/PST/``.
- We use the pretrained sci-bert model, which is saved in pretrain_models/bertmodel folder. Please download the sci-bert model from huggingface and save it in pretrained_models/bertmodel.

```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
export PYTHONPATH="`pwd`:$PYTHONPATH"
```
- 1. run get_ref_kg.py: will get 3 files in data/PST/ folder: strID_to_title.json, Valid_papers_refs.json, Test_papers_refs.json.
  ```bash
  python get_ref_kg.py
  ```
- 2. run extract_TV_info.py: will get 2 files in data/PST/ folder: Train_Valid_Test_paper_more_info_from_dblp.json, TVT_paper_info_from_xml.json
  ```bash
  python extract_TV_info.py
  ```
- 3. run extract_TV_ref_context.py: will get 3 files in data/PST/ folder: bib_to_contexts_labels.json, bib_to_contexts_valid.json, bib_to_contexts_test.json
  ```bash
  python extract_TV_ref_context.py
  ```
- 4. run dataload.py: will get 3 files in data/PST/ folder: train_text.npy, valid_text.npy, test_text.npy
  ```bash
  python dataload.py
  ```
- 5. run NCF.py: there are three arguments: 
    - if --train True, will train and save the model in out/ncf_text_model/ folder: torch_model_lr.bin; 
    - if --valid True, will use the saved model to predict the validation set and save the result in out/result/ folder: valid_submission_ncf_text_lr.json;
    - if --test True, will use the saved model to predict the test set and save the result in out/result/ folder: test_submission_ncf_text_lr.json.
  ```bash
  python NCF.py --train True --valid True --test True
  ```
- 6. If you want to run training script and test script separately, you can run the following scripts:
  ```bash
  python train.py
  python modeltest.py
  ```
  
### Results
- The map result on Valiation Set is 0.3906  
- The map result on Test Set are as follows is 0.3781

### !!! Trained Model and Submission Files
#### !!! We save the trained model, valid and test submission files which achieve the above results. You can find it in our_result/ folder.
- You can also download our result model and submission files from 
  - [Hugging Face] https://huggingface.co/MyloveAB/KDDPST-TeamAoboSama-our_result/tree/main
  - [BaiduPan] https://pan.baidu.com/s/1buZ3xwoHicIWXaRev6YV8w with extract code: uouc
  - [Google Drive] https://drive.google.com/drive/folders/1bRzp1OcKGAFeA_-XXBdFqdWXhOiA4LS_?usp=sharing
- If you have any questions or need more details, please contact us with email: upc20xab@s.upc.edu.cn
