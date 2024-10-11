# RAG 的實作與優化

## 環境設定
```bash
conda create -n rag python=3.10 ipykernel
```

## 作業環境
- Ubuntu Linux Server 22.04
- nVIDIA GeForce RTX 3090 * 1

## 安裝套件
```bash
pip install -r requestments.txt
```

## 會使用到的模型
嵌入模型
  - BAAI/**bge-m3** [連結](https://huggingface.co/BAAI/bge-m3)
    - 提供語義搜尋 (semantic search) 的功能：依「相似度」對文字（字詞、句子、段落、文件等）進行排序
    - 支援最高 8192 tokens，會產生 embeddings
    - 幫助我們理解從資料庫檢索/查詢出來的知識（法規、條文）
  - BAAI/**bge-reranker-large** [連結](https://huggingface.co/BAAI/bge-reranker-large)
    - 提供重新排序 (re-rank) 的功能：依「相關性」對文字（字詞、句子、段落、文件等）進行重新排序
    - 僅重新排序 embeddings，不會另外產生 embeddings
  - 以臺北市立圖書館的常見問答為例：[連結](https://tpml.gov.taipei/)

語言模型
  - MediaTek-Research/**Breeze-7B-Instruct-v1_0** [連結](https://huggingface.co/MediaTek-Research/Breeze-7B-Instruct-v1_0) 

## 整合式開發環境 (IDE)
- Visual Studio Code (vscode) [連結](https://code.visualstudio.com/)
  - vscode 擴充功能：SQLite3 Editor

## 背景知識來源
- 全國法規資料庫 [連結](https://law.moj.gov.tw/)
  - 嘗試整理出與**財政、金融**相關的法規

## 資料庫 (知識庫)
- financial_laws.db
  - 將法規進行 chunking，分成：
    - 以單部法規為主
    - 以法規的條文為主
    - 以條文的項、款、目為主
  - 資料表名稱：
    - `laws_full` -> 法律/法規的所有條文
    - `laws_part` -> 法律/法規的某個條文
    - `laws_row`  -> 法律/法規的特定條文的某一句話

## 操作指令
1. 先使用 `chart.ipynb` 來評估要以整部法規、各別條文，還是條文的項、款、目來作為 chunks。
2. 建立與儲存 embeddgins (包含對應的 passages，也就是 chunks)
    ```bash
    # 建立 emb.pkl 檔 (會在 test_reranking.py 和 web_api.py 使用)
    time python conv.py

    # 測試 RAG 的 dense vector search 功能 (要先建立 emb.pkl 檔)
    time python test_dense_vector_search.py
    ```
3. 在終端機啟動跟語言模型對話的 Web API 服務
    ```bash
    # 安裝 tmux (教學網頁: https://blog.gtwang.org/linux/linux-tmux-terminal-multiplexer-tutorial/)
    sudo apt-get install tmux

    # 建立終端機（terminal）的 session
    tmux new -s f_api

    # 切換至先前新增的 conda 虛擬環境
    conda activate rag

    # 建立跟語言模型對話的服務
    python web_api.py

    # 離開 tmux session (不會終止 session)
    先按下 ctrl + b 後，放掉，再按 d

    # 要再進入先前建立的終端機 session
    tmux a -t f_api

    # 終止 session
    進入 session，直接輸入 exit 後，按下 enter 即可
    ```
4. 在 RAG 架構下，跟語言模型對話 (使用 `requests` 模擬互動)
    ```bash
    python run.py
    ```