# RAG 的實作與優化

## 架構與基本實作流程
![](https://i.imgur.com/QZxb7ZP.png)
圖：RAG 常見架構

![](https://i.imgur.com/Lbo0Zwi.png)
圖：本範例的實作流程

## 從這裡可以學到什麼
- 評估使用哪種顆粒度的資料來作為背景知識。
- 如何使用 semantic search 和 re-ranking。
- 了解使用 RAG 和沒有使用 RAG 的差異。

## 整合式開發環境 (IDE)
- Visual Studio Code (vscode) [連結](https://code.visualstudio.com/)
  - vscode 擴充功能：SQLite3 Editor

## 環境設定
- Anaconda Installers [連結](https://www.anaconda.com/download/success)
```bash
# 安裝 conda 環境
conda create -n rag python=3.10 ipykernel

# 切換到剛安裝好的 conda 環境
conda activate rag
```

## 作業環境
- Ubuntu Linux Server 22.04
- nVIDIA GeForce RTX 3090 * 1

## 安裝套件
```bash
pip install -r requirements.txt
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

## 常見問題
1. **為什麼使用 RAG 當作問答系統的架構？**
當沒有資源去預訓練或微調一個大型語言模型（LLM）來於回答特定的問題時，可以透過資訊檢索的技術，將檢索結果當作 LLM 回答問題的參考資訊或背景知識，讓 LLM 可以依據資訊或知識來回答使用者的提問。
2. **Chunking Size 設定多少比較合適？**
個人認為 chunking size 只是一個概念，常見如每 256/512/1024 tokens 截斷一次，有時反而會失去原有的語義。人類寫作的時候，無論是單詞的使用、句子的成形、段落的舖陳，甚至是整篇文章所傳達的訊息，都有他的意義，例如迴文「上海自來水來自海上」，如果 chunking size 為 4，可能會被分成「上海自來」、「水來自海」、「上」這三個部分，如此一來，便失去要表達的意涵。
可能較好的方式，就是針對不同文件的內容，客製化建立它們的 chunks，例如一篇文章是常見作文，可以將每一個段落視為 chunk；若是一篇文章有段落也有表格，可以將段落與表格分別建立 chunk，有時候表格的前面加上原始文件的標題或是表格內容的摘要，檢索效果較好。
若是 Embedding model 的 max_seq_length 很大（可以理解更長的文字資料），有時候不進行 chunking 也是一種選擇。
3. **文件內容結構太複雜，怎麼辦？**
可以使用一些軟體或套件，將原始文件內容（如文字段落、表格等）轉變成結構化或半結構化的格式，例如 Dataframe、JSON、XML、CSV 等，再進行剖析，圖片的內容可以透過 OCR 技術進行擷取。