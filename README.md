# RAG 的實作與優化

## 架構與基本實作流程
![](https://i.imgur.com/QZxb7ZP.png)
圖：RAG 常見架構

![](https://i.imgur.com/UEL7DFT.png)
圖：本範例的實作流程

## 從這裡可以學到什麼
- 評估使用哪種[顆粒度](https://www.coursera.org/articles/data-granularity)（Granularity）的資料來作為背景知識。
  - 在這裡指的是 sentence-level、paragraph-level 和 document-level。
- 如何使用 semantic search 和 re-ranking。
- 了解使用 RAG 和沒有使用 RAG 的差異。

## 先備知識
- 閱讀或學習過 Natural Language Processing (NLP) 相關知識。
- Python 程式開發經驗。
- 以 Transformer 神經網路為基礎的模型使用經驗。

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

# 刪除 conda 環境
conda remove -n rag --all
```

## 作業環境
- Ubuntu Linux Server 22.04
- nVIDIA GeForce RTX 3090 * 1

## Bi-Encoder & Cross-Encoder
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks [論文](https://arxiv.org/abs/1908.10084)
![](https://i.imgur.com/OAsiTUU.png)
圖：雙向編碼器與交叉編碼器

## 會使用到的模型
Embedding model:
  - BAAI/**bge-m3** [連結](https://huggingface.co/BAAI/bge-m3) [論文](https://arxiv.org/abs/2402.03216)
    - 雙向編碼器。
    - 提供語義搜尋 (semantic search) 的功能：依「相似度」對文字（句子、段落、文件等）的 embeddings 進行排序。
    - 支援最高 8192 tokens，會產生 embeddings。
    - 幫助我們理解從資料庫檢索/查詢出來的知識（法規、條文）。
    - 使用 Question 搜尋相似的 Text (Question)。
  - BAAI/**bge-reranker-large** [連結](https://huggingface.co/BAAI/bge-reranker-large)
    - 交叉編碼器。
    - 提供重新排序 (re-rank) 的功能：依「相關性」對文字（句子、段落、文件等）的 embeddings 進行重新排序。
    - 僅重新排序 embeddings，不會另外產生 embeddings。
    - 使用 Question 找出相關性高的 Text (Answer)。
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
    - `laws_sent` -> 法律/法規的特定條文的某一句話

## 檔案說明
- `chart.ipynb`: 透過簡單的視覺化，來評估應該使用哪種顆粒度來建立 embeddings。
- `conv.py`: 將知識庫當中的文字資料，連同 embeddings 一起儲存在 Pickle 檔案中。
- `financial_laws.db`: 知識庫，以 sqlite3 的資料庫作為儲存資料的工具。
- `requirements.txt`: 套件列表。
- `run.py`: 對 Web API 發出請求，取得 LLM 回應的答案。
- `test_dense_vector_search.py`: 檢視 semantic search + re-ranking 的結果。
- `web_api.py`: 將 LLM 整合在 Web API 當中，接收請求，並生成回應。

## 安裝套件
```bash
pip install -r requirements.txt
```

## 操作指令
1. 先使用 `chart.ipynb` 來評估要以整部法規、各別條文，還是條文的項、款、目來作為 chunks。
2. 建立與儲存 embeddgins (包含對應的 passages，也就是 chunks)
    ```bash
    # 建立 emb.pkl 檔 (會在 test_dense_vector_search.py 和 web_api.py 使用)
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

    # 建立跟語言模型對話的服務 (使用 Flask 作為 Web Service)
    python web_api.py

    # 離開 tmux session (不會關閉 session)
    先按下 ctrl + b 後，放掉，再按 d

    # 要再進入先前建立的終端機 session
    tmux a -t f_api
    ```
4. 在 RAG 架構下，跟語言模型對話 (使用 `requests` 模擬互動)
    ```bash
    python run.py
    ```
5. 關閉 Web API 服務
    ```bash
    # 進入先前建立的終端機 session
    tmux a -t f_api

    # 關閉 Flask 服務
    按下 Ctrl + C，確認目前是能夠輸入指令的狀態
    ```
6. 關閉 tmux session
    ```bash
    # 在可以輸入指令的狀態下
    輸入 exit 後，按下 enter，回到 Terminal
    ```

## 有關文字生成的策略
- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
- Sampling
    ```python
    generation_kwargs = dict(
        ... 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.9,
        ...
    )
    ```
- Greedy Search or Beam Search
    ```python
    generation_kwargs = dict(
        ...
        do_sample=False, 
        num_beams=5, 
        no_repeat_ngram_size=2,
        early_stopping=True,
        ...
    )
    ```

## 常見問題
1. **為什麼使用 RAG 當作問答系統的架構？**

    當沒有資源去預訓練或微調一個大型語言模型（LLM）來於回答特定的問題時，可以透過資訊檢索的技術，將檢索結果當作 LLM 回答問題的參考資訊或背景知識，讓 LLM 可以依據資訊或知識來回答使用者的提問。

2. **Chunking Size 設定多少比較合適？**

    個人認為 chunking size 只是一個概念，常見如每 128/256/512/1024 tokens 截斷一次，有時反而會失去原有的語義。人類寫作的時候，無論是單詞的使用、句子的成形、段落的舖陳，甚至是整篇文章所傳達的訊息，都有他的意義，例如迴文「上海自來水來自海上」，如果 chunking size 為 4，可能會被分成「上海自來」、「水來自海」、「上」這三個部分，如此一來，便失去原先要表達的意涵。

    可能較好的方式，就是針對不同文件的內容，客製化建立它們的 chunks。例如一篇文章是常見作文，可以將每一個段落視為 chunk，若是文章當中有段落也有表格，可以將段落與表格分別建立 chunk，有時候段落和表格的前面加上原始文件的標題或是表格內容的摘要，檢索效果較好；若文章中的每個句子或短文都有各別的意義，例如成語、國語辭典（有時候包括解釋說明），那就以句子或短文形式來建立 chunk。

    倘若 Embedding model 的 max_seq_length 很大（可以理解更長的文字資料），有時候不進行 chunking 也是一種選擇。

3. **文件內容結構太複雜，怎麼辦？**

    可以使用一些軟體或套件，將原始文件內容（如文字段落、表格等）轉變成結構化或半結構化的格式，例如 Dataframe、JSON、XML、CSV 等，再進行剖析，圖片的內容可以透過 OCR 技術進行擷取。

4. **有其它檢索方案可以使用嗎？**

    如果指的是案例中的向量檢索，可以考慮使用 [Huggingface: intfloat](https://huggingface.co/intfloat) 的 E5 模型，它是直接以 Question-Answer 成對關係的資料進行訓練，或許不需要進行 semantic search + re-ranking，直接以 QA 格式進行相似度比對。bge-m3 可以支援到 8192 tokens，如果其用其它 embedding model，要連同資料可能佔用的 tokens (例如可支援的 max_seq_length) 一起考慮進去。

    如果指的是以統計量為基礎的檢索方法，可以考慮使用 [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) 進行檢索。它結合了 [TF-IDF](https://zh.wikipedia.org/zh-tw/Tf-idf) 和[詞袋模型](https://zh.wikipedia.org/zh-tw/词袋模型)的特性，加入了文件平均長度等元素，增強了檢索的效果。
    
    在實務中，BM25 通常可以達到很好的效果，若是沒有辦法使用向量檢索（例如沒有 GPU、主機效能不佳等因素），推薦使用 BM25，惟 BM25 需要針對特定領域建立關鍵字詞庫（例如 [jieba](https://github.com/fxsjy/jieba) 的 `jieba.load_userdict()` 功能），如果沒有預先整理出合適的關鍵字詞庫，檢索效果可能會不如預期。

5. **一定要使用 Re-ranking 嗎？**

    視任務需求而定，如果你只是希望找到跟查詢字串相似度高的文字資料（例如用 Question 搜尋很相似的 Question），語義搜尋就足夠了；倘若文字資料混雜了很多資訊，例如 Questions 和 Answers 都放在一起，沒有區分，使用 Re-ranking 的話，用來回答查詢字串的 Answers 有可能因為排名提升而被看見。

6. **Pickle 檔案太大，怎麼辦？**

    有一種作法，就是將不同用途或種類的 Pickle 檔案區分開來，需要用到的時候才讀取，不需要則釋放，彈性地使用 Pickle 檔。

    如果你是進階使用者，可以考慮使用 [FAISS](https://github.com/facebookresearch/faiss)、[ElasticSearch](https://www.elastic.co/elasticsearch/vector-database)、[Milvus](https://milvus.io/)、[Weaviate](https://weaviate.io/) 等向量索引/資料庫儲存工具，可以更便利地使用/儲存向量。