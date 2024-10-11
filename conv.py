import sqlite3
import pickle
from sentence_transformers import SentenceTransformer

# 讀取 BGE-M3 的 model
bi_encoder = SentenceTransformer('BAAI/bge-m3', device='cuda:0')

# 連接資料庫
conn = sqlite3.connect("./financial_laws.db")

# 查詢出來的結果 (tuple)，變成 key-value 型式 (dict)
conn.row_factory = sqlite3.Row

# 建立 cursor
cursor = conn.cursor()

'''
資料表名稱:
laws_full -> 法律/法規的所有條文
laws_part -> 法律/法規的某個條文
laws_row  -> 法律/法規的特定條文的某一句話
'''

# 定義 SQL 語法
table_name = 'laws_part'
sql = f'''
SELECT content
FROM {table_name};
'''

# 取得連線資料庫進行查詢的物件
stmt = cursor.execute(sql)

# 取得所有文字資料
passages = []
for index, obj in enumerate(stmt.fetchall()):
    passages.append(obj['content'])

# 建立嵌入/向量
passage_embeddings = bi_encoder.encode(
    passages, 
    batch_size=2, 
    device='cuda:0',
    convert_to_tensor=False, 
    show_progress_bar=True
)

# 儲存 documents 的 embeddings 和 passages
emb_file_path = "emb.pkl"
with open(emb_file_path, "wb") as fOut:
    pickle.dump({
        'passage_embeddings': passage_embeddings,
        'passages': passages
    }, fOut, protocol=pickle.HIGHEST_PROTOCOL)