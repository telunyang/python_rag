'''
# 安裝 Sentence Transformers
$ pip install sentence-transformers
'''
import pickle
from sentence_transformers import (
    SentenceTransformer, 
    CrossEncoder, 
    util
)


'''
查詢設定
'''
query = '''依票據法規定，票據上記載金額之文字與號碼不符時，下列何者正確？
(A)以最低額為準
(B)以號碼為準
(C)以文字為準
(D)以探求當事人真意為準'''
print("查詢文字:", query)

'''
讀取 Passages 和對應的 Embeddings
'''
# 讀取 embeddings 和 passages
emb_file_path = "emb.pkl"
with open(emb_file_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    passage_embeddings = stored_data['passage_embeddings']
    passages = stored_data['passages']
    del stored_data


print("=" * 80)


'''
Semantic Search 階段
'''
##### Semantic Search #####
bi_encoder = SentenceTransformer('BAAI/bge-m3', device='cuda:0')

# 取得 query 的 embedding
question_embedding = bi_encoder.encode(
    query, 
    batch_size=1, 
    device='cuda:0'
)

# 透過語義搜尋檢索出來的文章數量
SEARCH_SIZE = 100

# 取得前幾筆資料
TOP_K = 5

# 語義搜尋
hits = util.semantic_search(
    question_embedding, 
    passage_embeddings, 
    top_k=SEARCH_SIZE
)
hits = hits[0]

print(f"Top-{TOP_K} Bi-Encoder Retrieval hits (Semantic Search)")
hits = sorted(hits, key=lambda x: x['score'], reverse=True)
for hit in hits[0:TOP_K]:
    print(f"\t{hit['score']:.3f}\t{passages[hit['corpus_id']]}")


print("=" * 80)


'''
Re-Ranking 階段
'''
##### Re-Ranking #####
cross_encoder = CrossEncoder('BAAI/bge-reranker-large', device='cuda:0')

# 用 cross_encoder 對所有檢索到的文章進行評分
cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
cross_scores = cross_encoder.predict(cross_inp)

# 透過 Cross-Encoder Re-ranker 重新排序檢索結果
for idx in range(len(cross_scores)):
    hits[idx]['cross-score'] = cross_scores[idx]

print(f"Top-{TOP_K} Cross-Encoder Re-ranker hits (Relevance Score)")
hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
for hit in hits[0:TOP_K]:
    print(f"\t{hit['cross-score']:.3f}\t{passages[hit['corpus_id']]}")