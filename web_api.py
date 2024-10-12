import torch
import pickle
from time import sleep
from threading import Thread
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import  AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from flask import Flask, request
from flask_ipfilter import IPFilter, Whitelist

# logging 設定
log_filename = 'web_api'
logger = logging.getLogger(log_filename)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
fileHandler = logging.FileHandler(f'{log_filename}.log', mode='w', encoding='utf-8')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


'''
Flask Web API
'''
# 建立 Flask 物件
app = Flask(__name__)

# 設定白名單
ip_filter = IPFilter(app, ruleset=Whitelist())
ip_filter.ruleset.permit("127.0.0.1")

'''
變數初始化
'''
# 使用字典來保存每個會話的對話歷史
sessions_history = {}

# 讀取 model 與 tokenizer
llm_name = 'MediaTek-Research/Breeze-7B-Instruct-v1_0'

# 張量格式
torch_dtype = torch.bfloat16 # or torch.float16

# 讀取模型，自訂參數
model = AutoModelForCausalLM.from_pretrained(
    llm_name, 
    torch_dtype=torch_dtype,
)

# 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 取得 model 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_name)

# 建立 TextIteratorStreamer 物件
streamer = TextIteratorStreamer(tokenizer=tokenizer)

# 讀取 embedding models
bi_encoder = SentenceTransformer('BAAI/bge-m3', device='cuda:0')
cross_encoder = CrossEncoder('BAAI/bge-reranker-large', device='cuda:0')

# 讀取 embeddings 和 passages
emb_file_path = "emb.pkl"
with open(emb_file_path, "rb") as fIn:
    stored_data = pickle.load(fIn)
    passage_embeddings = stored_data['passage_embeddings']
    passages = stored_data['passages']
    del stored_data

# 取得 semantic search + re-ranking 之後的結果
def get_resutls(query, search_size, top_k):
    # 取得 query 的 embedding
    question_embedding = bi_encoder.encode(
        query, 
        batch_size=1, 
        device='cuda:0'
    )

    # 語義搜尋
    hits = util.semantic_search(
        question_embedding, 
        passage_embeddings, 
        top_k=search_size
    )
    hits = hits[0]

    # 用 cross_encoder 對所有檢索到的文章進行評分
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # 透過 Cross-Encoder (Re-ranker) 重新排序檢索結果
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # 放置 Re-Ranking 後的檢索結果
    results = []
    scores = []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[:top_k]:
        results.append(passages[hit['corpus_id']])
        scores.append(hit['cross-score'])

    return results, scores



# 搜尋特定文字
@app.route("/generate", methods=["POST"])
def generate():
    # 取得前端傳來的 JSON 格式資料
    data = request.json

    # 取得 session id 和 使用者的訊息
    session_id = data.get("session_id")
    message = data.get("message")
    multi_turn_conversation = data.get("multi_turn_conversation")
    rag = data.get("rag")

    # 如果 rag 為 True，則進行 retrieval (semantic search + re-ranking)
    if rag:
        # 設定檢索參數
        search_size = 100
        top_k = 5

        # 取得 retrieval 的結果
        results, scores = get_resutls(message, search_size, top_k)

        # 將結果轉換成文字
        knowledge = ''
        for index, context in enumerate(results):
            knowledge += f"{index+1}. {context}\n"

        # 建立 user prompt
        message = f'''請參考以下資訊：
{knowledge}

==================================================

問題：
{message}

==================================================

答案是：'''

        # 檢視 user prompt
        logger.info(message)

    # 如果 session id 不存在於對話記錄中，針對這個 session id 建立一個新的會話記錄; 如果要遺忘先前的對話，則重新建立一個新的會話記錄
    if session_id not in sessions_history or multi_turn_conversation == False:
        # 初始化對話記錄
        sessions_history[session_id] = [
            {"role": "system", "content": "你是個擁有金融科技 (FinTech) 專業能力的 AI 對話系統。請以繁體中文回答使用者的問題。請保持友善、禮貌，並回答詳細的資訊給使用者。"},
        ]

    # 將使用者的訊息，加入到會話記錄中
    sessions_history[session_id].append({"role": "user", "content": message})

    # 生成回應
    def generate_responses():
        # 將對話記錄中的多輪對話，從 dict 格式轉換成一段文字，幫助模型生成回應
        tokenized_chat = tokenizer.apply_chat_template(
            conversation=sessions_history[session_id],
            add_generation_prompt=True,
            tokenize=False
        )

        # 將文字進行 tokenize，讓 model 能了解文字的結構與順序
        inputs = tokenizer(tokenized_chat, return_tensors="pt").to('cuda:0')

        # 生成設定
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500, 
            do_sample=True, 
            temperature=0.2, 
            top_k=50, 
            top_p=0.9,
            # num_beams=1, 
            # no_repeat_ngram_size=3,
            # early_stopping=True, 
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer # `streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.
        )

        # 透過 thread 取得生成結果
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        
        # 啟動 thread
        thread.start()

        # 記錄生成文字的資訊
        generated_text = ''

        for index, new_text in enumerate(streamer):
            # index == 0 的 new_token 是 prompt 完整字串，後面才是生成的文字
            if index == 0: continue
            generated_text += new_text
            yield new_text

        logger.info(generated_text)
        logger.info('\n\n')

        # 將生成的回應加入到會話歷史中
        sessions_history[session_id].append({"role": "assistant", "content": generated_text})

        # 等待 thread 完成
        thread.join()

    # 回傳 ai assistant 生成的回應
    return generate_responses()
    
# 主程式區域
if __name__ == '__main__':
    app.debug = False
    app.json.ensure_ascii = False
    app.run(
        host='127.0.0.1', # 0.0.0.0 
        port=5004
    )

