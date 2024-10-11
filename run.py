import requests as req

headers = {
    "Content-Type": "application/json"
}

message = '''依票據法規定，票據上記載金額之文字與號碼不符時，下列何者正確？
(A)以最低額為準
(B)以號碼為準
(C)以文字為準
(D)以探求當事人真意為準
'''

json = {
    "session_id": "sess_5566", # 這裡可以隨意填入一個字串，代表這次對話的 session id，跟其它對話區隔開來
    "message": message, # 使用者的問題或對話內容
    "multi_turn_conversation": False, # 是否為多輪對話
    "rag": False, # 是否開啟 RAG 模式
}

# 與 LLM 對話
def chat():
    with req.post(url='http://127.0.0.1:5004/generate', stream=True, headers=headers, json=json) as res:
        for chunk in res.iter_content(decode_unicode=True):
            print(chunk, end='')
        print()

if __name__ == '__main__':
    chat()