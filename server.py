from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
import requests
from dotenv import load_dotenv
import os
import re

# 環境変数の読み込み
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face APIトークン

app = Flask(__name__)
CORS(app, origins=["https://react-client-x82h.onrender.com/"], supports_credentials=True)

#server立ち上がったかチェック用、デプロイすると「/」に１番にアクセスする
@app.route('/', methods=['GET'])
def index():
    return 'Flask server is up.'

# # YOLOモデルの読み込み
# model = YOLO("best.pt")
# print(model.info())

# ✅ モデルはまだ読み込まない、遅延読み込みモード
model = None

# 注文と最新の検出ラベルを保持するグローバル変数
orders_storage = []
latest_detected_labels = []

# Hugging Face LLMを使って、注文と画像ラベルのマッチングを判定する関数
def ask_chatgpt_match(detected_labels, order_products):
    try:
        detected_labels_str = '\n'.join(detected_labels) if isinstance(detected_labels, list) else detected_labels
        order_products_str = '\n'.join(order_products) if isinstance(order_products, list) else order_products

        prompt = f"""
あなたは、注文された複数の商品名と、画像認識で検出された複数のラベルとのマッチングを厳密に判定するAIです。

以下に「画像認識で検出された商品名のリスト」と、「注文された商品名のリスト」を与えます。  
各注文商品に対して、マッチする検出ラベルがあるかどうかを1つだけ判断してください。

マッチの判断基準は以下の通りです：

【マッチとみなす条件】
- 表記の揺れで意味が同じ場合はマッチ。
- 一般的な言い換えもマッチとする。

【マッチとしない条件】
- 同カテゴリでも意味が異なる場合はマッチとしない。
- 同じ料理でも具材や調理法に違いがある場合もマッチしない。
- あいまいなケースや不明瞭な一致は最大限の能力を活用し、サーチしてマッチングするかを検討する。

【入力】
画像認識で検出された商品名リスト：
{detected_labels_str}

注文された商品名リスト：
{order_products_str}

【出力】
各注文商品に対して、以下の形式で1件ずつマッチする検出ラベルを出力してください。

形式：
注文: [注文商品名] → 検出: [マッチした検出ラベル]  
※マッチしない場合は「検出: なし」と記載してください。

※ 各注文に対してマッチする検出ラベルは最大1つです。
※ 出力は上記形式に正確に従ってください。
"""
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.3,
                "return_full_text": False
            }
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            print("HF APIエラー:", response.status_code, response.text)
            return False, None

        result = response.json()
        generated_text = result[0].get('generated_text', '').strip()
        print("HF応答:", generated_text)

        matches = re.findall(r'注文:\s*(.+?)\s*→\s*検出:\s*(.+)', generated_text)
        if not matches:
            return False, None

        matched_label = matches[0][1].strip()

        for label in detected_labels:
            if matched_label.lower() == label.lower():
                return True, label

        return False, matched_label

    except Exception as e:
        print("Hugging Face通信エラー:", e)
        return False, None

@app.route('/predict', methods=['POST'])
def predict_route():
    global latest_detected_labels, orders_storage, model
    if model is None:
        model = YOLO("best.pt")  # 初回リクエスト時だけ読み込む
        print(model.info())

    if 'image' not in request.files:
        return jsonify({'error': '画像が送信されていません'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result = model(image_bgr)[0]
    labels = [model.names[int(cls)] for cls in result.boxes.cls]
    latest_detected_labels = labels
    print("検出されたラベル:", labels, flush=True)

    result_img = result.plot()
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # chatGPTなし
    # 注文とのマッチング
    matched_orders = []
    matched_customers = set()
    for order in orders_storage:
        product = order.get('comment', '')
        matched = product in latest_detected_labels
        order_with_flag = {**order, 'matched': matched}
        matched_orders.append(order_with_flag)
        if matched:
            matched_customers.add(order.get('name', ''))

    print("マッチングされた注文:", matched_orders, flush=True)

    return jsonify({
        'labels': labels,
        'image': img_base64,
        'matchedOrders': matched_orders,
        'matchedCustomers': list(matched_customers),
        'label': ', '.join(matched_customers) if matched_customers else 'なし'
    })
    
    # 森屋くんごめんね
    # chatGPTでラベルマッチング
    # matched_orders = []
    # matched_customers = set()

    # for order in orders_storage:
    #     product_name = order.get('comment', '')
    #     matched, matched_label = ask_chatgpt_match(latest_detected_labels, product_name)

    #     if matched:
    #         for label in latest_detected_labels:
    #             if product_name.lower() in label.lower() or label.lower() in product_name.lower():
    #                 matched_label = label
    #                 break
    #     else:
    #         matched_label = product_name

    #     order_with_flag = {
    #         **order,
    #         'matched': matched,
    #         'matched_label': matched_label
    #     }
    #     matched_orders.append(order_with_flag)

    #     if matched:
    #         matched_customers.add(order.get('name', ''))

    # print("マッチした注文一覧:", matched_orders, flush=True)

    # return jsonify({
    #     'labels': labels,
    #     'image': img_base64,
    #     'matchedOrders': matched_orders,
    #     'matchedCustomers': list(matched_customers),
    #     'label': ', '.join(matched_customers) if matched_customers else 'なし'
    # })

@app.route('/orders', methods=['POST'])
def receive_orders():
    global orders_storage

    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({'error': '注文リストの形式が不正です'}), 400

    orders_storage.extend(data)
    print("保存された注文:", orders_storage, flush=True)

    return jsonify({
        'message': '注文を保存しました',
        'orders_received': len(data)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)