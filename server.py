
# from flask import Flask, send_file
# from flask_cors import CORS  # 追加
# import cv2
# import numpy as np
# import io  # ioをインポート
# from PIL import Image  # PIL（Pillow）のImageモジュールをインポート
# app = Flask(__name__)
# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# # CORS を全てのオリジンに対して許可
# CORS(app)
# @app.route('/image')
# def main():
#     image_path="cats.jpeg"
#     result_image = predict(model, image_path)
#     return send_image(result_image)

# def send_image(result_image):
#     # メモリにJPEGとして保存
#     img_io = io.BytesIO()
#     Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).save(img_io, 'JPEG')
#     img_io.seek(0)

#     return send_file(img_io, mimetype='image/jpeg')

# def predict(model: YOLO, image_path):  
#     # Perform object detection on an image
#     # model(image_path)[0].save()
#     return model(image_path)[0].plot()


from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO

import base64

app = Flask(__name__)
CORS(app)

#server立ち上がったかチェック用、デプロイすると「/」に１番にアクセスする
@app.route('/', methods=['GET'])
def index():
    return 'Flask server is up.'

# model = YOLO("yolo11n.pt")
# model = YOLO("best.pt")
# print(model.info())

# ✅ モデルはまだ読み込まない、遅延読み込みモード
model = None

@app.route('/predict', methods=['POST'])
def predict_route():
    global model
    if model is None:
        model = YOLO("best.pt")  # 初回リクエスト時だけ読み込む
        print(model.info())
    
    if 'image' not in request.files:
        return '画像が送信されていません', 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    
    # 推論とラベル抽出
    result = model(image_bgr)[0]
    labels = [model.names[int(box.cls[0])] for box in result.boxes]

    # 可視化画像の生成
    result_img = result.plot()

    # OpenCV(BGR)画像をJPEGに変換し、base64エンコード
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'labels': labels,
        'image': img_base64
    })
    
    
    # # YOLO推論
    # result_image = model_predict(model, image_bgr)
    # return send_result_image(result_image)

def model_predict(model: YOLO, image_bgr):
    result = model(image_bgr)[0]
    return result.plot()  # 検出結果画像（np.ndarray）

def send_result_image(image_bgr):
    img_io = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# 🔽 この2行を忘れずに！

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# if __name__=='__main__':
#     main()
    
    


# from flask import Flask, send_file
# from flask_cors import CORS  # 追加

# app = Flask(__name__)

# # CORS を全てのオリジンに対して許可
# CORS(app)
# print("bbbbbbbbbbbbbbbbbbb")

# @app.route('/image')
# def send_image():
#     # 画像ファイルのパスを指定
#     image_path = './tameshi.jpeg'  # 実際の画像ファイルのパスを指定
#     print("画像を送信しています")
#     return send_file(image_path, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)




# from flask import Flask, request, send_file
# from flask_cors import CORS
# from PIL import Image
# import io

# app = Flask(__name__)

# # CORS を全てのオリジンに対して許可
# CORS(app)

# @app.route('/flip-image', methods=['POST'])
# def flip_image():
#     # リクエストから画像を取得
#     if 'image' not in request.files:
#         return "No image part", 400

#     image_file = request.files['image']
    
#     # 画像ファイルが送信されていない場合
#     if image_file.filename == '':
#         return "No selected file", 400

#     try:
#         # 画像を開く
#         image_file = "./tameshi.jpeg"
#         img = Image.open(image_file)
        
#         # 180度回転 (expand=Trueを指定して画像の切り取りを防ぐ)
#         flipped_img = img.rotate(180, expand=True)

#         # バッファに反転した画像を保存
#         img_io = io.BytesIO()
#         flipped_img.save(img_io, 'JPEG')
#         img_io.seek(0)

#         # 反転した画像をレスポンスとして送信
#         return send_file(img_io, mimetype='image/jpeg')
    
#     except Exception as e:
#         return f"エラーが発生しました: {str(e)}", 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
