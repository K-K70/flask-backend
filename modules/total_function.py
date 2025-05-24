from flask import Flask, jsonify,make_response,request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64

def upload_file(app):
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 画像をBase64でエンコードする
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return jsonify({
            'message': 'File successfully uploaded',
            'data': encoded_string
        }), 200
