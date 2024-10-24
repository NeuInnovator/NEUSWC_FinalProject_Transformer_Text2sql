from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from engine import get_answer

app = Flask(__name__)
CORS(app)  # 允许跨域请求

dataset_path = r'./model'
# 路由：主页，加载 index.html
@app.route('/')
def home():
    return render_template('index.html')

# 路由：返回数据集名称
@app.route('/get_datasets', methods=['GET'])
def get_datasets():
    datasets = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    return jsonify(datasets)

# 路由：返回某数据集下的模型文件
@app.route('/get_models/<dataset>', methods=['GET'])
def get_models(dataset):
    model_path = os.path.join(dataset_path, dataset)
    if not os.path.exists(model_path):
        return jsonify([])  # 如果路径不存在，返回空列表
    models = [file for file in os.listdir(model_path) if file.endswith('.ckpt')]
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端提交的数据
    data = request.json
    question = data.get('question')
    model_path = dataset_path + '/' + data.get('model')
    model = data.get('model').split('/')[0]
    answer = get_answer(model_path, question, model)
    return jsonify({'answer': answer})


# 启动 Flask 应用
if __name__ == '__main__':
    app.run(port=5000, debug=True)
