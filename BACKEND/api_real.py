from flask import Flask, request, jsonify
import service  # 导入你的service.py
app = Flask(__name__)

a = 1

def initialize():
    global a
    a = a + 1

# 执行预设代码
initialize()


@app.route('/run_service', methods=['POST'])
def run_service():
    data = request.json
    input_data = data.get('input')
    
    result = service.self_add(int(input_data)) + a
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)