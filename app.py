from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

def predict(text):
    return "Okay this works"

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    output = predict(user_input)
    return jsonify({'output': output})
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
