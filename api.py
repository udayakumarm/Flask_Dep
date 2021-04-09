from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

def func(text1):
    text1 = text1.upper()
    text1 = text1 + os.linesep + "Prod1" + os.linesep + "prod2"
    return text1

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/join', methods = ['GET', 'POST'])
def my_form_post():
    text1 = request.form['text1']
    word = request.args.get('text1')
    res = func(text1)
    result = {"output": res}
    
    result = {str(key):value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)