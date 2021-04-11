from flask import Flask, render_template, request, jsonify
import os
from reco_1 import myfuc_PR

app = Flask(__name__)

def func(text1):
    #text1 = text1.upper()
    #text1 = text1 + os.linesep + "Prod1" + os.linesep + "prod2"
    #myfuc_PR(text1)
    temp = myfuc_PR(text1)
    print(temp)
    return (temp)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/join', methods = ['GET', 'POST'])
def my_form_post():
    text1 = request.form['text1']
    word = request.args.get('text1')
    res = func(text1)
    #res = "Done"
    #print(res)
    res_1 = {"output": res}
    res_1 = {str(key):value for key, value in res_1.items()}
    return jsonify(result=res_1)

if __name__ == '__main__':
    app.run(debug=True)
