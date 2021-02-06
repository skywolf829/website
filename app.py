
from flask import Flask, render_template, Response, jsonify, request, json

app = Flask(__name__)

@app.route('/')
def index():
    a = render_template('index.html')
    print(a)
    return render_template('index.html')

@app.route('/projects')
def pjrojects():
    return render_template('/pages/projects.html')

@app.route('/teaching')
def teaching():
    return render_template('/pages/teaching.html')

@app.route('/CV')
def CV():
    return render_template('/documents/CVJan2020.pdf')

if __name__ == '__main__':
    #app.run(host='127.0.0.1',debug=True,port="12345")
    app.run(host='0.0.0.0',debug=False,port="80")