
from flask import Flask, render_template, Response, jsonify, request, json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('HTML_start.html') + \
        render_template('HTML_head.html') + \
        render_template('HTML_bodystart.html') + \
        render_template('HTML_sidebar.html') + \
        render_template('index_body.html') + \
        render_template('HTML_bodyend.html') + \
        render_template("HTML_end.html")

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