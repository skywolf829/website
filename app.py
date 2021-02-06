
from flask import Flask, render_template, Response, jsonify, request, json

app = Flask(__name__)

@app.route('/')
def index():
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/index_body.html') + \
           "</body></html>" 

@app.route('/projects')
def project():
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/projects_body.html') + \
           "</body></html>" 

@app.route('/projects/<pagename>')
def project_pages(pagename=None):
    if(pagename == "CSE5542_finalproject.html"):        
        return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            render_template('/pages/projects/CSE5542_finalproject_head.html') + \
            "</head><body id='lab5-body' onload='webGLStart();'>" + \
            render_template('sidebar.html') + \
            render_template('/pages/projects/CSE5542_finalproject_body.html') + \
           "</body></html>" 

    else:
        return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/projects/'+pagename) + \
           "</body></html>" 

@app.route('/teaching')
def teaching():
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/teaching_body.html') + \
           "</body></html>" 

@app.route('/teaching/<pagename>')
def teaching_pages(pagename=None):
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/teaching/'+pagename) + \
           "</body></html>" 

if __name__ == '__main__':
    #app.run(host='127.0.0.1',debug=True,port="12345")
    app.run(host='0.0.0.0',debug=False,port="80")