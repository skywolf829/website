from flask import Flask, render_template, Response, jsonify, request, json
import os
from datetime import datetime
import sys
import python_scripts.GAN_heightmaps as GAN_heightmaps
import base64
import cv2

app = Flask(__name__)

global heightmap_model
heightmap_model = None

def log_visitor():
    visitor_ip = request.remote_addr
    visitor_requested_path = request.full_path
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")

    f = open("log.txt", "a")
    f.write(dt + ": " + str(visitor_ip) + " " + str(visitor_requested_path) + "\n")
    f.close()

@app.route('/')
def index():
    log_visitor()
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/index_body.html') + \
           "</body></html>" 

@app.route('/projects')
def project():
    log_visitor()
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/projects_body.html') + \
           "</body></html>" 

@app.route('/projects/<pagename>')
def project_pages(pagename=None):
    log_visitor()
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
    log_visitor()
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/teaching_body.html') + \
           "</body></html>" 

@app.route('/teaching/<pagename>')
def teaching_pages(pagename=None):
    log_visitor()
    return '<!DOCTYPE html><html lang="en">' + \
            "<head>" + \
            render_template('head_items.html') + \
            "</head><body>" + \
            render_template('sidebar.html') + \
            render_template('/pages/teaching/'+pagename) + \
           "</body></html>" 

@app.route('/get_heightmap')
def get_generated_image():
    global heightmap_model
    if heightmap_model is None:
        heightmap_model = GAN_heightmaps.load_latest_model()
    
    generated_img = GAN_heightmaps.generate_heightmap(heightmap_model)
    success, return_img = cv2.imencode(".png", generated_img)
    return_img = return_img.tobytes()
    return jsonify({"img":str(base64.b64encode(return_img))})

if __name__ == '__main__':
    #app.run(host='127.0.0.1',debug=True,port="12345")
    app.run(host='0.0.0.0',debug=False,port="80")