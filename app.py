from flask import Flask, render_template, Response, jsonify, request, json
import os
from datetime import datetime
import sys
import python_scripts.GAN_heightmaps as GAN_heightmaps
import base64
import cv2
from python_scripts.quadtrees import compress_from_input

app = Flask(__name__)

global heightmap_model
heightmap_model = None

def log_visitor():
    visitor_ip = request.remote_addr
    visitor_requested_path = request.full_path
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")

    pth = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(pth,"log.txt"), "a")
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
        heightmap_model = GAN_heightmaps.load_model()
    
    generated_img = GAN_heightmaps.generate_heightmap(heightmap_model)
    success, return_img = cv2.imencode(".png", generated_img)
    return_img = return_img.tobytes()
    return jsonify({"img":str(base64.b64encode(return_img))})

@app.route('/img_to_hierarchy')
def img_to_hierarchy():
    img = request.args.get('img')
    criteria = request.args.get('metric')
    criteria_value = float(request.args.get('metricValue'))
    upscaling_method = request.args.get('upscalingMethod')
    downscaling_method = request.args.get('downscalingMethod')
    min_chunk = int(request.args.get('minChunk'))
    max_downscale = int(request.args.get('maxDownscale'))

    img_upscaled, img_upscaled_debug, img_upscaled_point, final_psnr, final_mse, final_mre = \
    compress_from_input(img, criteria, criteria_value,
    upscaling_method, downscaling_method, 
    min_chunk, max_downscale)

    img_upscaled = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2BGR)
    img_upscaled_debug = cv2.cvtColor(img_upscaled_debug, cv2.COLOR_RGB2BGR)
    img_upscaled_point = cv2.cvtColor(img_upscaled_point, cv2.COLOR_RGB2BGR)
    _, img_upscaled = cv2.imencode(".png", img_upscaled)
    _, img_upscaled_debug = cv2.imencode(".png", img_upscaled_debug)
    _, img_upscaled_point = cv2.imencode(".png", img_upscaled_point)
    to_return = {
        "img_upscaled":str(base64.b64encode(img_upscaled)),
        "img_upscaled_debug":str(base64.b64encode(img_upscaled_debug)),
        "img_upscaled_point":str(base64.b64encode(img_upscaled_point)),
        "psnr":final_psnr,
        "mse":final_mse,
        "mre":final_mre
    }    
    return jsonify(to_return)
    

if __name__ == '__main__':
    #app.run(host='127.0.0.1',debug=True,port="12345")
    app.run(host='0.0.0.0',debug=False,port="80")