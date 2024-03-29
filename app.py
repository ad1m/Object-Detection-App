__author__ = 'Adamlieberman'
from flask import Flask, render_template, request,flash
from object_detection.utils import *
from object_detection import Model_RUNNER
app = Flask(__name__)
import os
from os import listdir
from os.path import isfile, join
import time
import requests
import json


app.config['UPLOAD_FOLDER'] = 'test_images/images'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #SO WE DO NOT CACHE STATIC FILES LIKE IMAGES IN THE STATIC DIRECTORY
app.secret_key = 'blahblahuploaderblahhhh'
#index has a form that links to /classify_upload then we go below and handle what happens there and return stuff



def perform_update(count,room_name):
    ROOM_ID = '595524f783325f0d12d7dab1'
    BASE_URL = 'http://hvachack.cloud.rnoc.gatech.edu/'
    UPDATE_URL = BASE_URL + 'api/v1/rooms/' + ROOM_ID
    data = {"location": room_name, "numPeople": str(count), "currentTemperature":"0"}
    data_json = json.dumps(data)
    response = requests.put(UPDATE_URL, data=data_json, headers={"Content-Type": "application/json"})
    return



def does_file_exist_in_dir(path):
    return any(isfile(join(path, i)) for i in listdir(path))

@app.route('/')
def index():
    if does_file_exist_in_dir('static/images'):
        print('cleared_images')
        os.remove('static/images/image1d.jpg')
    else:
        print('no images to clear')
    if does_file_exist_in_dir('test_images/images'):
        os.remove('test_images/images/image1.png')
    return render_template('index.html')

@app.route('/classify_upload',methods=['GET','POST'])
def index2():
    if request.method == 'POST':
        f = request.files['file']
        f.filename = 'image1.png'
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
    #imagefile.save('image1.png')
        #return '<center><h1>successful upload</h1></center>'
        #message = '<center><font color="green"><h1>Successful Upload!</h1></font></center>'
        #flash(message)
        count = Model_RUNNER.run_model('test_images/images/image1.png')
        perform_update(count,'Klaus 1256')
        print('******------')
        print(count)
        millis = int(round(time.time() * 1000)) #A trick to prevent image static page caching
        return_image  = '<center><br><br><img src="static/images/image1d.jpg?"'+str(millis)+' style="width:75%;height:75%"/></center>'

        #return ('<center><br><br><img src="static/images/image1d.jpg?"'+str(millis)+'/></center>')
        return render_template('result.html',count=count,return_image=return_image)
    else:
        message = '<center><font color="red">Upload Failed Please Try Again</font></center>'
        flash(message)
        return render_template('index.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run(debug=True,port=5000)