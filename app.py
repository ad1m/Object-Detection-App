__author__ = 'Adamlieberman'
from flask import Flask, render_template, request,flash
from object_detection.utils import *
from object_detection import Model_RUNNER
app = Flask(__name__)
import os
from os import listdir
from os.path import isfile, join
import time
app.config['UPLOAD_FOLDER'] = 'test_images/images'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #SO WE DO NOT CACHE STATIC FILES LIKE IMAGES IN THE STATIC DIRECTORY
app.secret_key = 'blahblahuploaderblahhhh'
#index has a form that links to /classify_upload then we go below and handle what happens there and return stuff

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
        Model_RUNNER.run_model('test_images/images/image1.png')
        millis = int(round(time.time() * 1000)) #A trick to prevent image static page caching

        return ('<center><br><br><img src="static/images/image1d.jpg?"'+str(millis)+'/></center>')
    else:
        message = '<center><font color="red">Upload Failed Please Try Again</font></center>'
        flash(message)
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True,port=5000)