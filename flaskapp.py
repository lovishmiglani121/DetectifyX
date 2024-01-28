from flask import Flask, render_template, Response,jsonify,request,session, send_from_directory

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
from yolo_img import draw_bounding_boxes
import os


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from yolo_webcam import detect_objects_by_webcam
app = Flask(__name__)

app.config['SECRET_KEY'] = 'lm1697'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['PROCESSED_FOLDER'] = 'static/processed'

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def processImg(pictureName):
    model_path = 'yolov8s.pt'
    image_path = './static/files/'

def generate_frames(path_x = ''):
    yolo_output = detect_objects_by_webcam(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = detect_objects_by_webcam(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])
def loading():
    session.clear()
    return render_template('loading.html')

@app.get("/home")
def home():
    return render_template('index.html')
@app.get("/home/display")
def display():
    return render_template('display.html')
# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/demo/videoimg", methods=['GET','POST'])

def withvideoimg():
    session.clear()
    return render_template('demohomeimg.html')

@app.route("/demo/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('videoui.html')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():

    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/demo/video', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('demovideo.html', form=form)

@app.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/demo-image", methods=['GET', 'POST'])
def demoImg():
    return render_template('demo-image.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            # Save the file to the static/files directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            object_detection = draw_bounding_boxes(file_path)
            # Now you can use the saved file path for further processing
            # results, img = (file_path)
            
            # Return the filename to construct the URL
            return render_template('demo-image.html', image_path = object_detection)

    return 'File upload failed.'

@app.route('/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)