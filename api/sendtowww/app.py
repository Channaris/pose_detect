import configparser
from flask import Response
from flask import Flask
from flask import render_template
from imutils.video import VideoStream
import sys
sys.path.append('D:\\work\\braiven\\punching\\mediapipe\\pose_detect\\pyimg\\pyimagesearch\\motion_detection')
from singlemotiondetector import SingleMotionDetector

config = configparser.ConfigParser()
config.read('D:\\work\\braiven\\punching\\mediapipe\\pose_detect\\config.ini')

app = Flask(__name__)
@app.route('/')

def home():
    return "Hello My First Flask Project"

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)

	return 555 
	# return Response(generate(),	mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)


