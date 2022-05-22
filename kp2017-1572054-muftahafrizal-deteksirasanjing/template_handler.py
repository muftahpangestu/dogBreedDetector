# USAGE
# python simple_request.py

# import the necessary packages
import requests
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request,send_from_directory

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"

FILE_NAME = ""

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'upl'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

## functions
# check if uploaded file is in allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

## flask routes
# index
# upload form
@app.route("/")
def index():
    return render_template("index.html")



@app.route('/upload', methods=['POST'])
def upload():
	if request.method == 'POST':
		file=request.files['file']
		global model
		model = request.form['model']
		if file and allowed_file(file.filename):
			now = datetime.now()
			filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s.%s" % (now.strftime("%Y-%m-%d-%H-%M-%S-%f"), file.filename.rsplit('.', 1)[1]))
			file.save(filename)
			global IMAGE_PATH
			IMAGE_PATH=filename
			return redirect(url_for('result'))
		else:
			return render_template("exception.html")

@app.route('/result')
def result():
   # load the input image and construct the payload for the request
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image,"imagepath":IMAGE_PATH,"model":model}

	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()
	
	# ensure the request was sucessful
	if r["success"]:
		# loop over the predictions and display them
		arrResult=[]
		for (i, result) in enumerate(r["predictions"]):
			p = prediction(result["label"],round(result["probability"]*100,2))
			arrResult.append(p)
		if(r["dog"]==True):
			name=IMAGE_PATH.rsplit('/',1)[1]
			return render_template("dog.html", dir=name, data=arrResult,model=model)
		else:
			name=IMAGE_PATH.rsplit('/',1)[1]
			return render_template("non-dog.html",dir=name, data=arrResult,model=model)

    
@app.route('/show/<filename>')
def send_image(filename):
    return send_from_directory("upl",filename)


class prediction:
	def __init__(self, lbl, prob):
		self.label = lbl
		self.probability = prob



if __name__ == "__main__":
	port = int(os.environ.get('PORT', 9000))
	app.run(debug=True, port=port)
