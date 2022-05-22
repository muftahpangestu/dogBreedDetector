# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import keras
from keras.applications import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.preprocessing import image
from PIL import Image
import numpy as np
import flask
import io 



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")
	global modelXception
	modelXception = Xception(weights="static/model/0fc18241-ecc1-4c84-97f8-efc4e4f9dad8_Xception.h5")
	global modelVGG16
	modelVGG16 = VGG16(weights="static/model/4a05295f-317f-4348-b792-c6c44658e7f7_vgg16_weights_tf_dim_ordering_tf_kernels.h5")


def prepare_image(img, target):
	# if the image mode is not RGB, convert it
	if img.mode != "RGB":
		img = img.convert("RGB")

	# resize the input image and preprocess it
	img = img.resize(target)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = imagenet_utils.preprocess_input(img)

	# return the processed image
	return img

def dog_detector(pred):
	dog = np.argmax(pred)
	if((dog<=268)&(dog>=151)):
		return True
	else:
		return False

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			preds=""
			results=""
			select = flask.request.files["model"].read().decode("utf-8")


			if(select=='ResNet50'):
				# preprocess the image and prepare it for classification
				imageRead = flask.request.files["image"].read()
				imageRead = Image.open(io.BytesIO(imageRead))
				imageResNet = prepare_image(imageRead, target=(224, 224))
				# classify the input image and then initialize the list
				# of predictions to return to the client
				preds = model.predict(imageResNet)
				results = imagenet_utils.decode_predictions(preds)

			elif(select=='Xception'):
				from keras.applications.xception import preprocess_input, decode_predictions
				img = image.load_img(flask.request.files["imagepath"].read().decode("utf-8"), target_size=(299,299))
				img_arr = np.expand_dims(image.img_to_array(img), axis=0)
				x = preprocess_input(img_arr)
				preds = modelXception.predict(x)
				results=decode_predictions(preds)
			elif(select=="VGG16"):
				from keras.applications.vgg16 import preprocess_input, decode_predictions
				img = image.load_img(flask.request.files["imagepath"].read().decode("utf-8"), target_size=(224,224))
				img_arr = np.expand_dims(image.img_to_array(img), axis=0)
				x = preprocess_input(img_arr)
				preds = modelVGG16.predict(x)
				results=decode_predictions(preds)


			



			data["predictions"] = []

			dog=False

			#if(face_detector(imagepath)):
				#human = True
			if(dog_detector(preds)):
				dog=True


			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True
			data["dog"] = dog
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
