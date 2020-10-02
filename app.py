
import flask, os
from flask import request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf, datetime
import easyocr, string
from pre_process import *
from manage_db import *


app = flask.Flask(__name__)

CORS(app)

# Date time
date_time = str(datetime.datetime.now()).split('.')[0]


# load model
tf.keras.backend.clear_session()
model = tf.saved_model.load(os.path.join(os.getcwd(),'model/inference_graph/saved_model'))

# class label
labelmap_path = os.path.join(os.getcwd(), 'label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)


UPLOAD_FOLDER = os.path.join(os.getcwd(), "static/upload")
DETECT_FOLDER = os.path.join(os.getcwd(), "static/detect")


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

# CLEAR the image
def delete_img(img_paths):
	paths = os.listdir(img_paths)
	if len(paths) > 1:
  		for i in paths:
  			os.remove(os.path.join(img_paths, i))


# create ocr object
reader = easyocr.Reader(['en'])

# OCR
def ocr(img):
	no_plate = []

	file_path = os.path.join(os.getcwd(), 'static/upload/'+str(img))
	output = reader.readtext(file_path, allowlist=string.ascii_uppercase[:26] + '1234567890')

	for i in  range(len(output)):
		letter_len = output[i][-2]
		if len(letter_len) > 2:
			no_plate.append(letter_len)

	return no_plate


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	if request.method == 'POST':

		delete_img(os.path.join(os.getcwd(), 'static/detect'))
		delete_img(os.path.join(os.getcwd(), 'static/upload'))

		f = request.files['file']
		filename = secure_filename(f.filename)
		print(filename)

		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		print(filepath)
		f.save(filepath)
		
	      
		image_np = load_image_into_numpy_array(filepath)
		output_dict = run_inference_for_single_image(model, image_np)

		if output_dict['detection_scores'][0] > 0.80:
			vis_util.visualize_boxes_and_labels_on_image_array(
		    image_np,
		    output_dict['detection_boxes'],
		    output_dict['detection_classes'],
		    output_dict['detection_scores'],
		    category_index,
		    instance_masks=output_dict.get('detection_masks_reframed', None),
		    use_normalized_coordinates=True,
		    line_thickness=4)

			img = Image.fromarray(image_np)
			img.save(os.path.join(DETECT_FOLDER, filename))

			# cal orc
			detected_ocr = ocr(filename)

		else:
			print('Please upload Nother image')
			img = Image.fromarray(image_np)
			img.save(os.path.join(DETECT_FOLDER, filename))


		return render_template('index.html', display_detection = filepath, 
			fname = filename, ocr = detected_ocr, datetime = date_time)



@app.route('/savecomplain', methods = ['GET', 'POST'])
def savecomplain():
	if request.method == 'POST':
		license_no = request.form['licenseNo']
		complain = request.form['complain']
		date_n_time = request.form['datetime']

		# create table
		create_table()

		# add data to DB
		data_added = add_data(license_no, complain, date_n_time)

		if data_added:
			send_messages = 'Successfully Regsitered..!!'
		else:
			send_messages = 'Something went wrong..!! Could not Register.'


		
		fetch_detail = fetch_data(license_no)

		# fetch each detail from tuple within list
		unpack_detail = []
		for data_list in reversed(fetch_detail):
			unpack_detail.append(list(data_list))


	return render_template('index.html', send_message = send_messages, detail_with_id = unpack_detail)



if __name__ == '__main__':
	app.run(debug=True)