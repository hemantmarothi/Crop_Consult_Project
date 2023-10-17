import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from tensorflow.keras.optimizers import SGD
from werkzeug.utils import secure_filename
from PIL import Image  # Import the PIL library


app = Flask(__name__)

upload_counter = 1
# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get the full path to the directory where your script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Specify the full path to the model file
model_path = os.path.join(script_dir, 'citrus.h5')

# Define a function for image preprocessing (same as before)


def preprocess_image(img_data):
    # Load the image from binary data and resize it to the target size expected by the model
    print("bytes IO successful")
    img = image.load_img(img_data, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Normalize the image data
    img_array = img_array / 255.0
    # Expand the dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)[0]
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model(model_path)
        img_file = request.files['image']
        unique_filename = f"{upload_counter:04d}_{secure_filename(img_file.filename)}"
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
        print("image received")
        # Ensure the file is an image
        if img_file.content_type.startswith('image'):
            img = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
            predictions = model.predict(np.expand_dims(img, axis=0))
            Categories=['healthy','multiple_disease','rust','scab']
            disease = Categories[np.argmax(predictions)]
            return render_template("output.html", name = disease, img_file = unique_filename)
        else:
            return jsonify({'result': 'error', 'message': 'Invalid image file'}), 400

    except Exception as e:
        return jsonify({'result': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
