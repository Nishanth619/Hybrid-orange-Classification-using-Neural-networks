from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the paths to the models
models = {
    "InceptionV3": r'C:\Users\ADMIN\PycharmProjects\Citrus\hybrid_orange_inception_v3(80).h5',
    "MobileNet": r'C:\Users\ADMIN\PycharmProjects\Citrus\hybrid_orange_mobilenet(70).keras',
    "CNN": r'C:\Users\ADMIN\PycharmProjects\Citrus\optimized_cnn_model(70-30).keras',
}

# Define the input sizes for the models
input_sizes = {
    "InceptionV3": (299, 299),
    "MobileNet": (224, 224),
    "CNN": (299, 299)  # Adjust to your CNN model's input size
}

# Define the class names
class_names = ['LIMON_CRIOLLO', 'LIMON_MANDARINO', 'LIMON_TAHITI', 'MANDARINA_ISRAELI', 'MANDARINA_PIELDESAPO', 'NARANJA_VALENCIA', 'TANGELO', 'TORONJA']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1] range
    return img_array

def classify_image(model, img_path, input_size):
    img_array = load_and_preprocess_image(img_path, target_size=input_size)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    confidence = predictions[0][predicted_class[0]]
    return predicted_class_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_model_name = request.form['model']

        if 'image' not in request.files:
            return "No file part"

        file = request.files['image']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load the selected model
            model_path = models.get(selected_model_name)
            if not model_path:
                return "Selected model not found"

            model = load_model(model_path)

            # Classify the image
            input_size = input_sizes.get(selected_model_name, (299, 299))
            predicted_class_name, confidence = classify_image(model, file_path, input_size)

            return render_template('result.html', model_name=selected_model_name,
                                   predicted_class_name=predicted_class_name, confidence=confidence)

    return render_template('index.html', models=models.keys())

if __name__ == '__main__':
    app.run(debug=True)
