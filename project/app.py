from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import os
import uuid
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the saved model
model = load_model(r'A:\face_disease\skin_disease_model.h5')

# Define class labels
classes = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczema', 'Rosacea']

# Folder to save uploaded or captured images temporarily
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

camera = cv2.VideoCapture(0)  # Initialize the webcam


def generate_frames():
    """Generate video frames for OpenCV stream."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Route for streaming video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    predicted_label = None
    image_path = None

    if request.method == 'POST':
        if 'file' in request.files:
            # File upload logic
            file = request.files['file']
            if file and file.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(image_path)
        elif 'capture' in request.form:
            # Capture photo logic
            success, frame = camera.read()
            if success:
                image_name = f"{uuid.uuid4().hex}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
                cv2.imwrite(image_path, frame)

        if image_path:
            # Process the uploaded or captured image
            image = Image.open(image_path)
            input_shape = (150, 150)  # Input shape of the model
            test_image = image.resize(input_shape)
            test_image_array = np.array(test_image)
            test_image_array = np.expand_dims(test_image_array, axis=0)
            test_image_array = test_image_array / 255.0

            # Predict
            predictions = model.predict(test_image_array)
            predicted_class = np.argmax(predictions)
            predicted_label = classes[predicted_class]
            prediction = predictions[0].tolist()

            # Save and display the result using Matplotlib
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Predicted: {predicted_label}")
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'result.png'))

    return render_template('index.html', prediction=prediction, predicted_label=predicted_label,
                           image_path=image_path, classes=classes)


if __name__ == '__main__':
    app.run(debug=True)
