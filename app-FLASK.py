from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import numpy as np
from PIL import Image



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Define class labels
class_labels = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}
# Load the model
mod3 = 'Mob2_torch.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
model = torch.load(mod3, map_location=device)
model.eval()  

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)

    # Convert RGBA to RGB if the image has an alpha channel
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')

    # Resize and crop
    pil_image = pil_image.resize((256, 256))  # Resize to fixed size
    pil_image = pil_image.crop((16, 16, 240, 240))  # Center crop 224x224

    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)

    # Convert image to PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32)

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.exp(output)

    # Convert probabilities and indices to lists
    probabilities = probabilities.squeeze().numpy()
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_labels[predicted_class_idx]
    predicted_prob = round(probabilities[predicted_class_idx] * 100, 2)  # Multiply by 100 here

    # Convert probabilities to percentages
    total_prob = sum(probabilities)
    predicted_confidence = round(predicted_prob / total_prob * 1, 2)

    probs_percent = {class_labels[i]: round(prob / total_prob * 100, 2) for i, prob in enumerate(probabilities)}

    return predicted_class, predicted_confidence, probs_percent



@app.route('/')
def index():
    return render_template('index.html')

import os
import uuid

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Generate a unique ID for the image file
        img_id = str(uuid.uuid4())
        # Save the image file in the static/uploads folder with the unique ID
        filepath = os.path.join('static/uploads', img_id + '.png')
        file.save(filepath)

        # Make prediction
        predicted_class, predicted_confidence, probs_percent = predict(filepath, model)

        # Prepare prediction results
        prediction = {
            'class': predicted_class,
            'confidence': predicted_confidence,
            'probs': probs_percent,
            'image_name': os.path.basename(filepath)  # Change this line
        }

        return render_template('index.html', prediction=prediction)


    # Code will be added in Part 4
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

