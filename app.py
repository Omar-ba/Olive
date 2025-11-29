import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import sys
import base64
import io
import numpy as np
from flask_cors import CORS
# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model

app = Flask(__name__)
CORS(app)
# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODEL_CONFIGS = {
    'disease': {
        'model_path': 'C:\\Users\\P16v GEN1\\Desktop\\maching learning\\PROGET\\Olive\\Olive\\models\\disease_model.pth',
        'classes': ['disease' , 'healthy'],  # Update with your actual classes
        'input_size': 224
    },
    'tree': {
        'model_path': 'C:\\Users\\P16v GEN1\\Desktop\\maching learning\\PROGET\\Olive\\Olive\\models\\tree_model.pth',
        'classes': ['olive', 'non_olive'],  # Update with your actual classes
        'input_size': 224
    }
}

# Global model instances
models = {}


def load_models():
    """Load both models into memory"""
    try:
        # Load disease model
        if os.path.exists(MODEL_CONFIGS['disease']['model_path']):
            disease_model = get_model(len(MODEL_CONFIGS['disease']['classes']))
            disease_model.load_state_dict(torch.load(MODEL_CONFIGS['disease']['model_path'], map_location=device))
            disease_model.to(device)
            disease_model.eval()
            models['disease'] = disease_model
            print("✅ Disease model loaded successfully")
        else:
            print("❌ Disease model file not found")

        # Load tree model
        if os.path.exists(MODEL_CONFIGS['tree']['model_path']):
            tree_model = get_model(len(MODEL_CONFIGS['tree']['classes']))
            tree_model.load_state_dict(torch.load(MODEL_CONFIGS['tree']['model_path'], map_location=device))
            tree_model.to(device)
            tree_model.eval()
            models['tree'] = tree_model
            print("✅ Tree model loaded successfully")
        else:
            print("❌ Tree model file not found")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, input_size=224):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def preprocess_image_from_base64(base64_string, input_size=224):
    """Preprocess image from base64 string"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Remove data URL prefix if present
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]

    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def predict_image(model, image_tensor, classes):
    """Make prediction using the model"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        return {
            'predicted_class': classes[predicted.item()],
            'confidence': round(confidence.item() * 100, 2),
            'all_predictions': {
                classes[i]: round(probabilities[0][i].item() * 100, 2)
                for i in range(len(classes))
            }
        }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'device': str(device)
    })


# NEW ENDPOINT FOR BASE64 IMAGES
@app.route('/predict', methods=['POST'])
def predict_base64():
    """Predict using base64 image - for frontend compatibility"""
    try:
        data = request.get_json()

        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        base64_string = data['image_base64']

        # Use disease model by default (you can modify this logic)
        if 'disease' not in models:
            return jsonify({'error': 'Disease model not loaded'}), 500

        # Preprocess and predict
        image_tensor = preprocess_image_from_base64(base64_string, MODEL_CONFIGS['disease']['input_size'])
        result = predict_image(
            models['disease'],
            image_tensor,
            MODEL_CONFIGS['disease']['classes']
        )

        return jsonify({
            'prediction': f"{result['predicted_class']} (Confidence: {result['confidence']}%)",
            'details': result
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict/disease', methods=['POST'])
def predict_disease():
    """Predict leaf disease"""
    return predict_handler('disease')


@app.route('/api/predict/tree', methods=['POST'])
def predict_tree():
    """Predict tree type (olive/non-olive)"""
    return predict_handler('tree')


@app.route('/api/predict/both', methods=['POST'])
def predict_both():
    """Predict using both models"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = {}

            # Process with disease model
            if 'disease' in models:
                image_tensor = preprocess_image(filepath, MODEL_CONFIGS['disease']['input_size'])
                results['disease'] = predict_image(
                    models['disease'],
                    image_tensor,
                    MODEL_CONFIGS['disease']['classes']
                )

            # Process with tree model
            if 'tree' in models:
                image_tensor = preprocess_image(filepath, MODEL_CONFIGS['tree']['input_size'])
                results['tree'] = predict_image(
                    models['tree'],
                    image_tensor,
                    MODEL_CONFIGS['tree']['classes']
                )

            # Clean up
            os.remove(filepath)

            return jsonify(results)

        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


def predict_handler(model_type):
    """Generic prediction handler for individual models"""
    if model_type not in models:
        return jsonify({'error': f'{model_type} model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            image_tensor = preprocess_image(filepath, MODEL_CONFIGS[model_type]['input_size'])
            result = predict_image(
                models[model_type],
                image_tensor,
                MODEL_CONFIGS[model_type]['classes']
            )

            # Clean up
            os.remove(filepath)

            return jsonify(result)

        except Exception as e:
            # Clean up in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Loading models...")
    load_models()
    print("✅ Starting Flask server...")
    app.run(debug=True, host='localhost', port=800)  # Changed to port 8000 to match frontend