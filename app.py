"""
Flask Web Application for Wine Classification
Production-ready deployment for Render
"""

from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model architectures
from models import LogisticRegressionModel, DecisionTreeLikeNN, SVMLikeNN

app = Flask(__name__)

# Configuration
MODEL_PATH = 'svm_nn_model.pt'  # Using best performing model
MODEL_TYPE = 'svm'  # Change to 'lr' or 'dt' if using different model
INPUT_DIM = 13
NUM_CLASSES = 3

# Wine class names
CLASS_NAMES = ['Cultivar 0', 'Cultivar 1', 'Cultivar 2']

# Feature names for the form
FEATURE_NAMES = [
    'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium',
    'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins',
    'Color Intensity', 'Hue', 'OD280/OD315', 'Proline'
]

# Feature ranges for validation (from wine dataset)
FEATURE_RANGES = {
    'alcohol': (11.0, 15.0),
    'malic_acid': (0.5, 6.0),
    'ash': (1.0, 4.0),
    'alcalinity_of_ash': (10.0, 30.0),
    'magnesium': (70.0, 162.0),
    'total_phenols': (0.5, 4.0),
    'flavanoids': (0.0, 6.0),
    'nonflavanoid_phenols': (0.1, 0.7),
    'proanthocyanins': (0.4, 4.0),
    'color_intensity': (1.0, 13.0),
    'hue': (0.4, 1.8),
    'od280': (1.0, 4.0),
    'proline': (250.0, 1700.0)
}


def load_model():
    """Load the trained model"""
    try:
        # Initialize model architecture
        if MODEL_TYPE == 'lr':
            model = LogisticRegressionModel(INPUT_DIM, NUM_CLASSES)
        elif MODEL_TYPE == 'dt':
            model = DecisionTreeLikeNN(INPUT_DIM, NUM_CLASSES)
        elif MODEL_TYPE == 'svm':
            model = SVMLikeNN(INPUT_DIM, NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
        
        # Load trained weights (CPU only for deployment)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully: {MODEL_TYPE.upper()}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load model once at startup
model = load_model()

if model is None:
    print("WARNING: Model failed to load. Application may not work correctly.")


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                          feature_names=FEATURE_NAMES,
                          class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for wine classification
    
    Expected JSON format:
    {
        "features": [13.2, 2.5, 2.3, 19.0, 105.0, 1.8, 2.2, 0.4, 1.5, 5.0, 1.0, 3.2, 1100.0]
    }
    
    Returns:
    {
        "success": true,
        "prediction": 0,
        "class_name": "Cultivar 0",
        "confidence": 0.95,
        "probabilities": [0.95, 0.03, 0.02],
        "all_classes": ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "features" in request body'
            }), 400
        
        features = data['features']
        
        # Validate input length
        if len(features) != INPUT_DIM:
            return jsonify({
                'success': False,
                'error': f'Expected {INPUT_DIM} features, got {len(features)}'
            }), 400
        
        # Validate feature values are numeric
        try:
            features = [float(f) for f in features]
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'All features must be numeric values'
            }), 400
        
        # Convert to tensor
        input_tensor = torch.FloatTensor([features])
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'class_name': CLASS_NAMES[prediction],
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities[0].tolist()],
            'all_classes': CLASS_NAMES
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment platforms"""
    model_status = 'loaded' if model is not None else 'not loaded'
    
    return jsonify({
        'status': 'healthy',
        'model': MODEL_TYPE,
        'model_status': model_status,
        'version': '1.0',
        'author': 'Modupe - Bioinformatics Masters'
    })


@app.route('/api/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': MODEL_TYPE,
        'input_dimension': INPUT_DIM,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'feature_names': FEATURE_NAMES,
        'feature_count': len(FEATURE_NAMES)
    })


@app.route('/api/example', methods=['GET'])
def example_prediction():
    """
    Example prediction with sample data
    Useful for testing the API
    """
    # Example wine features (typical Class 0 sample)
    example_features = [
        13.2,    # Alcohol
        2.77,    # Malic Acid
        2.51,    # Ash
        18.5,    # Alcalinity of Ash
        96.0,    # Magnesium
        1.09,    # Total Phenols
        0.52,    # Flavanoids
        0.86,    # Nonflavanoid Phenols
        1.69,    # Proanthocyanins
        5.8,     # Color Intensity
        0.48,    # Hue
        1.03,    # OD280/OD315
        415.0    # Proline
    ]
    
    # Make prediction
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        input_tensor = torch.FloatTensor([example_features])
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        return jsonify({
            'success': True,
            'example_features': example_features,
            'feature_names': FEATURE_NAMES,
            'prediction': int(prediction),
            'class_name': CLASS_NAMES[prediction],
            'probabilities': [float(p) for p in probabilities[0].tolist()],
            'note': 'This is an example prediction using typical wine features'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For local development
    import os
    port = int(os.environ.get('PORT', 5008)) 
    
    print("\n" + "=" * 60)
    print("🍷 Wine Classification API Starting...")
    print("=" * 60)
    print(f"\nModel Type: {MODEL_TYPE.upper()}")
    print(f"Model Status: {'Loaded ✓' if model else 'Not Loaded ✗'}")
    print(f"\nPort: {port}")
    print("\nEndpoints:")
    print(f"  - http://localhost:{port}/")
    print(f"  - http://localhost:{port}/predict")
    print(f"  - http://localhost:{port}/health")
    print(f"  - http://localhost:{port}/api/info")
    print(f"  - http://localhost:{port}/api/example")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)