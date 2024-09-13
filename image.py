from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io
import requests
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load DELF model
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

def preprocess_image(image, size=(256, 256)):
    """Load and preprocess the image."""
    image = image.convert('RGB')
    image = image.resize(size)  # Resize to match model input
    return image

def extract_features(image):
    """Extract features from an image using DELF model."""
    np_image = np.array(image)
    print(f"Image shape: {np_image.shape}")
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)
    result = delf(
        image=float_image,
        score_threshold=tf.constant(100.0),
        image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
        max_feature_num=tf.constant(1000))
    descriptors = result['descriptors'].numpy()
    print(f"Extracted descriptors shape: {descriptors.shape}")
    return descriptors


def fetch_image_from_url(image_url):
    """Fetch an image from a URL."""
    response = requests.get(image_url)
    response.raise_for_status()  # Raise an error if the request failed
    return Image.open(io.BytesIO(response.content))

def match_uploaded_image(uploaded_image, base_url, image_names):
    """Match the uploaded image with remote images from the base URL and return the matched image and similarity score."""
    uploaded_image = preprocess_image(uploaded_image)
    uploaded_features = extract_features(uploaded_image)
    
    if uploaded_features.size == 0:
        return None, 0

    uploaded_features = uploaded_features.mean(axis=0)  # Mean of features

    sample_images = []
    sample_features = []

    # Process all sample images
    for image_name in image_names:
        image_url = f'{image_name}'
        try:
            image = fetch_image_from_url(image_url)
            image = preprocess_image(image)
            features = extract_features(image)
            
            if features.size == 0:
                continue
            
            features = features.mean(axis=0)  # Mean of features
            sample_images.append(image_url)
            sample_features.append(features)
        except Exception as e:
            print(f'Error fetching image {image_url}: {e}')

    if not sample_images:
        return None, 0

    # Compare features
    similarities = []
    for features in sample_features:
        similarity = cosine_similarity([uploaded_features], [features])
        similarities.append(similarity[0][0])

    # Find the best match
    best_match_index = np.argmax(similarities)
    best_match_image = sample_images[best_match_index]
    best_match_similarity = similarities[best_match_index]

    return best_match_image, best_match_similarity


@app.route('/match', methods=['POST'])
def match_image():
    """API endpoint to match uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        uploaded_image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Get user_id from the form data
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400

    # Get image names from the form data
    image_names = [value for key, value in request.form.items() if key.startswith('image_names[')]
    # return jsonify(image_names), 200
    if not image_names:
        return jsonify({'error': 'No image names provided'}), 400

    base_url = f'https://buybestthemes.com/mobile_app_api/tuned_ink/storage/app/public/tattoos/user_{user_id}'
    matched_image_path, similarity_score = match_uploaded_image(uploaded_image, base_url, image_names)

    # Return the result with similarity percentage
    if similarity_score < 0.70:
        return jsonify({'matched_image_path': 'No matched image found', 'similarity_score': similarity_score * 100}), 500

    return jsonify({
        'matched_image_path': matched_image_path,
        'similarity_score': similarity_score * 100  # Convert to percentage
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)

