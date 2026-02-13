from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

# Incarcam modelul si clasele
model = tf.keras.models.load_model('model/plant_model_final.keras')
with open('model/class_names.json', 'r') as f:
    class_names = json.load(f)

# Recomandari pentru fiecare boala
recommendations = {
    'Pepper__bell___Bacterial_spot': 'Aplicați fungicide cu cupru. Evitați udarea frunzelor.',
    'Pepper__bell___healthy': 'Planta este sănătoasă! Continuați îngrijirea normală.',
    'Potato___Early_blight': 'Aplicați fungicide preventiv. Îndepărtați frunzele afectate.',
    'Potato___Late_blight': 'Tratați urgent cu fungicide sistemice. Boală gravă!',
    'Potato___healthy': 'Planta este sănătoasă! Continuați îngrijirea normală.',
    'Tomato_Bacterial_spot': 'Aplicați bactericide cu cupru. Evitați ploile artificiale.',
    'Tomato_Early_blight': 'Îndepărtați frunzele afectate. Aplicați fungicide.',
    'Tomato_Late_blight': 'Tratați urgent! Aplicați fungicide sistemice imediat.',
    'Tomato_Leaf_Mold': 'Îmbunătățiți ventilația. Aplicați fungicide specifice.',
    'Tomato_Septoria_leaf_spot': 'Îndepărtați frunzele afectate. Aplicați fungicide cu cupru.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Aplicați acaricide. Spălați frunzele cu apă.',
    'Tomato__Target_Spot': 'Aplicați fungicide. Evitați umiditatea excesivă.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Combateți insectele vectoare. Nu există tratament direct.',
    'Tomato__Tomato_mosaic_virus': 'Îndepărtați plantele afectate. Dezinfectați uneltele.',
    'Tomato_healthy': 'Planta este sănătoasă! Continuați îngrijirea normală.'
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nicio imagine încărcată'})

    file = request.files['image']

    # Procesare imagine
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predictie
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    disease = class_names[class_idx]

    return jsonify({
        'disease': disease.replace('_', ' '),
        'confidence': round(confidence, 2),
        'recommendation': recommendations.get(disease, 'Consultați un specialist.')
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)