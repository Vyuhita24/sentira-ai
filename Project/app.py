import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from config import *
from flexible_stress_model import FlexibleStressModel
from universal_preprocessing import UniversalPreprocessor

app = Flask(__name__)

# Load Model
print("Initializing Stress Model and Preprocessor...")
stress_model = FlexibleStressModel()

universal_model_path = os.path.join(BASE_DIR, "models", "universal", "universal_model_model.keras")
universal_config_path = os.path.join(BASE_DIR, "models", "universal", "universal_model_config.json")

model_loaded = stress_model.load_model(universal_model_path, universal_config_path)
if not model_loaded:
    print("Warning: Universal model not found. Please train it first.")

preprocessor = UniversalPreprocessor()


def summarize_probabilities(probabilities):
    items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top1_emotion, top1_prob = items[0]
    if len(items) > 1:
        top2_emotion, top2_prob = items[1]
    else:
        top2_emotion, top2_prob = top1_emotion, 0.0
    return top1_emotion, float(top1_prob), top2_emotion, float(top2_prob)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': bool(model_loaded)
    }), 200

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file:
            os.makedirs(DATA_DIR, exist_ok=True)
            allowed_extensions = {'.wav', '.webm', '.ogg', '.mp3', '.flac', '.m4a', '.mp4'}
            extension = os.path.splitext(file.filename or '')[1].lower()
            if extension not in allowed_extensions:
                extension = '.wav'

            filepath = os.path.join(DATA_DIR, f"temp_upload{extension}")
            file.save(filepath)

            # 1. Prediction with Universal Model (multi-window inference for stability)
            features_list = preprocessor.extract_features(
                filepath,
                use_augmentation=False,
                num_inference_windows=3
            )
            if not features_list:
                return jsonify({'error': 'Failed to extract audio features'}), 500

            predictions = [stress_model.predict_emotion(features) for features in features_list]
            all_emotions = list(stress_model.emotion_mapping.values())

            prob_accumulator = {emotion_name: 0.0 for emotion_name in all_emotions}
            for prediction in predictions:
                probs = prediction.get('all_probabilities', {})
                for emotion_name in all_emotions:
                    prob_accumulator[emotion_name] += float(probs.get(emotion_name, 0.0))

            num_predictions = max(1, len(predictions))
            averaged_probabilities = {
                emotion_name: prob_accumulator[emotion_name] / num_predictions
                for emotion_name in all_emotions
            }

            emotion, top_prob, second_emotion, second_prob = summarize_probabilities(averaged_probabilities)
            raw_confidence = top_prob

            probs = np.array(list(averaged_probabilities.values()), dtype=float)
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / probs.sum()

            margin = max(0.0, top_prob - second_prob)
            if probs.size > 1:
                entropy = float(-np.sum(probs * np.log(probs)))
                max_entropy = float(np.log(probs.size))
                certainty = max(0.0, 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0))
            else:
                certainty = top_prob

            confidence = float(np.clip(0.50 * top_prob + 0.35 * margin + 0.15 * certainty, 0.0, 0.97))

            # More granular stress level for UI with uncertainty gating.
            emotion_lower = str(emotion).lower()
            stress_emotions = {'angry', 'fearful', 'disgust', 'sad'}
            calm_emotions = {'happy', 'surprised', 'pleasant_surprise', 'pleasant_surprised', 'neutral', 'calm'}

            if emotion_lower in calm_emotions:
                stress_status = "Low Stress"
            elif emotion_lower in stress_emotions:
                stress_status = "High Stress" if (confidence >= 0.82 and margin >= 0.18) else "Medium Stress"
            else:
                stress_status = "Medium Stress"

            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'result': stress_status,
                'confidence': float(confidence),
                'emotion': emotion,
                'raw_confidence': raw_confidence,
                'top2': {
                    'primary': {'emotion': emotion, 'probability': float(top_prob)},
                    'secondary': {'emotion': second_emotion, 'probability': float(second_prob)}
                }
            })

    except Exception as e:
        print(f"Unhandled Exception in predict_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    # use_reloader=False prevents TensorFlow conflicts on Windows
    app.run(debug=True, use_reloader=False)
