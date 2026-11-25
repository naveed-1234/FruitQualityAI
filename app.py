import os
import cv2
import numpy as np
import tensorflow as tf
from google import genai
from google.genai import types
from dotenv import load_dotenv
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from gradcam import generate_and_save_gradcam

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Google GenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing GenAI Client: {e}")
else:
    print("WARNING: GEMINI_API_KEY is missing in your .env file.")

SYSTEM_INSTRUCTION = (
    "You are an expert fruit and post-harvest produce assistant in a Fruit Quality Inspection app. "
    "Your SOLE purpose is to answer questions strictly about fruits (global varieties, ripeness, decay signs, "
    "shelf life, post-harvest storage, nutrition, vitamins). "
    "FORMATTING RULES: Provide concise, informative answers in 2 to 4 clear sentences. "
    "Do NOT use markdown symbols like asterisks (** or *) for bold or italic text; write in clean, plain readable text. "
    "CRITICAL RULE: If a user asks about anything that is NOT directly related to fruits or produce, politely decline: "
    "'I am a specialized Fruit AI Assistant. My sole purpose is to provide insights, storage advice, and nutritional details related to fruits. Please ask any fruit-related questions!'"
)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # Keep uploads below Cloud Run's 32 MiB HTTP/1 request limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)
FRUIT_CLASSES = ['Apple', 'Banana', 'Kiwi', 'Mango', 'Orange', 'Papaya', 'Peach', 'Pear', 'Pineapple', 'Strawberry']
QUALITY_CLASSES = ['Fresh', 'Overripe', 'Rotten']

# Load pre-trained models
print("Loading models...")
fruit_model = tf.keras.models.load_model('models/fruit_model.keras')
quality_model = tf.keras.models.load_model('models/quality_model.keras')
print("Models loaded successfully!")

NUTRITION_DATA = {
    'Apple': {'calories': 52, 'carbohydrates': '14g', 'protein': '0.3g', 'fiber': '2.4g', 'vitamins': ['Vitamin C (14% DV)', 'Potassium', 'Vitamin K'], 'benefits': 'Rich in antioxidants, promotes gut health.'},
    'Banana': {'calories': 89, 'carbohydrates': '23g', 'protein': '1.1g', 'fiber': '2.6g', 'vitamins': ['Vitamin B6 (20% DV)', 'Vitamin C (14% DV)', 'Potassium'], 'benefits': 'Provides quick energy, regulates blood pressure.'},
    'Kiwi': {'calories': 61, 'carbohydrates': '15g', 'protein': '1.1g', 'fiber': '3.0g', 'vitamins': ['Vitamin C (155% DV)', 'Vitamin K', 'Vitamin E'], 'benefits': 'Boosts immune system, aids digestion.'},
    'Mango': {'calories': 60, 'carbohydrates': '15g', 'protein': '0.8g', 'fiber': '1.6g', 'vitamins': ['Vitamin A (21% DV)', 'Vitamin C (60% DV)', 'Folate'], 'benefits': 'Supports eye health, strengthens immunity.'},
    'Orange': {'calories': 47, 'carbohydrates': '12g', 'protein': '0.9g', 'fiber': '2.4g', 'vitamins': ['Vitamin C (89% DV)', 'Thiamine', 'Folate'], 'benefits': 'High antioxidant content, boosts immunity.'},
    'Papaya': {'calories': 43, 'carbohydrates': '11g', 'protein': '0.5g', 'fiber': '1.7g', 'vitamins': ['Vitamin C (103% DV)', 'Vitamin A', 'Folate'], 'benefits': 'Contains papain for smooth protein digestion.'},
    'Peach': {'calories': 39, 'carbohydrates': '9.5g', 'protein': '0.9g', 'fiber': '1.5g', 'vitamins': ['Vitamin C (11% DV)', 'Vitamin A', 'Potassium'], 'benefits': 'Improves skin health, low calorie density.'},
    'Pear': {'calories': 57, 'carbohydrates': '15g', 'protein': '0.4g', 'fiber': '3.1g', 'vitamins': ['Vitamin C (7% DV)', 'Vitamin K', 'Copper'], 'benefits': 'High soluble dietary fiber, supports bowel health.'},
    'Pineapple': {'calories': 50, 'carbohydrates': '13g', 'protein': '0.5g', 'fiber': '1.4g', 'vitamins': ['Vitamin C (79% DV)', 'Manganese (46% DV)', 'B6'], 'benefits': 'Contains bromelain for digestion.'},
    'Strawberry': {'calories': 32, 'carbohydrates': '7.7g', 'protein': '0.7g', 'fiber': '2.0g', 'vitamins': ['Vitamin C (98% DV)', 'Manganese', 'Folate'], 'benefits': 'Controls blood sugar spikes, rich in antioxidants.'}
}

SHELF_LIFE_RULES = {
    'Fresh': {'min_days': 5, 'max_days': 7, 'storage_condition': 'Store in a cool, well-ventilated dry area (15-20°C) or refrigerate.', 'recommendation': 'Peak quality for fresh consumption.'},
    'Overripe': {'min_days': 1, 'max_days': 2, 'storage_condition': 'Refrigerate immediately at 4°C to prevent decay.', 'recommendation': 'Consume promptly; ideal for smoothies, baking, or purees.'},
    'Rotten': {'min_days': 0, 'max_days': 0, 'storage_condition': 'Isolate and do not store near fresh produce.', 'recommendation': 'Discard immediately or compost. Unsafe for human consumption.'}
}

def preprocess_numpy_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    img_array = np.expand_dims(frame_resized.astype(np.float32), axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def process_video(video_path, num_samples=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, None, None, None, None, None

    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames_collected = []
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frames_collected.append(frame)
    cap.release()

    if not frames_collected:
        return None, None, None, None, None, None

    keyframe_filename = f"keyframe_{os.path.basename(video_path)}.jpg"
    keyframe_path = os.path.join(app.config['UPLOAD_FOLDER'], keyframe_filename)
    cv2.imwrite(keyframe_path, frames_collected[len(frames_collected) // 2])

    fruit_votes = []
    quality_predictions = []

    for frame in frames_collected:
        processed = preprocess_numpy_frame(frame)
        
        f_preds = fruit_model.predict(processed, verbose=0)[0]
        f_idx = int(np.argmax(f_preds))
        fruit_votes.append((FRUIT_CLASSES[f_idx], f_preds[f_idx] * 100))
        
        q_preds = quality_model.predict(processed, verbose=0)[0]
        q_idx = int(np.argmax(q_preds))
        quality_predictions.append((QUALITY_CLASSES[q_idx], q_preds[q_idx] * 100, processed, q_idx))

    dominant_fruit = Counter([v[0] for v in fruit_votes]).most_common(1)[0][0]
    avg_fruit_conf = np.mean([v[1] for v in fruit_votes if v[0] == dominant_fruit])

    severity_order = {'Rotten': 3, 'Overripe': 2, 'Fresh': 1}
    worst_case = max(quality_predictions, key=lambda item: (severity_order[item[0]], item[1]))
    dominant_quality = worst_case[0]
    quality_conf = worst_case[1]
    worst_processed_tensor = worst_case[2]
    worst_q_idx = worst_case[3]

    # Generate Grad-CAM targeting the worst-case defect class
    gradcam_filename = f"cam_{keyframe_filename}"
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    generate_and_save_gradcam(keyframe_path, worst_processed_tensor, quality_model, gradcam_path, pred_index=worst_q_idx)

    return dominant_fruit, avg_fruit_conf, dominant_quality, quality_conf, keyframe_filename, gradcam_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        is_video = filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))

        if is_video:
            res = process_video(filepath)
            if res[0] is None:
                return "Error processing video file.", 400
            fruit, f_conf, quality, q_conf, display_img, cam_img = res
        else:
            frame = cv2.imread(filepath)
            processed = preprocess_numpy_frame(frame)
            display_img = filename

            f_preds = fruit_model.predict(processed, verbose=0)[0]
            f_idx = int(np.argmax(f_preds))
            f_conf = float(f_preds[f_idx]) * 100
            fruit = FRUIT_CLASSES[f_idx]

            q_preds = quality_model.predict(processed, verbose=0)[0]
            q_idx = int(np.argmax(q_preds))
            q_conf = float(q_preds[q_idx]) * 100
            quality = QUALITY_CLASSES[q_idx]

            # Fixed: Ensure distinct filename so original image is never overwritten
            cam_filename = f"cam_{filename}"
            cam_full_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_filename)
            generate_and_save_gradcam(filepath, processed, quality_model, cam_full_path, pred_index=q_idx)
            cam_img = cam_filename

        # 75% Confidence threshold gating (aligned with paper ICMSEE-254)
        if f_conf < 75.0:
            return render_template(
                'result.html',
                img_file=display_img,
                cam_file=None,
                fruit="Unknown Item",
                fruit_conf=f"{f_conf:.1f}%",
                quality="N/A",
                quality_conf="N/A",
                shelf_life={"min_days": "-", "max_days": "-", "storage_condition": "-", "recommendation": "Unrecognized object or low confidence (<75%). Please ensure clear lighting and centered fruit."},
                nutrition={}
            )

        return render_template(
            'result.html',
            img_file=display_img,
            cam_file=cam_img,
            fruit=fruit,
            fruit_conf=f"{f_conf:.1f}%",
            quality=quality,
            quality_conf=f"{q_conf:.1f}%",
            shelf_life=SHELF_LIFE_RULES.get(quality, {}),
            nutrition=NUTRITION_DATA.get(fruit, {})
        )

    return render_template('index.html')

ACTIVE_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-pro-latest"
]

# Feature 2: Interactive Fruit Chatbot Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').strip()
    if not user_msg:
        return jsonify({'reply': "Please ask a question about fruit quality, storage, or nutrition."})

    # 1. Primary AI Call with automatic failover across active models
    if client is not None:
        for model_name in ACTIVE_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=800
                    )
                )
                if response and response.text:
                    return jsonify({'reply': response.text.strip()})
            except Exception as e:
                # Log the temporary spike and silently move to the next model
                print(f"Model {model_name} unavailable ({e}). Trying fallback model...")
                continue

    # 2. Local Fallback (Only executes if all API models fail or network is down)
    msg_lower = user_msg.lower()
    for fruit, data in NUTRITION_DATA.items():
        if fruit.lower() in msg_lower:
            vitamins = ', '.join(data['vitamins']) if isinstance(data['vitamins'], list) else data['vitamins']
            return jsonify({
                'reply': f"{fruit}: {data['benefits']} It contains {data.get('calories', 50)} kcal and {vitamins}."
            })

    if any(k in msg_lower for k in ['store', 'storage', 'fridge', 'shelf']):
        return jsonify({
            'reply': "Fresh fruits last 5–7 days in a cool area. Overripe fruits should be refrigerated immediately (1–2 days). Rotten produce should be discarded."
        })

    return jsonify({
        'reply': "I am your Fruit AI Assistant! Ask me anything regarding fruit varieties, storage conditions, or nutritional facts."
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)