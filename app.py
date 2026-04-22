import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from nudenet import NudeDetector

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/nsfw_model.h5"
THRESHOLD = 0.35
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading your trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Your model loaded ✅")

print("Loading NudeNet...")
nude_detector = NudeDetector()
print("NudeNet loaded ✅")

NUDE_UNSAFE_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED", "ANUS_COVERED"
]

def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_your_model(img_path):
    arr = preprocess_image(img_path)
    prob = float(model.predict(arr, verbose=0)[0][0])
    # NSFW=0, Neutral=1
    if prob >= THRESHOLD:
        return "Safe", "Neutral", round(prob * 100, 2), round(prob, 4)
    else:
        return "Unsafe", "NSFW", round((1 - prob) * 100, 2), round(prob, 4)

def predict_with_nudenet(img_path):
    try:
        results = nude_detector.detect(img_path)
        for item in results:
            if item["class"] in NUDE_UNSAFE_LABELS and item["score"] > 0.5:
                return "Unsafe", round(item["score"] * 100, 2)
        return "Safe", 100.0
    except:
        return "Safe", 100.0

def ensemble_predict(img_path):
    your_label, your_category, your_conf, raw_prob = predict_with_your_model(img_path)
    nude_label, nude_conf = predict_with_nudenet(img_path)

    # Ensemble logic
    if your_label == "Unsafe" and nude_label == "Unsafe":
        return "Unsafe", "NSFW", max(your_conf, nude_conf), raw_prob, "Both models flagged"
    elif your_label == "Safe" and nude_label == "Safe":
        return "Safe", "Neutral", your_conf, raw_prob, "Both models approved"
    elif your_label == "Unsafe" and nude_label == "Safe":
        return "Safe", "Neutral", 60.0, raw_prob, "NudeNet overrided — Safe"
    else:
        return "Unsafe", "NSFW", nude_conf, raw_prob, "NudeNet flagged"

def generate_gradcam(img_path):
    try:
        efficientnet = model.layers[1]
        last_conv = None
        for layer in efficientnet.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=efficientnet.inputs,
            outputs=[efficientnet.get_layer(last_conv).output, efficientnet.output]
        )
        img_array = preprocess_image(img_path)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        orig = cv2.imread(img_path)
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)
        gradcam_path = img_path.rsplit(".", 1)[0] + "_gradcam.jpg"
        cv2.imwrite(gradcam_path, overlay)
        return gradcam_path
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/image", methods=["POST"])
def predict_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    label, category, confidence, raw_prob, reason = ensemble_predict(filepath)
    gradcam_path = generate_gradcam(filepath)
    gradcam_url = "/" + gradcam_path.replace("\\", "/") if gradcam_path else None

    return jsonify({
        "label": label,
        "category": category,
        "confidence": confidence,
        "raw_prob": raw_prob,
        "reason": reason,
        "gradcam_url": gradcam_url
    })

@app.route("/predict/video", methods=["POST"])
def predict_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    results = []
    frame_count = 0
    sample_every = max(1, int(fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            label, category, confidence, _, reason = ensemble_predict(frame_path)
            results.append({
                "timestamp": round(frame_count / fps, 2),
                "label": label,
                "category": category,
                "confidence": confidence,
                "reason": reason
            })
            os.remove(frame_path)
        frame_count += 1

    cap.release()
    unsafe_count = sum(1 for r in results if r["label"] == "Unsafe")

    return jsonify({
        "total_frames_analyzed": len(results),
        "unsafe_frames": unsafe_count,
        "safe_frames": len(results) - unsafe_count,
        "verdict": "Unsafe" if unsafe_count > len(results) - unsafe_count else "Safe",
        "timeline": results
    })

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)