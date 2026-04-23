import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from nudenet import NudeDetector
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads")
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "model/best_model.h5")
IMG_SIZE = (300, 300)

NUDENET_LABEL_THRESHOLDS = {
    "FEMALE_GENITALIA_EXPOSED": 0.55,
    "MALE_GENITALIA_EXPOSED":   0.60,
    "FEMALE_BREAST_EXPOSED":    0.55,
    "BUTTOCKS_EXPOSED":         0.60,
    "ANUS_EXPOSED":             0.60,
    "FEMALE_GENITALIA_COVERED": 0.72,
    "ANUS_COVERED":             0.75,
    "BUTTOCKS_COVERED":         0.75,
    "FEMALE_BREAST_COVERED":    0.72,
}

NUDE_HARD_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
NUDE_SOFT_LABELS = {
    "FEMALE_GENITALIA_COVERED",
    "ANUS_COVERED",
    "BUTTOCKS_COVERED",
    "FEMALE_BREAST_COVERED",
}

# Model threshold: if nsfw_prob >= this, model alone can flag Unsafe (no NudeNet needed)
# Set high (0.9990) so only images the model is extremely sure about get flagged
IMAGE_MODEL_STRONG_THRESHOLD  = float(os.getenv("IMAGE_MODEL_STRONG_THRESHOLD",  "0.999"))
IMAGE_MODEL_ALONE_THRESHOLD   = float(os.getenv("IMAGE_MODEL_ALONE_THRESHOLD",   "0.9990"))
IMAGE_MODEL_NUDENET_SUPPORT   = float(os.getenv("IMAGE_MODEL_NUDENET_SUPPORT",   "0.50"))

VIDEO_UNSAFE_RATIO_THRESHOLD    = float(os.getenv("VIDEO_UNSAFE_RATIO_THRESHOLD",    "0.45"))
VIDEO_AVG_UNSAFE_CONF_THRESHOLD = float(os.getenv("VIDEO_AVG_UNSAFE_CONF_THRESHOLD", "60.0"))

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {"mp4", "mov", "avi", "mkv", "webm"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model loaded ✅")

print("Loading NudeNet...")
nude_detector = NudeDetector()
print("NudeNet loaded ✅")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_file(filename: str, exts: Set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in exts


def _build_upload_path(filename: str, exts: Set[str]) -> Path:
    safe = secure_filename(filename or "")
    if not safe:
        raise ValueError("Invalid filename")
    if not _allowed_file(safe, exts):
        raise ValueError("Unsupported file type")
    return UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{safe}"


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _to_array(img: Image.Image) -> np.ndarray:
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return np.asarray(img.resize(IMG_SIZE, resample), dtype=np.float32) / 255.0


def _center_crop(img: Image.Image, ratio: float = 0.92) -> Image.Image:
    w, h = img.size
    c = int(min(w, h) * ratio)
    if c <= 0:
        return img
    return img.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))


# ── Model (TTA) ───────────────────────────────────────────────────────────────

def predict_with_your_model(img_path: str) -> Dict:
    base   = _load_rgb(img_path)
    center = _center_crop(base)
    batch  = np.stack([_to_array(v) for v in
                       [base, ImageOps.mirror(base), center, ImageOps.mirror(center)]])
    preds        = model.predict(batch, verbose=0).reshape(-1)
    neutral_prob = float(np.clip(np.mean(preds), 0.0, 1.0))
    nsfw_prob    = 1.0 - neutral_prob
    tta_std      = float(np.std(preds))
    return {
        "neutral_prob": round(neutral_prob, 4),
        "nsfw_prob":    round(nsfw_prob, 4),
        "tta_std":      round(tta_std, 4),
    }


# ── NudeNet ───────────────────────────────────────────────────────────────────

def predict_with_nudenet(img_path: str) -> Dict:
    try:
        detections = nude_detector.detect(img_path) or []
    except Exception as e:
        print(f"NudeNet error: {e}")
        return {"hard_flag": False, "soft_flag": False, "top_score": 0.0, "top_label": ""}

    best_hard = {"score": 0.0, "label": ""}
    best_soft = {"score": 0.0, "label": ""}

    for item in detections:
        cls   = item.get("class", "")
        score = float(item.get("score", 0.0))
        req   = NUDENET_LABEL_THRESHOLDS.get(cls, 1.0)

        if cls in NUDE_HARD_LABELS and score >= req:
            if score > best_hard["score"]:
                best_hard = {"score": score, "label": cls}
        elif cls in NUDE_SOFT_LABELS and score >= req:
            if score > best_soft["score"]:
                best_soft = {"score": score, "label": cls}

    hard_flag = best_hard["score"] > 0.0
    soft_flag = best_soft["score"] > 0.0
    top = best_hard if best_hard["score"] >= best_soft["score"] else best_soft

    return {
        "hard_flag": hard_flag,
        "soft_flag": soft_flag,
        "top_score": round(top["score"], 4),
        "top_label": top["label"],
    }


# ── Ensemble ──────────────────────────────────────────────────────────────────
#
#  Rule 1 — NudeNet detects exposed body part             → UNSAFE
#  Rule 2 — NudeNet detects covered body part             → UNSAFE
#  Rule 3 — Model > 99.9% NSFW AND NudeNet sees ≥ 0.50   → UNSAFE
#  Rule 3.5— Model > 99.90% NSFW (model alone, no NudeNet needed) → UNSAFE
#            This catches gore, violence, graphic content that NudeNet misses
#  Rule 4 — Everything else                               → SAFE

def ensemble_predict(img_path: str) -> Tuple[str, str, float, float, str, Dict]:
    model_out = predict_with_your_model(img_path)
    nude_out  = predict_with_nudenet(img_path)

    neutral_prob   = float(model_out["neutral_prob"])
    nsfw_prob      = float(model_out["nsfw_prob"])
    nude_top_score = float(nude_out["top_score"])
    model_strong   = nsfw_prob >= IMAGE_MODEL_STRONG_THRESHOLD
    model_alone    = nsfw_prob >= IMAGE_MODEL_ALONE_THRESHOLD

    # Rule 1 — NudeNet hard flag (exposed body parts)
    if nude_out["hard_flag"]:
        label  = "Unsafe"
        reason = "Unsafe content detected"

    # Rule 2 — NudeNet soft flag (covered body parts)
    elif nude_out["soft_flag"]:
        label  = "Unsafe"
        reason = "Unsafe content detected"

    # Rule 3 — Model very confident + NudeNet partial support
    elif model_strong and nude_top_score >= IMAGE_MODEL_NUDENET_SUPPORT:
        label  = "Unsafe"
        reason = "Unsafe content detected"

    # Rule 3.5 — Model extremely confident ALONE (catches gore/violence NudeNet misses)
    elif model_alone:
        label  = "Unsafe"
        reason = "Unsafe content detected"

    # Rule 4 — Safe
    else:
        label  = "Safe"
        reason = "No explicit content detected"

    category = "NSFW" if label == "Unsafe" else "Neutral"

    confidence = round(nsfw_prob * 100, 2) if label == "Unsafe" else round(neutral_prob * 100, 2)

    scores = {
        "model_nsfw":     round(nsfw_prob, 4),
        "model_neutral":  round(neutral_prob, 4),
        "model_tta_std":  round(float(model_out["tta_std"]), 4),
        "nudenet_score":  round(nude_top_score, 4),
        "nudenet_label":  nude_out["top_label"],
        "nude_hard_flag": nude_out["hard_flag"],
        "nude_soft_flag": nude_out["soft_flag"],
        "model_strong":   model_strong,
        "model_alone":    model_alone,
    }

    return label, category, confidence, round(neutral_prob, 4), reason, scores


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _find_last_conv(m) -> Optional[tf.keras.layers.Layer]:
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            r = _find_last_conv(layer)
            if r:
                return r
    return None


def generate_gradcam(img_path: str, target_label: str = "Unsafe") -> Optional[str]:
    try:
        conv = _find_last_conv(model)
        if conv is None:
            return None

        gm  = tf.keras.Model(inputs=model.input, outputs=[conv.output, model.output])
        arr = np.expand_dims(_to_array(_load_rgb(img_path)), axis=0)

        with tf.GradientTape() as tape:
            co, preds   = gm(arr)
            neutral_out = preds[:, 0]
            loss        = (1.0 - neutral_out) if target_label == "Unsafe" else neutral_out

        grads = tape.gradient(loss, co)
        if grads is None:
            return None

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.maximum(tf.reduce_sum(co[0] * pooled, axis=-1), 0)
        mx = tf.reduce_max(heatmap)
        if mx > 0:
            heatmap = heatmap / mx
        heatmap = heatmap.numpy()

        base    = _load_rgb(img_path)
        bgr     = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
        h       = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))
        colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.65, colored, 0.35, 0)

        out = str(Path(img_path).with_name(f"{Path(img_path).stem}_gradcam.jpg"))
        cv2.imwrite(out, overlay)
        return out
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/image", methods=["POST"])
def predict_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    try:
        fp = _build_upload_path(file.filename, ALLOWED_IMAGE_EXTENSIONS)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    file.save(str(fp))
    label, category, confidence, raw_prob, reason, scores = ensemble_predict(str(fp))
    gc     = generate_gradcam(str(fp), target_label=label)
    gc_url = f"/uploads/{Path(gc).name}" if gc else None

    return jsonify({
        "label": label, "category": category, "confidence": confidence,
        "raw_prob": raw_prob, "reason": reason, "scores": scores, "gradcam_url": gc_url,
    })


@app.route("/predict/video", methods=["POST"])
def predict_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    try:
        fp = _build_upload_path(file.filename, ALLOWED_VIDEO_EXTENSIONS)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    file.save(str(fp))
    cap   = cv2.VideoCapture(str(fp))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    results, fc = [], 0
    every = max(1, int(fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fc % every == 0:
            ffp = UPLOAD_FOLDER / f"frame_{uuid.uuid4().hex}_{fc}.jpg"
            cv2.imwrite(str(ffp), frame)
            label, category, conf, _, reason, scores = ensemble_predict(str(ffp))
            results.append({
                "timestamp": round(fc / fps, 2), "label": label,
                "category": category, "confidence": conf,
                "reason": reason, "scores": scores,
            })
            if ffp.exists():
                ffp.unlink()
        fc += 1

    cap.release()
    if fp.exists():
        fp.unlink()

    if not results:
        return jsonify({"error": "No frames analyzed"}), 400

    unsafe  = sum(1 for r in results if r["label"] == "Unsafe")
    total   = len(results)
    ratio   = unsafe / total if total else 0.0
    confs   = [r["confidence"] for r in results if r["label"] == "Unsafe"]
    avg_c   = float(np.mean(confs)) if confs else 0.0
    verdict = (
        "Unsafe"
        if unsafe >= 2 and ratio >= VIDEO_UNSAFE_RATIO_THRESHOLD and avg_c >= VIDEO_AVG_UNSAFE_CONF_THRESHOLD
        else "Safe"
    )

    return jsonify({
        "total_frames_analyzed": total, "unsafe_frames": unsafe,
        "safe_frames": total - unsafe, "unsafe_ratio": round(ratio, 4),
        "avg_unsafe_confidence": round(avg_c, 2),
        "video_thresholds": {
            "unsafe_ratio": VIDEO_UNSAFE_RATIO_THRESHOLD,
            "avg_unsafe_confidence": VIDEO_AVG_UNSAFE_CONF_THRESHOLD,
        },
        "verdict": verdict, "timeline": results,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5000")))