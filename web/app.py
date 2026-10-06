"""Flask Web Server for ChronoFrame Interactive UI."""
import os
import sys
import time
import uuid
import math
import threading
from werkzeug.utils import secure_filename

# Add parent directory to sys.path so models and utils can be imported cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from models.pipeline import EnhancedTextToVideoPipeline
from utils.video_utils import frame_to_base64

app = Flask(__name__, static_folder="static")
CORS(app)

# Pipeline Singleton
pipeline = EnhancedTextToVideoPipeline()

# In-Memory Job Storage with Thread Safety
jobs = {}
jobs_lock = threading.Lock()


def sanitize_json_obj(obj):
    """Recursively converts any float Inf / NaN values into None for standard RFC 8259 JSON compliance."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json_obj(v) for v in obj]
    return obj


def parse_int_param(val, default_val, min_val, max_val):
    """Safely parses an integer parameter from input json with min/max boundary clamping."""
    try:
        if val is None:
            return default_val
        val_int = int(val)
        return max(min_val, min(max_val, val_int))
    except (ValueError, TypeError):
        return default_val


def run_pipeline_job(job_id: str, prompt: str, mode: str, num_keyframes: int, resolution: int, fps: int):
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "GENERATING KEYFRAMES",
            "progress": 10,
            "prompt": prompt,
            "mode": mode,
            "result": None,
            "error": None,
            "timestamp": time.time()
        }

    def callback(stage: str, pct: int):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = stage
                jobs[job_id]["progress"] = pct

    try:
        res_dir = os.path.abspath("outputs")
        os.makedirs(res_dir, exist_ok=True)
        filename = f"{mode}_{job_id[:8]}.mp4"
        out_path = os.path.join(res_dir, filename)

        res = pipeline.generate(
            prompt=prompt,
            mode=mode,
            num_keyframes=num_keyframes,
            resolution=(resolution, resolution),
            fps=fps,
            output_path=out_path,
            callback=callback
        )

        kf_b64 = [frame_to_base64(kf) for kf in res["keyframes"]]

        result_obj = {
            "mode": res["mode"],
            "mode_label": res["mode_label"],
            "prompt": res["prompt"],
            "keyframes_b64": kf_b64,
            "video_url": f"/outputs/{filename}",
            "filename": filename,
            "frame_count": res["frame_count"],
            "fps": res["fps"],
            "duration": res["duration"],
            "generation_time": res["generation_time"],
            "interp_latency": res["interp_latency"],
            "metrics": res["metrics"]
        }

        with jobs_lock:
            jobs[job_id]["status"] = "COMPLETE"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = sanitize_json_obj(result_obj)

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "ERROR"
            jobs[job_id]["error"] = str(e)
        print(f"[Job Error] {job_id}: {e}")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    
    # Robust Input Validation & Parsing
    prompt = str(data.get("prompt") or "A panda making latte art").strip()
    if not prompt:
        prompt = "A panda making latte art"
        
    mode = str(data.get("mode") or "proposed").lower()
    if mode not in ["proposed", "chronoframe", "baseline"]:
        mode = "proposed"

    keyframes = parse_int_param(data.get("keyframes"), 4, 1, 8)
    resolution = parse_int_param(data.get("resolution"), 256, 64, 512)
    fps = parse_int_param(data.get("fps"), 8, 1, 60)

    job_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, prompt, mode, keyframes, resolution, fps),
        daemon=True
    )
    thread.start()

    return jsonify(sanitize_json_obj({"job_id": job_id, "status": "GENERATING KEYFRAMES", "progress": 10}))


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        response_data = {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error": job["error"],
            "result": job["result"]
        }
    return jsonify(sanitize_json_obj(response_data))


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json or {}
    prompt = str(data.get("prompt") or "A panda making latte art").strip()
    if not prompt:
        prompt = "A panda making latte art"

    keyframes = parse_int_param(data.get("keyframes"), 4, 1, 8)
    resolution = parse_int_param(data.get("resolution"), 256, 64, 512)
    fps = parse_int_param(data.get("fps"), 8, 1, 60)

    res_dir = os.path.abspath("outputs")
    os.makedirs(res_dir, exist_ok=True)
    
    out_b = os.path.join(res_dir, "compare_baseline.mp4")
    out_p = os.path.join(res_dir, "compare_proposed.mp4")

    res_baseline = pipeline.generate(prompt=prompt, mode="baseline", num_keyframes=keyframes, resolution=(resolution, resolution), fps=fps, output_path=out_b)
    res_proposed = pipeline.generate(prompt=prompt, mode="proposed", num_keyframes=keyframes, resolution=(resolution, resolution), fps=fps, output_path=out_p)

    compare_res = {
        "baseline": {
            "mode": "baseline",
            "video_url": "/outputs/compare_baseline.mp4",
            "metrics": res_baseline["metrics"],
            "generation_time": res_baseline["generation_time"]
        },
        "proposed": {
            "mode": "proposed",
            "video_url": "/outputs/compare_proposed.mp4",
            "metrics": res_proposed["metrics"],
            "generation_time": res_proposed["generation_time"]
        }
    }
    return jsonify(sanitize_json_obj(compare_res))


@app.route("/outputs/<filename>", methods=["GET"])
def serve_output(filename):
    res_dir = os.path.abspath("outputs")
    
    # Security: Prevent Directory Traversal Attacks
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return jsonify({"error": "Invalid filename"}), 400
        
    file_path = os.path.join(res_dir, safe_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    download = request.args.get("download", "0") == "1"
    return send_file(file_path, as_attachment=download, download_name=safe_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
