"""Flask Web Server for ChronoFrame Interactive UI."""
import os
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from models.pipeline import EnhancedTextToVideoPipeline
from utils.video_utils import frame_to_base64

app = Flask(__name__, static_folder="static")
CORS(app)

# Pipeline Singleton
pipeline = EnhancedTextToVideoPipeline()

# In-Memory Job Storage
jobs = {}


def run_pipeline_job(job_id: str, prompt: str, mode: str, num_keyframes: int, resolution: int, fps: int):
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

        # Encode keyframes to base64 for direct UI display
        kf_b64 = [frame_to_base64(kf) for kf in res["keyframes"]]

        jobs[job_id]["status"] = "COMPLETE"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = {
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
    except Exception as e:
        jobs[job_id]["status"] = "ERROR"
        jobs[job_id]["error"] = str(e)
        print(f"[Job Error] {job_id}: {e}")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    prompt = data.get("prompt", "A panda making latte art").strip()
    mode = data.get("mode", "proposed").lower()
    keyframes = int(data.get("keyframes", 4))
    resolution = int(data.get("resolution", 256))
    fps = int(data.get("fps", 8))

    job_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, prompt, mode, keyframes, resolution, fps),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "GENERATING KEYFRAMES", "progress": 10})


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "result": job["result"]
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json or {}
    prompt = data.get("prompt", "A panda making latte art").strip()
    keyframes = int(data.get("keyframes", 4))
    resolution = int(data.get("resolution", 256))
    fps = int(data.get("fps", 8))

    res_dir = os.path.abspath("outputs")
    os.makedirs(res_dir, exist_ok=True)
    
    out_b = os.path.join(res_dir, "compare_baseline.mp4")
    out_p = os.path.join(res_dir, "compare_proposed.mp4")

    res_baseline = pipeline.generate(prompt=prompt, mode="baseline", num_keyframes=keyframes, resolution=(resolution, resolution), fps=fps, output_path=out_b)
    res_proposed = pipeline.generate(prompt=prompt, mode="proposed", num_keyframes=keyframes, resolution=(resolution, resolution), fps=fps, output_path=out_p)

    return jsonify({
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
    })


@app.route("/outputs/<filename>", methods=["GET"])
def serve_output(filename):
    res_dir = os.path.abspath("outputs")
    file_path = os.path.join(res_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    download = request.args.get("download", "0") == "1"
    return send_file(file_path, as_attachment=download, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
