"""
app.py
Flask web application for converting study protocols to study reports.
"""

import os
import json
import uuid
import shutil
import threading
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, send_file,
)
from werkzeug.utils import secure_filename
from document_processor import ProtocolProcessor

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
JOBS_FILE = os.path.join(os.path.dirname(__file__), "jobs.json")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def _load_jobs():
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_jobs(jobs):
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f, indent=2)
    except IOError:
        pass


conversion_jobs = _load_jobs()

for _jid, _job in conversion_jobs.items():
    if _job.get("status") == "converting":
        _job["status"] = "error"
        _job["error"] = "Server restarted during conversion. Please try again."
_save_jobs(conversion_jobs)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "docx"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .docx files are accepted"}), 400

    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    base, ext = os.path.splitext(filename)
    saved_name = f"{base}_{job_id}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(filepath)

    try:
        processor = ProtocolProcessor(filepath)
        structure = processor.get_structure()
        para_count = processor.get_paragraph_count()
    except Exception as e:
        return jsonify({"error": f"Failed to read document: {str(e)}"}), 400

    conversion_jobs[job_id] = {
        "filepath": filepath,
        "original_filename": filename,
        "status": "uploaded",
        "progress": 0,
        "total": para_count,
        "changes_made": 0,
        "title_replacements": 0,
        "error": None,
        "conversion_log": [],
    }
    _save_jobs(conversion_jobs)

    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "structure": structure,
        "paragraph_count": para_count,
    })


@app.route("/convert/<job_id>", methods=["POST"])
def start_conversion(job_id):
    if job_id not in conversion_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = conversion_jobs[job_id]
    if job["status"] == "converting":
        return jsonify({"error": "Conversion already in progress"}), 400

    job["status"] = "converting"
    job["progress"] = 0
    job["changes_made"] = 0
    job["error"] = None
    _save_jobs(conversion_jobs)

    def run_conversion():
        try:
            processor = ProtocolProcessor(job["filepath"])

            # Step 1: Replace "Protocol" with "Technical Report"
            title_count = processor.replace_protocol_with_report()
            job["title_replacements"] = title_count

            # Step 2: Convert tense
            def progress_cb(current, total, changes):
                job["progress"] = current
                job["total"] = total
                job["changes_made"] = changes

            result = processor.convert_to_past_tense(progress_callback=progress_cb)
            processor.save_report(job["filepath"])

            job["status"] = "converted"
            job["changes_made"] = result["changes_made"]
            job["conversion_log"] = result["log"]
            _save_jobs(conversion_jobs)

        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            _save_jobs(conversion_jobs)
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_conversion, daemon=True)
    thread.start()

    return jsonify({"message": "Conversion started"})


@app.route("/progress/<job_id>")
def get_progress(job_id):
    if job_id not in conversion_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = conversion_jobs[job_id]
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "changes_made": job["changes_made"],
        "title_replacements": job.get("title_replacements", 0),
        "error": job.get("error"),
    })


@app.route("/add-sections/<job_id>", methods=["POST"])
def add_sections(job_id):
    """
    Accept a JSON structure describing sections with subsections and items.
    Images are uploaded as separate files referenced by key.
    """
    if job_id not in conversion_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = conversion_jobs[job_id]

    # Parse sections JSON from form data
    try:
        sections_json = request.form.get("sections", "[]")
        sections_data = json.loads(sections_json)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid sections data: {str(e)}"}), 400

    heading_level = int(request.form.get("heading_level", 1))

    # Save uploaded images and map them by their form field key
    image_map = {}
    for key in request.files:
        if key.startswith("image_"):
            img_file = request.files[key]
            if img_file.filename:
                img_name = secure_filename(img_file.filename)
                img_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{job_id}_{key}_{img_name}"
                )
                img_file.save(img_path)
                image_map[key] = img_path

    # Resolve image paths in sections_data
    for section in sections_data:
        for sub in section.get("subsections", []):
            for item in sub.get("items", []):
                img_key = item.get("image_key")
                if img_key and img_key in image_map:
                    item["image_path"] = image_map[img_key]

    try:
        processor = ProtocolProcessor(job["filepath"])
        processor.add_report_sections(sections_data, heading_level)
        processor.save_report(job["filepath"])
        job["status"] = "sections_added"
        _save_jobs(conversion_jobs)

        return jsonify({"message": "Sections added successfully"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to add sections: {str(e)}"}), 500


@app.route("/export/<job_id>")
def export_report(job_id):
    if job_id not in conversion_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = conversion_jobs[job_id]
    original = job["original_filename"]
    base, ext = os.path.splitext(original)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_name = f"{base}_Report_{timestamp}{ext}"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)

    shutil.copy2(job["filepath"], output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name=output_name,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.route("/conversion-log/<job_id>")
def conversion_log(job_id):
    if job_id not in conversion_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = conversion_jobs[job_id]
    return jsonify({
        "changes_made": job.get("changes_made", 0),
        "log": job.get("conversion_log", []),
    })


@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    if job_id in conversion_jobs:
        job = conversion_jobs[job_id]
        try:
            if os.path.exists(job["filepath"]):
                os.remove(job["filepath"])
        except OSError:
            pass
        del conversion_jobs[job_id]
        _save_jobs(conversion_jobs)
    return jsonify({"message": "Cleaned up"})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Protocol -> Report Converter")
    print("  Open http://localhost:5000 in your browser")
    print("  Tense conversion runs locally (no API costs)")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
