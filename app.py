import os
import secrets
import threading
from pathlib import Path
import tempfile
import uuid
import zipfile
from functools import wraps

from flask import (
    Flask,
    jsonify,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import yaml

from config import BASE_DIR, Config, ISSUE_TYPES, init_database, session_scope
from ai_service.model import train_waste_model, verify_waste
from services.complaint_service import (
    bootstrap_sample_data,
    create_complaint,
    list_complaints,
    update_complaint_status,
)
from services.hotspot_service import (
    build_dashboard_summary,
    build_heatmap,
    get_route_payload,
    list_hotspots,
)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

profiles_store = {}
training_state = {
    "status": "idle",
    "message": "No training run started yet.",
    "runName": None,
    "bestModelPath": None,
}
dataset_upload_dir = BASE_DIR / "uploads" / "datasets"
# Create dataset upload folder only locally
# Vercel filesystem is read-only
if os.environ.get("VERCEL") != "1":
    dataset_upload_dir.mkdir(parents=True, exist_ok=True)


def _is_authenticated():
    return bool(session.get("user"))


def _is_admin():
    return session.get("role") == "admin"


def require_auth(admin_only=False):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            is_api = request.path.startswith("/api/")
            if not _is_authenticated():
                if is_api:
                    return jsonify({"error": "Authentication required."}), 401
                return redirect(url_for("login"))
            if admin_only and not _is_admin():
                if is_api:
                    return jsonify({"error": "Admin access required."}), 403
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


@app.context_processor
def inject_session_context():
    return {
        "current_user": session.get("user", "Guest"),
        "current_role": session.get("role", "guest"),
        "is_admin": _is_admin(),
        "is_authenticated": _is_authenticated(),
    }


def _start_training_async(data_path, epochs, imgsz):
    training_state["status"] = "running"
    training_state["message"] = "Training in progress..."
    training_state["runName"] = None
    training_state["bestModelPath"] = None

    def runner():
        try:
            result = train_waste_model(data_path=data_path, epochs=epochs, imgsz=imgsz)
            training_state["status"] = "completed"
            training_state["message"] = "Training completed successfully."
            training_state["runName"] = result.get("run_name")
            training_state["bestModelPath"] = result.get("best_model_path")
        except Exception as exc:
            training_state["status"] = "failed"
            training_state["message"] = str(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


def _parse_yaml_file(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError("Dataset YAML is invalid. It must be a mapping/object.")
    return payload


def _resolve_data_path(dataset_path):
    raw = Path(dataset_path)
    candidate = raw if raw.is_absolute() else (BASE_DIR / raw)
    return candidate.resolve()


def _validate_dataset_path(dataset_path):
    resolved = _resolve_data_path(dataset_path)
    if not resolved.exists():
        raise RuntimeError(f"Dataset path not found: {resolved}")

    if resolved.is_file() and resolved.suffix.lower() in {".yaml", ".yml"}:
        payload = _parse_yaml_file(resolved)
        required_keys = ["train", "val", "names"]
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise RuntimeError("Dataset YAML missing required keys: " + ", ".join(missing))

        names = payload.get("names")
        class_count = len(names) if isinstance(names, list) else len(names.keys()) if isinstance(names, dict) else 0
        return {
            "resolvedPath": str(resolved),
            "classCount": class_count,
            "classes": names,
            "train": payload.get("train"),
            "val": payload.get("val"),
        }

    if not resolved.is_dir():
        raise RuntimeError("Dataset path must be a directory or YAML file.")

    train_dir = resolved / "train"
    val_dir = resolved / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError("Classification directory must contain train/ and val/ folders.")

    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError("No class folders found under train/.")

    return {
        "resolvedPath": str(resolved),
        "classCount": len(class_names),
        "classes": class_names,
        "train": "train",
        "val": "val",
    }


def _safe_extract_zip(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        for member in members:
            member_path = target_dir / member.filename
            if not str(member_path.resolve()).startswith(str(target_dir.resolve())):
                raise RuntimeError("Unsafe zip content detected.")
        archive.extractall(target_dir)


def _build_cnn_item_insight(predicted_type):
    label = (predicted_type or "Unknown item").strip()
    normalized = label.lower()

    catalog = [
        {
            "keywords": ["plastic"],
            "itemInfo": "Plastic item detected. Clean and dry plastics are commonly recyclable.",
            "recyclable": True,
            "recycleLabel": "Useful for recycling",
            "guidance": "Rinse, dry, and send to dry-waste collection. Avoid mixing with food waste.",
        },
        {
            "keywords": ["glass"],
            "itemInfo": "Glass item detected. Most glass containers are recyclable.",
            "recyclable": True,
            "recycleLabel": "Useful for recycling",
            "guidance": "Keep separate from mixed waste and avoid breakage during collection.",
        },
        {
            "keywords": ["metal", "aluminium", "aluminum", "can"],
            "itemInfo": "Metal item detected. Metals generally have high recovery value.",
            "recyclable": True,
            "recycleLabel": "Useful for recycling",
            "guidance": "Send to dry-waste or scrap stream after removing food residue.",
        },
        {
            "keywords": ["paper", "cardboard", "carton"],
            "itemInfo": "Paper/cardboard detected. Usually recyclable when dry and clean.",
            "recyclable": True,
            "recycleLabel": "Useful for recycling",
            "guidance": "Keep dry. Oily or wet paper should go to organic/residual handling.",
        },
        {
            "keywords": ["organic", "food", "biodegradable"],
            "itemInfo": "Organic waste detected. Better for composting than recycling.",
            "recyclable": False,
            "recycleLabel": "Not for dry recycling",
            "guidance": "Send to wet-waste stream for composting or bio-methanation.",
        },
        {
            "keywords": ["trash", "mixed"],
            "itemInfo": "Mixed/residual waste detected. Usually low recycling value.",
            "recyclable": False,
            "recycleLabel": "Not suitable for direct recycling",
            "guidance": "Try source segregation first. Only clean recoverables should go to recycling.",
        },
        {
            "keywords": ["hazard", "biomedical", "battery", "e-waste", "ewaste"],
            "itemInfo": "Hazardous stream detected. Requires specialized handling.",
            "recyclable": False,
            "recycleLabel": "Special handling required",
            "guidance": "Do not mix with household dry waste. Use authorized hazardous/e-waste collection.",
        },
    ]

    for entry in catalog:
        if any(keyword in normalized for keyword in entry["keywords"]):
            return {
                "itemName": label,
                "itemInfo": entry["itemInfo"],
                "recyclable": entry["recyclable"],
                "recycleLabel": entry["recycleLabel"],
                "guidance": entry["guidance"],
            }

    return {
        "itemName": label,
        "itemInfo": "Item detected, but recycling category is uncertain.",
        "recyclable": False,
        "recycleLabel": "Needs manual check",
        "guidance": "Please verify item type manually before sending it to recycling.",
    }


def _load_dashboard_carousel_images(limit=12):
    uploads_dir = BASE_DIR / "static" / "uploads"
    if not uploads_dir.exists():
        return []

    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files = [p for p in uploads_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    selected = files[:limit]

    return [url_for("static", filename=f"uploads/{p.name}") for p in selected]


init_database()
with session_scope() as db_session:
    bootstrap_sample_data(db_session)


@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/index")
def index_page():
    return render_template("index_page.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html", login_mode="user")

        session["user"] = email
        session["role"] = "user"
        return redirect(url_for("dashboard"))
    return render_template("login.html", login_mode="user")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if email == Config.ADMIN_EMAIL and password == Config.ADMIN_PASSWORD:
            session["user"] = email
            session["role"] = "admin"
            return redirect(url_for("dashboard"))

        flash("Invalid admin credentials.", "error")
    return render_template("login.html", login_mode="admin")


@app.route("/demo-login/google")
def demo_google_login():
    session["user"] = "google.user@geosmart.local"
    session["role"] = "user"
    return redirect(url_for("dashboard"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        session["user"] = name or email or "User"
        session["role"] = "user"
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/dashboard")
@require_auth()
def dashboard():
    user = session.get("user", "User")
    if _is_admin():
        return render_template("admin_dashboard.html", user=user, role="admin")
    carousel_images = _load_dashboard_carousel_images()
    return render_template("dashboard.html", user=user, role="user", carousel_images=carousel_images)


@app.route("/admin")
@require_auth(admin_only=True)
def admin_home():
    return redirect(url_for("dashboard"))


@app.route("/analyze")
@require_auth()
def analyze():
    return render_template("analyze.html")


@app.route("/complaint")
@require_auth()
def complaint():
    return render_template(
        "complaint.html",
        complaints=[],
        issue_types=ISSUE_TYPES,
        can_update_status=_is_admin(),
    )


@app.route("/municipal")
@require_auth()
def municipal():
    return render_template("municipal.html")


@app.route("/profile")
@require_auth()
def profile():
    return render_template("profile.html")


@app.route("/api/complaints", methods=["GET"])
@require_auth()
def api_get_complaints():
    current_user = session.get("user", "")
    with session_scope() as db_session:
        if _is_admin():
            return jsonify(list_complaints(db_session))
        return jsonify(list_complaints(db_session, contact=current_user))


@app.route("/api/complaints", methods=["POST"])
@require_auth()
def api_create_complaint():
    if request.files:
        payload = request.form.to_dict(flat=True)
        image_file = request.files.get("image")
    else:
        payload = request.get_json(silent=True) or {}
        image_file = None

    current_user = session.get("user", "")
    # Always map citizen-submitted complaints to the logged-in user so
    # the user dashboard can reliably show "my complaints" history.
    if current_user and not _is_admin():
        payload["contact"] = current_user
    elif current_user and not payload.get("contact"):
        payload["contact"] = current_user

    with session_scope() as db_session:
        record = create_complaint(db_session, payload, image_file=image_file)
        return jsonify(record), 201


@app.route("/api/complaints/<complaint_id>/status", methods=["PATCH"])
@require_auth(admin_only=True)
def api_update_complaint_status(complaint_id):
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    if new_status not in ("Pending", "Assigned", "In Progress", "Resolved"):
        return jsonify({"error": "Invalid status"}), 400

    with session_scope() as db_session:
        complaint = update_complaint_status(db_session, complaint_id, new_status)
        if not complaint:
            return jsonify({"error": "Not found"}), 404
        return jsonify(complaint)


@app.route("/api/hotspots", methods=["GET"])
@require_auth()
def api_hotspots():
    with session_scope() as db_session:
        hotspots = list_hotspots(db_session, refresh=True)
        heatmap = build_heatmap(db_session)
        summary = build_dashboard_summary(db_session)
        return jsonify({
            "generatedAt": summary["generatedAt"],
            "hotspots": hotspots,
            "heatmap": heatmap,
        })


@app.route("/api/routes/optimize", methods=["GET"])
@require_auth()
def api_optimize_route():
    hotspot_id = request.args.get("hotspotId")
    with session_scope() as db_session:
        route_payload = get_route_payload(db_session, hotspot_id)
        if not route_payload:
            return jsonify({"error": "No hotspot available"}), 404
        return jsonify(route_payload)


@app.route("/api/dashboard-summary", methods=["GET"])
@require_auth()
def api_dashboard_summary():
    with session_scope() as db_session:
        return jsonify(build_dashboard_summary(db_session))


@app.route("/api/profile", methods=["GET"])
@require_auth()
def api_get_profile():
    profile_data = profiles_store.get(
        "default",
        {
            "name": "Ananya Rao",
            "email": "ananya@geosmart.org",
            "location": "Ward 14, Koramangala",
            "role": "Community Sustainability Officer",
            "organisation": "GEOSMART Foundation",
        },
    )
    return jsonify(profile_data)


@app.route("/api/profile", methods=["PUT"])
@require_auth()
def api_update_profile():
    data = request.get_json(silent=True) or {}
    profiles_store["default"] = {
        "name": data.get("name", ""),
        "email": data.get("email", ""),
        "location": data.get("location", ""),
        "role": data.get("role", "Community Sustainability Officer"),
        "organisation": data.get("organisation", ""),
    }
    return jsonify(profiles_store["default"])


@app.route("/api/model/train", methods=["POST"])
@require_auth(admin_only=True)
def api_train_model():
    if training_state["status"] == "running":
        return jsonify({"error": "Training is already running."}), 409

    payload = request.get_json(silent=True) or {}
    data_path = payload.get("dataPath", "").strip()
    epochs = int(payload.get("epochs", 25))
    imgsz = int(payload.get("imgsz", 224))

    if not data_path:
        return jsonify({"error": "dataPath is required."}), 400

    _start_training_async(data_path=data_path, epochs=epochs, imgsz=imgsz)
    return jsonify({"status": "started", "message": "Training started."}), 202


@app.route("/api/model/validate-dataset", methods=["POST"])
@require_auth(admin_only=True)
def api_validate_dataset():
    payload = request.get_json(silent=True) or {}
    data_path = payload.get("dataPath", "").strip()
    if not data_path:
        return jsonify({"error": "dataPath is required."}), 400

    try:
        summary = _validate_dataset_path(data_path)
        return jsonify({"status": "valid", "summary": summary})
    except Exception as exc:
        return jsonify({"status": "invalid", "error": str(exc)}), 400


@app.route("/api/model/upload-dataset-zip", methods=["POST"])
@require_auth(admin_only=True)
def api_upload_dataset_zip():
    if training_state["status"] == "running":
        return jsonify({"error": "Cannot upload a dataset while training is running."}), 409

    zip_file = request.files.get("datasetZip")
    if not zip_file or not zip_file.filename:
        return jsonify({"error": "datasetZip file is required."}), 400
    if not zip_file.filename.lower().endswith(".zip"):
        return jsonify({"error": "Please upload a .zip dataset archive."}), 400

    run_id = uuid.uuid4().hex[:10]
    target_dir = dataset_upload_dir / f"dataset-{run_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "dataset.zip"
    zip_file.save(zip_path)

    try:
        _safe_extract_zip(zip_path, target_dir)
        discovered = list(target_dir.rglob("data.yaml"))
        if not discovered:
            raise RuntimeError("No data.yaml found inside uploaded zip.")
        data_yaml = discovered[0]
        summary = _validate_dataset_path(str(data_yaml))
        return jsonify(
            {
                "status": "uploaded",
                "dataYamlPath": summary["resolvedPath"],
                "summary": summary,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/model/train-status", methods=["GET"])
@require_auth(admin_only=True)
def api_train_status():
    return jsonify(training_state)


@app.route("/api/analyze/cnn", methods=["POST"])
@require_auth()
def api_analyze_cnn():
    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"error": "image file is required."}), 400

    suffix = Path(image_file.filename).suffix or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp.name)
            temp_path = tmp.name

        prediction = verify_waste(temp_path)
        status = "Verified" if prediction["verified"] else "Needs review"
        confidence = float(prediction["verification_score"])
        model_name = prediction.get("model", "GEOSMART CNN")
        if "incompatible labels" in model_name.lower():
            confidence_note = "Model labels are not waste-specific for this image. Please use a waste-trained model."
        elif prediction["verified"]:
            confidence_note = "High confidence classification."
        else:
            confidence_note = "Low confidence result. Try a clearer, well-lit photo for better accuracy."
        insight = _build_cnn_item_insight(prediction["predicted_type"])
        return jsonify(
            {
                "predictedType": prediction["predicted_type"],
                "confidence": round(confidence, 4),
                "status": status,
                "model": model_name,
                "confidenceNote": confidence_note,
                "itemInfo": insight["itemInfo"],
                "recyclable": insight["recyclable"],
                "recycleLabel": insight["recycleLabel"],
                "guidance": insight["guidance"],
            }
        )
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("splash"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)