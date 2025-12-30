import os
import json
import time
import traceback

from flask_sqlalchemy import SQLAlchemy
from models import db as models_db, User, Upload

from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session
from werkzeug.utils import secure_filename
from werkzeug.routing import BuildError
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

from predict import WasteClassifier
print("✅ Successfully imported WasteClassifier from predict.py")

# ======================================================
# CONFIG
# ======================================================

root_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, root_path=root_dir)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///ecoscan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "super_secret_dev_key_change_in_prod"

models_db.init_app(app)
migrate = Migrate(app, models_db)

UPLOAD_FOLDER = 'static/uploads/temp'
os.makedirs(os.path.join(root_dir, UPLOAD_FOLDER), exist_ok=True)
app.config['UPLOAD_FOLDER'] = os.path.join(root_dir, UPLOAD_FOLDER)
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

# ======================================================
# 🚨 AUTO DATABASE CREATION (Fix for Render)
# ======================================================

with app.app_context():
    try:
        print("🛠️ Checking database & tables...")
        models_db.create_all()
        print("📌 Database ready!")
    except Exception as e:
        print("❌ Error creating database:", e)

# ======================================================
# LOAD MODEL AT START
# ======================================================

print("\n🚀 Pre-loading ML Model... (please wait)")

try:
    CLASSIFIER = WasteClassifier()
    CLASSIFIER._ensure_model_loaded()
    print("✅ ML model loaded!\n")
except Exception as e:
    print("❌ Model load failed:", e)
    CLASSIFIER = None


# ======================================================
# HELPERS
# ======================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def get_recycling_suggestions(label):
    suggestions = {
        'plastic': ['Rinse and flatten bottle/container.', 'Check local collection rules.'],
        'glass': ['Rinse clean, remove lids.', 'Glass may require drop-off.'],
        'metal': ['Empty and rinse cans.', 'Crush if possible.'],
        'paper': ['Keep dry, remove plastic windows.'],
        'cardboard': ['Flatten boxes.', 'Remove tape.'],
        'trash': ['Dispose in standard waste bin.'],
    }
    return suggestions.get(label.lower(), ['No guidance available.'])

def classify_image(path):
    try:
        if CLASSIFIER:
            result, details = CLASSIFIER.predict(path, topk=1)
            if 'error' in result:
                raise Exception(result['error'])
            label = result['prediction']
            result['label'] = label
            result['is_recyclable'] = result.get('is_recyclable', False)
            result['suggestions'] = get_recycling_suggestions(label)
            result['classification_details'] = details
            return result
        return {"label":"trash","is_recyclable":False,"suggestions":[]}
    except Exception as e:
        return {"label":"Server Error","is_recyclable":False,"suggestions":[str(e)],"error":str(e)}


# ======================================================
# AUTH ROUTES
# ======================================================

@app.route('/register_user', methods=['GET','POST'])
def register_user():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        pwd = generate_password_hash(request.form['password'])
        if User.query.filter_by(email=email).first():
            flash("Email already exists","error")
            return redirect('/register_user')
        u = User(username=name,email=email,password=pwd,role="user")
        models_db.session.add(u)
        models_db.session.commit()
        flash("Account created! Login now.","success")
        return redirect('/login')
    return render_template("register_user.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['password']
        u = User.query.filter_by(email=email).first()
        if u and check_password_hash(u.password,pwd):
            session['email']=email
            session['user_id']=u.id
            session['username']=u.username
            session['user_role']=u.role
            return redirect('/')
        flash("Invalid login","error")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ======================================================
# MAIN ROUTES
# ======================================================

@app.route('/')
def index():
    if 'email' not in session: return redirect('/login')
    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    if 'email' not in session: return redirect('/login')
    user_id = session['user_id']
    uploads = Upload.query.filter_by(user_id=user_id).all()
    return render_template("dashboard.html", uploads=uploads)


# ======================================================
# IMAGE UPLOAD + CLASSIFY
# ======================================================

@app.route('/upload_file',methods=['GET','POST'])
def upload_file():
    if 'email' not in session: return jsonify({"error":"Login required"}),401
    if request.method=='GET': return render_template("upload.html")

    try:
        file = request.files['wasteImage']
        if not file or file.filename=="":
            return jsonify({"success":False,"error":"No file"}),400
        if not allowed_file(file.filename):
            return jsonify({"success":False,"error":"Invalid type"}),400

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        result = classify_image(save_path)
        user_id = session['user_id']

        rel_path = os.path.relpath(save_path,start=os.path.join(root_dir,'static')).replace("\\","/")

        upload = Upload(
            user_id=user_id,
            image_path=rel_path,
            label=result['label'],
            is_recyclable=result['is_recyclable'],
            suggestions=json.dumps(result['suggestions']),
            description=f"{result['label']} waste detected."
        )
        models_db.session.add(upload)
        models_db.session.commit()

        session['prediction_result'] = {
            "upload_id": upload.id,
            "label": upload.label,
            "image_url": url_for("static",filename=rel_path),
            "is_recyclable":upload.is_recyclable,
            "points_earned":100 if upload.is_recyclable else 0,
            "suggestions":result['suggestions'],
            "classification_details":result.get("classification_details",[])
        }

        return jsonify({"success":True})

    except Exception as e:
        print("UPLOAD ERROR:",e)
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route('/results')
def results():
    data = session.pop("prediction_result",None)
    if not data:
        flash("No result found. Upload again.","error")
        return redirect('/upload_file')
    return render_template("results.html", prediction_data_json=json.dumps(data))


# ======================================================
# RUN
# ======================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
