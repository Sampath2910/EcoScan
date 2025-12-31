import os
import json
import time 
import traceback 

# --- New imports for database ---
from flask_sqlalchemy import SQLAlchemy
from models import db as models_db, User, Upload

from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, session 
from werkzeug.utils import secure_filename
from werkzeug.routing import BuildError
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

# Import the custom classifier class
# --- Import WasteClassifier directly ---
from predict import WasteClassifier
print("✅ Successfully imported WasteClassifier from predict.py")

# =================================================================
# --- CONFIGURATION & INITIALIZATION ---
# =================================================================

# 1. Calculate the path to the project root
root_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Initialize Flask 
app = Flask(__name__, root_path=root_dir)
# --- SQLAlchemy Database Setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///ecoscan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the DB
models_db.init_app(app)

migrate = Migrate(app, models_db)
# 🚨 AUTO-CREATE TABLES ON FIRST RUN (For deployment / Render)
with app.app_context():
    try:
        models_db.create_all()
        print("📌 Database tables verified/created successfully.")
    except Exception as db_err:
        print("❌ Failed to create database tables:", db_err)

# Set configuration variables
app.secret_key = 'super_secret_dev_key_change_in_prod'
UPLOAD_FOLDER = 'static/uploads/temp'
os.makedirs(os.path.join(root_dir, UPLOAD_FOLDER), exist_ok=True) 
app.config['UPLOAD_FOLDER'] = os.path.join(root_dir, UPLOAD_FOLDER) 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

print("\n🚀 Pre-loading ML Model... (please wait)")

try:
    CLASSIFIER = WasteClassifier()
    CLASSIFIER._ensure_model_loaded()      # <<< IMPORTANT: force model load at startup
    print("✅ ML model fully ready and loaded!\n")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize WasteClassifier: {e}")
    CLASSIFIER = None


# --- MOCK DATA STRUCTURES (UPDATED for detailed user/reclaimer fields) ---
"""USERS = {
    'user@peerconnect.com': {
        'username': 'StandardUser', 'password': 'userpass', 'role': 'user',
        'phone': '1234567890', 'address': '123 Main St', 'village': 'Greenville', 'pincode': '10001'
    },
    'reclaimer@citydump.com': {
        'username': 'CityReclaimer', 'password': 'reclaimpass', 'role': 'reclaimer',
        'company_name': 'City Dump Inc.', 'address': '456 Industrial Rd'
    }
}"""
MOCK_RECYCLERS = [
    {"id": 1, "name": "Green City Recycling", "type": "Plastic", "city": "Mumbai", "lat": 19.0760, "lng": 72.8777},
    {"id": 2, "name": "Metal Scraps Inc.", "type": "Metal", "city": "Delhi", "lat": 28.7041, "lng": 77.1025},
    {"id": 3, "name": "Paper Pulp Co.", "type": "Paper", "city": "Bangalore", "lat": 12.9716, "lng": 77.5946},
    {"id": 4, "name": "Glass Rebirth", "type": "Glass", "city": "Kolkata", "lat": 22.5726, "lng": 88.3639},
    {"id": 5, "name": "Trash Disposal Ltd.", "type": "Trash", "city": "Chennai", "lat": 13.0827, "lng": 80.2707},
]
# Base structure for dashboard uploads (uses relative paths instead of url_for)
MOCK_UPLOADS_BASE = [
    {'image_path': 'uploads/sample_plastic.jpg', 'label': 'Plastic', 'is_recyclable': True, 'suggestions': ['Rinse container before disposal.', 'Find drop-off point.']},
    {'image_path': 'uploads/sample_trash.jpg', 'label': 'Trash', 'is_recyclable': False, 'suggestions': ['Dispose in standard waste bin.', 'Consider upcycling or reuse.']},
    {'image_path': 'uploads/sample_glass.jpg', 'label': 'Glass', 'is_recyclable': True, 'suggestions': ['Rinse clean, remove lids.', 'Glass may require drop-off.']},
]
USER_UPLOAD_HISTORY = []
MOCK_UPLOADS_TO_PROCESS = [
    {'id': 'UP-001', 'material': 'Paper', 'city': 'Mumbai', 'date': '2025-10-10', 'user': 'UserX'},
    {'id': 'UP-002', 'material': 'Metal', 'city': 'Delhi', 'date': '2025-10-10', 'user': 'UserY'},
]


# =================================================================
# --- UTILITY FUNCTIONS ---
# =================================================================

def allowed_file(filename):
    """Checks if a file extension is allowed."""
    return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_recycling_suggestions(label):
    """Provides specific recycling guidance based on the material label."""
    label_lower = label.lower()
    
    # Define suggestions based on material, matching the logic in predict.py for consistency
    RECYCLABLE_MAP = {
        'plastic': {'suggestions': ['Rinse and flatten bottle/container.', 'Check local collection rules.']},
        'glass': {'suggestions': ['Rinse clean, remove lids.', 'Glass may require drop-off.']},
        'metal': {'suggestions': ['Empty and rinse cans.', 'Crush if possible to save space.']},
        'paper': {'suggestions': ['Keep dry, remove plastic windows.', 'Shredding is not usually necessary.']},
        'cardboard': {'suggestions': ['Flatten boxes completely.', 'Remove all packing tape/labels.']},
        'trash': {'suggestions': ['Dispose in standard waste bin.', 'No recycling options available.']},
    }
    
    return RECYCLABLE_MAP.get(label_lower, {}).get('suggestions', ['No specific guidance available.'])

def classify_image(filepath):
    """
    Runs the AI classification or returns a mock result if the model is unavailable.
    Returns a dictionary with 'label', 'is_recyclable', and 'suggestions'.
    """
    try:
        if CLASSIFIER:
            # Real classification using the initialized model
            result, details = CLASSIFIER.predict(filepath, topk=1)
            
            # If the model prediction succeeded (no error returned from predict.py)
            if 'error' not in result:
                label = result.get('prediction', 'Unknown')
                
                # Augment the result with detailed suggestions
                result['suggestions'] = get_recycling_suggestions(label)
                result['label'] = label # Use 'label' key for consistency with results page
                result['is_recyclable'] = result.get('is_recyclable', False)
                result['classification_details'] = details # Add model details
                return result
            else:
                # Handle case where predict.py returned an explicit error object
                raise Exception(f"CLASSIFIER.predict returned an error: {result['error']}")
        else:
            # Fallback Mock Response for testing
            mock_pred = 'plastic' if 'plastic' in filepath.lower() or 'glass' in filepath.lower() else 'trash'
            is_rec = mock_pred != 'trash'
            return {
                'label': mock_pred.capitalize(), 
                'is_recyclable': is_rec, 
                'suggestions': get_recycling_suggestions(mock_pred),
                'classification_details': {
                    "model_version": "v1.0 (Mock)",
                    "confidence_score": "85% (Mock)",
                },
                'error': 'ML Model is not running on the server. Using Mock Result.'
            }

    except Exception as e:
        # CRITICAL: Catch any exception during the model run
        error_message = f"Critical ML Classification Failure: {e}"
        print(f"!!! {error_message} !!!")
        traceback.print_exc()
        
        # Return a structured error response that can still be rendered
        return {
            'label': 'Server Error', 
            'is_recyclable': False, 
            'suggestions': [
                "A severe error occurred during image processing (ML model crash).", 
                f"Technical details: {str(e)}", 
                "Please check the server console for the traceback to debug predict.py."
            ],
            'classification_details': {},
            'error': error_message
        }


def get_dashboard_data():
    """Builds the final dashboard data in the structure expected by dashboard.html."""
    final_uploads = []
    for upload in MOCK_UPLOADS_BASE:
        try:
            image_url = url_for('static', filename=upload['image_path'])
        except BuildError:
            image_url = '/' + upload['image_path']

        # Align with dashboard.html variable names
        new_upload = {
            'time': datetime.now(),  # mock timestamp for demo purposes
            'image_url': image_url,
            'material_label': upload.get('label', 'Unknown'),
            'is_recyclable': upload.get('is_recyclable', False),
            'suggestions': upload.get('suggestions', [])
        }

        final_uploads.append(new_upload)

    return {
        'metrics': {
            'total_uploads': 42,
            'recyclable_count': 35,
            'trash_count': 7,
            'rewards_earned': 1500
        },
            'uploads': uploads_data  # fetched directly from Upload table


    }



# =================================================================
# --- CONTEXT PROCESSOR ---
# =================================================================

@app.context_processor
def inject_global_context():
    """Injects common variables into all templates."""
    # Assuming Canvas environment variables are passed (or they will default to empty strings)
    app_id = os.environ.get('__app_id', 'default-app-id')
    firebase_config = os.environ.get('__firebase_config', '{}')
    initial_auth_token = os.environ.get('__initial_auth_token', '')

    return {
        'current_year': datetime.now().year,
        'cache_buster': int(time.time()),
        'user_is_authenticated': 'email' in session,
        'user_role': session.get('user_role', 'guest'),
        'username': session.get('username', 'Guest'),
        
        # Inject Canvas global variables for Firebase use
        'app_id': app_id, 
        'firebase_config': firebase_config, 
        'initial_auth_token': initial_auth_token
    }

# =================================================================
# --- ROUTES (Authentication, Core, Features) ---
# =================================================================

@app.route('/register_user', methods=['GET', 'POST'])
def register_user():
    if 'email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = generate_password_hash(request.form.get('password'))
        phone = request.form.get('phone')
        address = request.form.get('address')
        village = request.form.get('village')
        pincode = request.form.get('pincode')

        # ✅ Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('This email is already registered.', 'error')
            return redirect(url_for('register_user'))

        # ✅ Create and save new user
        new_user = User(
            username=name,
            email=email,
            password=password,
            role='user',
            phone=phone,
            address=address,
            village=village,
            pincode=pincode
        )
        models_db.session.add(new_user)
        models_db.session.commit()

        flash('User account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register_user.html')


@app.route('/register_reclaimer', methods=['GET', 'POST'])
def register_reclaimer():
    if 'email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        company_name = request.form.get('company_name')
        email = request.form.get('email')
        password = generate_password_hash(request.form.get('password'))
        address = request.form.get('address')

        # ✅ Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('This email is already registered.', 'error')
            return redirect(url_for('register_reclaimer'))

        # ✅ Create and save new reclaimer
        new_reclaimer = User(
            username=company_name,
            email=email,
            password=password,
            role='reclaimer',
            company_name=company_name,
            address=address
        )
        models_db.session.add(new_reclaimer)
        models_db.session.commit()

        flash('Reclaimer account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register_reclaimer.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['email'] = user.email
            session['username'] = user.username
            session['user_role'] = user.role
            session['user_id'] = user.id  # ✅ This fixes "Authenticated User ID: Loading..."

            flash(f'Welcome back, {user.username}!', 'success')

            if user.role == 'user':
                return redirect(url_for('index'))
            elif user.role == 'reclaimer':
                return redirect(url_for('reclaimer_page'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear() 
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'email' not in session: return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/about')
def about():
    if 'email' not in session:
        flash('Please log in to view the site content.', 'info')
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if 'email' not in session:
        flash('Please log in to contact us.', 'info')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        flash('Thank you for your message! We will get back to you shortly.', 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        flash('Please log in to view your dashboard.', 'info')
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    # ✅ Fetch all uploads of this user
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.created_at.desc()).all()

    uploads_data = [
        {
            'time': u.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'image_url': url_for('static', filename=u.image_path),
            'material_label': u.label or 'Unknown',
            'is_recyclable': u.is_recyclable,
            'suggestions': json.loads(u.suggestions) if u.suggestions else [],
            'description': u.description or '',
            'location': u.location or ''
        }
        for u in uploads
    ]

    total_uploads = len(uploads)
    recyclable_count = len([u for u in uploads if u.is_recyclable])
    trash_count = total_uploads - recyclable_count
    rewards_earned = recyclable_count * 100  # 100 points per recyclable

    metrics = {
        'total_uploads': total_uploads,
        'recyclable_count': recyclable_count,
        'trash_count': trash_count,
        'rewards_earned': rewards_earned,
    }

    return render_template(
        'dashboard.html',
        username=session.get('username', 'Guest'),
        metrics=metrics,
        uploads=uploads_data
    )


@app.route('/directory')
def directory():
    if 'email' not in session:
        flash('Please log in to view the recycler directory.', 'info')
        return redirect(url_for('login'))
    return render_template('directory.html')

@app.route('/directory_data')
def directory_data():
    # AJAX endpoint for directory.js
    return jsonify(MOCK_RECYCLERS)

# --- NEW: Dedicated Results Route ---
@app.route('/results')
def results_page():
    # Retrieve prediction data from session, which was set in upload_file.
    # We use session.pop() to ensure the result is only retrieved once.
    prediction_data = session.pop('prediction_result', None)

    if not prediction_data:
        # If no data is found, redirect back to upload page
        flash('No classification result found. Please upload an image.', 'error')
        # NOTE: Using 'upload_file' route name, which renders 'upload.html'
        return redirect(url_for('upload_file'))

    # Prepare data for the template, making sure complex objects are JSON strings
    # for safe passage to JavaScript.
    
    # FIX: Pass a single JSON string containing all prediction data to simplify
    # JavaScript handling and prevent TemplateSyntaxErrors.
    prediction_data_json = json.dumps({
        'upload_id': prediction_data.get('upload_id'),  # include this line
        'image_url': prediction_data.get('image_url', url_for('static', filename='uploads/placeholder.jpg')),
        'material_label': prediction_data.get('label', 'Unknown').capitalize(),
        'recyclable': prediction_data.get('is_recyclable', False),
        'points_earned': prediction_data.get('points_earned', 0),
        'classification_details': prediction_data.get('classification_details', {}),
        'suggestions': prediction_data.get('suggestions', ['No specific guidance available.']),
        'error': prediction_data.get('error', None)
    })

    
    # We now pass ONLY the complete JSON string to the template
    return render_template('results.html', 
                            prediction_data_json=prediction_data_json)

@app.route('/reclaimer_page')
def reclaimer_page():
    if session.get('user_role') != 'reclaimer':
        flash('Access denied. This portal is for Reclaimers only.', 'error')
        return redirect(url_for('index'))
    return render_template('reclaimers_page.html', uploads_to_process=MOCK_UPLOADS_TO_PROCESS)


# =================================================================
# --- ERROR HANDLERS ---
# =================================================================

@app.errorhandler(BuildError)
def handle_build_error(e):
    if 'login' in str(e):
        flash("System initializing. Please log in.", "info")
        return redirect('/login')
    return str(e), 500

@app.route('/api/public_queue')
def public_queue():
    try:
        uploads = Upload.query.order_by(Upload.created_at.desc()).all()
        if not uploads:
            return jsonify({"status": "success", "uploads": []})

        data = [
            {
                "id": u.id,
                "material": u.label or "Unknown",
                "image_path": url_for('static', filename=u.image_path, _external=True),
                "submitted_by": User.query.get(u.user_id).username if u.user_id else "Unknown",
                "created_at": u.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            for u in uploads
        ]
        return jsonify({"status": "success", "uploads": data})

    except Exception as e:
        print("Queue error:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/get_current_user')
def get_current_user():
    try:
        if 'user_id' not in session:
            return jsonify({"status": "error", "message": "Not logged in"}), 401

        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        return jsonify({
            "status": "success",
            "user_id": user.id,
            "username": user.username,
            "role": user.role
        })
    except Exception as e:
        print("Error fetching current user:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/get_uploads')
def get_uploads():
    """Return all uploads for reclaimers."""
    try:
        uploads = Upload.query.order_by(Upload.created_at.desc()).all()
        upload_list = [
        {
            "id": u.id,
            "image_path": url_for('static', filename=u.image_path),
            "label": u.label,
            "username": User.query.get(u.user_id).username if u.user_id else "Unknown",
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "description": u.description,
            "collected_by": u.collected_by,
            "status": u.status,
            "company_info": u.company_info,
        }
        for u in uploads
    ]

        return jsonify({"success": True, "uploads": upload_list})
    except Exception as e:
        print("Error fetching uploads:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/collect_upload/<int:upload_id>', methods=['POST'])
def collect_upload(upload_id):
    """Mark an upload as collected and record recycler details."""
    try:
        upload = Upload.query.get(upload_id)
        if not upload:
            return jsonify({"success": False, "error": "Upload not found"}), 404

        recycler_id = session.get('user_id')
        recycler = User.query.get(recycler_id)

        if not recycler or recycler.role != 'reclaimer':
            return jsonify({"success": False, "error": "Unauthorized action"}), 403

        # ✅ Update collection details
        upload.collected_by = recycler.company_name or recycler.username
        upload.status = "Collected"
        upload.company_info = (
            f"Company: {recycler.company_name or recycler.username}\n"
            f"Address: {recycler.address or 'Not provided'}\n"
            f"Contact: {recycler.email}"
        )

        models_db.session.commit()

        return jsonify({
            "success": True,
            "message": "Waste collected successfully!",
            "collected_by": upload.collected_by,
            "status": upload.status,
            "company_info": upload.company_info
        })
    except Exception as e:
        print("Error in collect_upload:", e)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/get_collection_info/<int:upload_id>')
def get_collection_info(upload_id):
    """Return collection details for a specific upload."""
    try:
        upload = Upload.query.get(upload_id)
        if not upload:
            return jsonify({"success": False, "error": "Upload not found"}), 404

        return jsonify({
            "success": True,
            "collected_by": upload.collected_by,
            "status": upload.status,
            "company_info": upload.company_info
        })
    except Exception as e:
        print("Error fetching collection info:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/save_description', methods=['POST'])
def save_description():
    from models import Upload, db
    data = request.get_json()
    upload_id = data.get('upload_id')
    description = data.get('description', '').strip()

    upload = Upload.query.get(upload_id)
    if upload:
        upload.description = description
        db.session.commit()
        return jsonify({'success': True, 'message': 'Description saved successfully!'})
    else:
        return jsonify({'success': False, 'message': 'Upload not found.'}), 404

    
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    """Handle image upload, classification, and database storage."""
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    if request.method == 'GET':
        return render_template('upload.html')

    try:
        # Validate file
        if 'wasteImage' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400

        file = request.files['wasteImage']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        file.save(save_path)

        # Run ML classification
        result = classify_image(save_path)

        # Prepare info for database
        user_id = session.get('user_id')
        label = result.get('label', 'Unknown')
        is_recyclable = result.get('is_recyclable', False)
        suggestions = json.dumps(result.get('suggestions', []))
        description = f"{label} waste detected. {'Recyclable' if is_recyclable else 'Non-recyclable'}."

        # Store upload in database
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        relative_path = os.path.relpath(save_path, start=os.path.join(root_dir, 'static'))

        new_upload = Upload(
            user_id=user_id,
            image_path=relative_path.replace('\\', '/'),
            label=label,
            is_recyclable=is_recyclable,
            suggestions=suggestions,
            description=description
        )
        models_db.session.add(new_upload)
        models_db.session.commit()


        # ✅ Store prediction in session, now including upload_id
        session['prediction_result'] = {
            'upload_id': new_upload.id,  # important for save_description()
            'image_url': url_for('static', filename=relative_path.replace('\\', '/')),
            'label': label,
            'is_recyclable': is_recyclable,
            'points_earned': 100 if is_recyclable else 0,
            'suggestions': result.get('suggestions', []),
            'classification_details': result.get('classification_details', {}),
        }

        return jsonify({'success': True})


    except Exception as e:
        print("❌ Upload error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
