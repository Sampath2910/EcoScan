from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

# ------------------------
# User Table
# ------------------------
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')

    # User-specific fields
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    village = db.Column(db.String(100), nullable=True)
    pincode = db.Column(db.String(20), nullable=True)

    # Reclaimer-specific fields
    company_name = db.Column(db.String(150), nullable=True)
    company_info = db.Column(db.Text, nullable=True)  # ✅ Added

    uploads = db.relationship('Upload', backref='user', lazy=True, cascade="all, delete-orphan")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ------------------------
# Upload Table
# ------------------------
class Upload(db.Model):
    __tablename__ = 'uploads'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    image_path = db.Column(db.String(300), nullable=False)
    label = db.Column(db.String(100), nullable=True)
    is_recyclable = db.Column(db.Boolean, nullable=True)
    suggestions = db.Column(db.Text, nullable=True)

    waste_type = db.Column(db.String(100), nullable=True)
    description = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(200), nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # ✅ New columns for collection tracking
    collected_by = db.Column(db.String(150), nullable=True)
    status = db.Column(db.String(50), nullable=True, default="Pending")
    company_info = db.Column(db.Text, nullable=True)

    def suggestions_list(self):
        try:
            return json.loads(self.suggestions or "[]")
        except Exception:
            return []

    def __repr__(self):
        return f"<Upload {self.label} by User {self.user_id}>"


# ------------------------
# Contact Table
# ------------------------
class Contact(db.Model):
    __tablename__ = 'contacts'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Contact {self.name} ({self.email})>"
    
class RecycledProduct(db.Model):
    __tablename__ = 'recycled_products'

    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(300), nullable=False)
    company = db.Column(db.String(150), nullable=False)
    waste_used = db.Column(db.String(100), nullable=False)
    products = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RecycledProduct {self.company}>"
    
class RecyclerReview(db.Model):
    __tablename__ = 'recycler_reviews'

    id = db.Column(db.Integer, primary_key=True)

    recycler_name = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    rating = db.Column(db.Integer, nullable=False)  # 1 to 5
    review = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='recycler_reviews')

    def __repr__(self):
        return f"<Review {self.recycler_name} ⭐{self.rating}>"


