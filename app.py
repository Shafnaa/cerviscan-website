from flask import Flask, render_template, request, url_for, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
import os
import cv2
from werkzeug.utils import secure_filename
from model.rgb_to_gray import rgb_to_gray_converter
from model.multiotsu_segmentation import multiotsu_masking
from model.bitwise_operation import get_segmented_image
from model.cerviscan_feature_extraction import get_cerviscan_features
import pickle
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SECRET_KEY"] = "abc"
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['PROCESSED_FOLDER'] = './static/processed'

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "You need to log in to access this page."
login_manager.login_message_category = "info"
login_manager.init_app(app)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# User model
class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

# History model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(250), nullable=False)
    dob = db.Column(db.String(50), nullable=False)
    original = db.Column(db.String(250), nullable=False)
    gray = db.Column(db.String(250), nullable=False)
    mask = db.Column(db.String(250), nullable=False)
    segmented = db.Column(db.String(250), nullable=False)
    features = db.Column(db.PickleType, nullable=False)
    prediction = db.Column(db.String(250), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

# Routes for user management
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        password_confirm = request.form.get("password_confirm")

        if not username or not password:
            flash("Username and password cannot be empty", "danger")
            return render_template("register.html")
        if password != password_confirm:
            flash("Passwords do not match", "danger")
            return render_template("register.html")
        if Users.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)
        user = Users(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = Users.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            flash("Login successful!", "success")
            return redirect(next_page) if next_page else redirect(url_for("index"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# Main application routes
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result = None
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        dob = request.form['dob']
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_path)

            gray_image = rgb_to_gray_converter(original_path)
            gray_path = os.path.join(app.config['PROCESSED_FOLDER'], f'gray_{filename}')
            cv2.imwrite(gray_path, gray_image)

            mask_image = multiotsu_masking(gray_image)
            mask_path = os.path.join(app.config['PROCESSED_FOLDER'], f'mask_{filename}')
            cv2.imwrite(mask_path, mask_image)

            original_image = cv2.imread(original_path)
            segmented_image = get_segmented_image(original_image, mask_image)
            segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], f'segmented_{filename}')
            cv2.imwrite(segmented_path, segmented_image)

            image_features = get_cerviscan_features(segmented_path)
            model = pickle.load(open('./model/xgb_best', 'rb'))
            prediction = model.predict(image_features)
            print(prediction)
            if prediction[0] == 0:
                prediction = "normal"
            else:
                prediction = "abnormal"
            print(prediction)

            entry = History(
                user_id=current_user.id,
                name=f"{first_name} {last_name}",
                dob=dob,
                original=original_path,
                gray=gray_path,
                mask=mask_path,
                segmented=segmented_path,
                features=image_features,
                prediction=prediction,
                date=datetime.now()
            )
            db.session.add(entry)
            db.session.commit()

            result = entry

    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template('index.html', result=result, history=user_history)

@app.route('/history', methods=['GET'])
@login_required
def history_page():
    user_history = History.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', history=user_history)

@app.route('/history/<int:id>', methods=['GET'])
@login_required
def history_detail(id):
    entry = History.query.get_or_404(id)
    if entry.user_id != current_user.id:
        flash("You are not authorized to view this entry", "danger")
        return redirect(url_for('history_page'))
    return render_template('detail.html', entry=entry)

@app.route('/delete/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    entry = History.query.get_or_404(id)
    if entry.user_id != current_user.id:
        flash("You are not authorized to delete this entry", "danger")
        return redirect(url_for('history_page'))
    db.session.delete(entry)
    db.session.commit()
    flash("History deleted successfully.", "success")
    return redirect(url_for('history_page'))

@app.route('/usage')
@login_required
def usage():
    return render_template('usage.html')

if __name__ == '__main__':
    app.run()
