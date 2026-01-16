from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
import os
import pickle
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'ABCDEF'  # change in production

# -------------------------
# DB config
# -------------------------
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'final-project'
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def db_init():
    """Create only what we need (histry). User table already exists in your DB."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS histry (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            filename VARCHAR(255) NOT NULL,
            caption TEXT,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

db_init()

# -------------------------
# Folders
# -------------------------
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROFILE_FOLDER = os.path.join('static', 'profiles')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROFILE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER

# -------------------------
# Model assets
# -------------------------
MODEL_PATH = "best_model_efficientnetb4.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH_PATH = "max_length.pkl"

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(MAX_LENGTH_PATH, "rb") as f:
    max_length = pickle.load(f)

base_model = EfficientNetB4(weights="imagenet")
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# -------------------------
# Caption helpers
# -------------------------
def idx_to_word(integer, tok):
    for w, i in tok.word_index.items():
        if i == integer:
            return w
    return None

@tf.function
def predict_caption_tf(m, image_feature, sequence):
    yhat = m([image_feature, sequence], training=False)
    return tf.argmax(yhat, axis=-1)

def predict_caption(m, image_feature, tok, max_len):
    in_text = "startseq"
    for _ in range(max_len):
        seq = tok.texts_to_sequences([in_text])[0]
        if len(seq) < max_len:
            seq = [0]*(max_len - len(seq)) + seq
        else:
            seq = seq[-max_len:]
        seq = np.array(seq).reshape(1, max_len).astype(np.int32)

        yhat = predict_caption_tf(m, image_feature, seq).numpy()[0]
        word = idx_to_word(yhat, tok)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB").resize((380, 380))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = feature_model.predict(arr, verbose=0)
    return feat

# -------------------------
# User helpers (users table already exists)
# columns expected (minimum): id, fullname, email, password, mobile, dob, profile_image, theme, accent_color, language
# -------------------------
def get_user(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user WHERE email=%s", (email,))
    user = cur.fetchone()
    conn.close()
    return user

def get_user_by_id(uid):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM user WHERE id=%s", (uid,))
    user = cur.fetchone()
    conn.close()
    return user

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/demo", methods=["GET", "POST"])
def demo():
    # Check if demo limit is reached
    if session.get("demo_used"):
        if "user_id" in session:
            # Logged-in users → go to main page
            return redirect(url_for("main_app"))
        else:
            # Not logged-in users → go to registration page
            flash("Demo limit reached. Please sign up to continue.", "warning")
            return redirect(url_for("register"))

    caption = None
    filename = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No image uploaded!", "danger")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected!", "danger")
            return redirect(request.url)

        # Save demo image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "demo_" + filename)
        file.save(filepath)

        # Generate AI caption
        features = extract_features(filepath)
        caption = predict_caption(model, features, tokenizer, max_length)

        # Mark demo as used
        session["demo_used"] = True

    return render_template("demo.html", caption=caption, filename=filename)






@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")

        # ----------------------------
        # MySQL insert
        # ----------------------------
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO contact (name, email, subject, message)
                VALUES (%s, %s, %s, %s)
            """, (name, email, subject, message))
            conn.commit()
            conn.close()
            flash("Message sent successfully!", "success")
        except Exception as e:
            flash(f"Failed to send message: {e}", "danger")

    return render_template("contact.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    # Registration template/flow will always show, even if logged in
    if request.method == "POST":
        fullname = request.form["fullname"].strip()
        email = request.form["email"].strip()
        password = request.form["password"].strip()
        mobile = request.form["mobile"].strip()
        dob = request.form["dob"]

        errors = []
        if not mobile.isdigit() or len(mobile) != 11:
            errors.append("Mobile number must be exactly 11 digits!")
        
        if len(password) < 6:
            errors.append("Password must be at least 6 characters long!")

        if get_user(email):
            errors.append("This email is already registered!")

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("register.html")

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            hashed = generate_password_hash(password)
            # new columns will take defaults (theme/accent_color/language) and profile_image = NULL
            cur.execute("""
                INSERT INTO user (fullname, email, password, mobile, dob)
                VALUES (%s,%s,%s,%s,%s)
            """, (fullname, email, hashed, mobile, dob))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except mysql.connector.IntegrityError:
            flash("Registration failed due to database error!", "danger")

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("main_app"))
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"].strip()
        user = get_user(email)
        # original order preserved; password is still index 3
        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            return redirect(url_for("main_app"))
        else:
            flash("Incorrect username or password. Please try again", "danger")
    return render_template("login.html")

@app.route("/main", methods=["GET", "POST"])
def main_app():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    username = session.get("username")
    filename = None
    caption = None

    user_row = get_user_by_id(uid)
    # fallback defaults if columns empty
    prefs = {
        "theme": user_row.get("theme") or "system",
        "accent": user_row.get("accent_color") or "purple",
        "language": user_row.get("language") or "bn"
    }
    profile_image = user_row.get("profile_image")

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part!", "danger")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No selected file!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        features = extract_features(filepath)
        caption = predict_caption(model, features, tokenizer, max_length)

        # save to history
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO histry (user_id, filename, caption, created_at) VALUES (%s,%s,%s,%s)",
            (uid, filename, caption, datetime.now())
        )
        conn.commit()
        conn.close()

    # last 10 items
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM histry WHERE user_id=%s ORDER BY created_at DESC LIMIT 10", (uid,))
    histry = cur.fetchall()
    conn.close()

    return render_template("index.html",
                           username=username,
                           filename=filename,
                           caption=caption,
                           prefs=prefs,
                           profile_image=profile_image,
                           histry=histry)

# ------------- General prefs (user table) -------------
@app.route("/prefs/update", methods=["POST"])
def update_prefs():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]
    theme = request.form.get("theme")
    accent = request.form.get("accent")
    language = request.form.get("language")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE user
        SET theme = COALESCE(%s, theme),
            accent_color = COALESCE(%s, accent_color),
            language = COALESCE(%s, language)
        WHERE id=%s
    """, (theme, accent, language, uid))
    conn.commit()
    conn.close()
    flash("Preferences updated!", "success")
    return redirect(url_for("main_app"))

# ------------- Profile image (users.profile_image) -------------
@app.route("/profile/upload", methods=["POST"])
def upload_profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]
    file = request.files.get("profile_image")
    if not file or file.filename == "":
        flash("No profile image selected!", "danger")
        return redirect(url_for("main_app"))

    fname = secure_filename(file.filename)
    save_path = os.path.join(app.config["PROFILE_FOLDER"], f"{uid}_{fname}")
    file.save(save_path)
    rel_path = f"profiles/{uid}_{fname}"  # relative to /static

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE user SET profile_image=%s WHERE id=%s", (rel_path, uid))
    conn.commit()
    conn.close()
    flash("Profile photo updated!", "success")
    return redirect(url_for("main_app"))

@app.route("/profile/remove", methods=["POST"])
def remove_profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]

    # delete file if exists
    row = get_user_by_id(uid)
    if row and row.get("profile_image"):
        try:
            os.remove(os.path.join("static", row["profile_image"]))
        except Exception:
            pass

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE user SET profile_image=NULL WHERE id=%s", (uid,))
    conn.commit()
    conn.close()
    flash("Profile photo removed.", "success")
    return redirect(url_for("main_app"))

# ------------- Password change -------------
@app.route("/password/change", methods=["POST"])
def change_password():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]

    current = request.form.get("current_password", "")
    newp = request.form.get("new_password", "")
    conf = request.form.get("confirm_password", "")

    if newp != conf or len(newp) < 6:
        flash("Password mismatch or too short (min 6).", "danger")
        return redirect(url_for("main_app"))

    user = get_user_by_id(uid)
    if not user or not check_password_hash(user["password"], current):
        flash("Current password incorrect.", "danger")
        return redirect(url_for("main_app"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE user SET password=%s WHERE id=%s",
                (generate_password_hash(newp), uid))
    conn.commit()
    conn.close()
    flash("Password changed successfully!", "success")
    return redirect(url_for("main_app"))

# ------------- Histry + account + logout -------------
@app.route("/histry/clear", methods=["POST"])
def clear_history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM histry WHERE user_id=%s", (uid,))
    conn.commit()
    conn.close()
    flash("Histry cleared.", "success")
    return redirect(url_for("main_app"))

@app.route("/account/delete", methods=["POST"])
def account_delete():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM user WHERE id=%s", (uid,))
    conn.commit()
    conn.close()
    session.clear()
    flash("Account deleted.", "success")
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home_page"))

if __name__ == "__main__":
    app.run(debug=True)