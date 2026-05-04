from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory database (replace with real DB if needed)
users_db = {}

# Load model and preprocessing tools
model = load_model('eeg_lstm.h5')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Ensure plots folder exists
os.makedirs('static/plots', exist_ok=True)


@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        if len(phone) != 10:
            flash("Phone number must be 10 digits", "danger")
            return redirect(url_for('register'))

        if len(password) < 8:
            flash("Password must be at least 8 characters long", "danger")
            return redirect(url_for('register'))

        if username in users_db:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        users_db[username] = {
            'email': email,
            'phone': phone,
            'password': hashed_password
        }
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users_db and check_password_hash(users_db[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file.stream)

        # --- Intermediate Results ---
        raw_preview = df.head(1).to_html(classes='table table-bordered table-striped', index=False)

        # Scale
        X_scaled = scaler.transform(df.values)
        scaled_df = pd.DataFrame(X_scaled)
        scaled_preview = scaled_df.head(1).to_html(classes='table table-bordered table-striped', index=False)

        # Reshape
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        reshaped_df = pd.DataFrame(X_reshaped.reshape(X_reshaped.shape[0], -1))
        reshaped_preview = reshaped_df.head(1).to_html(classes='table table-bordered table-striped', index=False)

        # Prediction
        prediction = model.predict(X_reshaped)
        predicted_class = le.inverse_transform([np.argmax(prediction[0])])[0].capitalize()
        confidence = float(np.max(prediction)) * 100

        # Training accuracy plot
        plot_path = 'static/plots/training_accuracy.png'

        return render_template(
            'results.html',
            predicted_class=predicted_class,
            confidence=f"{confidence:.2f}%",
            raw_preview=raw_preview,
            scaled_preview=scaled_preview,
            reshaped_preview=reshaped_preview,
            plot_path=plot_path
        )

    return render_template('predict.html')


@app.route('/graphs')
def graphs():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('graphs.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)