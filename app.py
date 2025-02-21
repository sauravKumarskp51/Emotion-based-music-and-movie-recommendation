from __future__ import division, print_function
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import statistics as st

app = Flask(__name__)


# Load model and face cascade
model = tf.keras.models.load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')

def detect_emotion(frame):
    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    output = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_face, (48, 48))
        reshaped = resized.reshape(1, 48, 48, 1) / 255.0
        predictions = model.predict(reshaped)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        output.append(predicted_emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), GR_dict[1], 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), GR_dict[1], -1)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, output

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame, _ = detect_emotion(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/")
def home():
    return render_template("index1.html")


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template("emotion_detection.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion', methods=['GET'])
def detect_emotion_route():
    cap = cv2.VideoCapture(0)
    output_emotions = []
    for _ in range(30):  # Capture 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        _, output = detect_emotion(frame)
        output_emotions.extend(output)
    cap.release()
    dominant_emotion = st.mode(output_emotions)
    return jsonify(emotion=dominant_emotion)

@app.route('/show_buttons')
def show_buttons():
    emotion = request.args.get('emotion')
    return render_template("buttons.html", final_output=emotion)

@app.route('/movies/surprise', methods=['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods=['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods=['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods=['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesSad.html")

@app.route('/movies/happy', methods=['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods=['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods=['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")





@app.route('/songs/surprise', methods=['GET', 'POST'])
def songsSurprise():
    return render_template("songsNeutral.html")

@app.route('/songs/angry', methods=['GET', 'POST'])
def songsAngry():
    return render_template("songsSoothing.html")

@app.route('/songs/sad', methods=['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods=['GET', 'POST'])
def songsDisgust():
    return render_template("songsSad.html")

@app.route('/songs/happy', methods=['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods=['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods=['GET', 'POST'])
def songsNeutral():
    return render_template("songsNeutral.html")

@app.route('/templates/join_page', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")

@app.route('/templates/contact', methods=['GET', 'POST'])
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
