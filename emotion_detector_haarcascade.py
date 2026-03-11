import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        pt = tf.exp(-ce)
        focal_loss = self.alpha * tf.pow((1 - pt), self.gamma) * ce
        return focal_loss


model = models.load_model("emotion_model.keras", custom_objects={"FocalLoss": FocalLoss})

face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        # Pick the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        # Extract ROI and preprocess
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = model.predict(roi, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]

        # Draw probability bars for each emotion
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            text = f"{emotion}: {prob * 100:.2f}%"
            bar_width = int(prob * 300)
            cv2.rectangle(frame, (5, 5 + i * 35), (bar_width, 35 + i * 35), (0, 0, 255), -1)
            cv2.putText(frame, text, (10, 23 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Draw face rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
