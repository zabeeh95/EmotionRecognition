import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array


class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25,
                 from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

        pt = tf.exp(-ce)
        focal_loss = self.alpha * tf.pow((1 - pt), self.gamma) * ce

        return focal_loss


model = models.load_model("emotion_model.keras", custom_objects={"FocalLoss": FocalLoss})

face_net = cv2.dnn.readNetFromCaffe("data/deploy.prototxt", "data/res10_300x300_ssd_iter_140000.caffemodel")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (x1, y1, x2, y2) = box.astype("int")

            faces.append((x1, y1, x2 - x1, y2 - y1))

    if len(faces) > 0:

        (x, y, fw, fh) = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, frame.shape[1] - x)
        fh = min(fh, frame.shape[0] - y)

        roi = frame_gray[y:y + fh, x:x + fw]

        if roi.size != 0:

            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]

            label = EMOTIONS[preds.argmax()]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                bar_width = int(prob * 300)

                cv2.rectangle(frame, (5, (i * 35) + 5), (bar_width, (i * 35) + 35), (0, 0, 255), -1)

                cv2.putText(frame, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
