import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models

# import argparse
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
# ap.add_argument("-m", "--model", required=True, help="path to pre-trained emotion detector CNN")
# ap.add_argument("-v", "--video", help="path to the (optional) video file")
# args = vars(ap.parse_args())
# detector = cv2.CascadeClassifier(args["cascade"])
# model = models.load_model(args["model"])

detector = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")
model = models.load_model('data\model.keras')

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangle = detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face_rectangle) > 0:
        # determine the largest face area
        rect = sorted(face_rectangle, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (x, y, w, h) = rect

        roi = frame_gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)  # shape (48,48,1)
        roi = np.expand_dims(roi, axis=0)  # shape (1,48,48,1)

        preds = model.predict(roi)[0]
        print(preds)
        label = EMOTIONS[preds.argmax()]

        # loop over the labels + probabilities and draw them
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(frame, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(frame, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    cv2.imshow("Camera Window", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
