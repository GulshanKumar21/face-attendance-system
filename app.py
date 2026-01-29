import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Punch In", "Punch Out"]).to_csv(
        ATTENDANCE_FILE, index=False
    )

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("🎯 Face Authentication Attendance System")

menu = st.sidebar.selectbox("Select Option", ["Register Face", "Mark Attendance"])

if menu == "Register Face":
    name = st.text_input("Enter Name")

    if st.button("Capture Face"):
        if not name:
            st.warning("Please enter name")
        else:
            user_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(user_dir, exist_ok=True)

            cap = cv2.VideoCapture(0)
            count = 0
            st.info("Capturing 20 face images...")

            while count < 20:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    count += 1
                    face = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"{user_dir}/{count}.jpg", face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                cv2.imshow("Register Face", frame)
                cv2.waitKey(1)

            cap.release()
            cv2.destroyAllWindows()
            st.success("Face registered successfully")

def train_model():
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for user in os.listdir(DATASET_DIR):
        label_map[label_id] = user
        user_dir = os.path.join(DATASET_DIR, user)

        for img in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img)
            faces.append(cv2.imread(img_path, 0))
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    return label_map

if menu == "Mark Attendance":

    label_map = train_model()

    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        matched_name = None
        frame_count = 0

        st.info("Press ESC to stop camera")

        while frame_count < 200:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                label, confidence = recognizer.predict(face)

                if confidence < 70:
                    matched_name = label_map[label]
                    cv2.putText(
                        frame,
                        matched_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if matched_name:
            today = datetime.now().strftime("%Y-%m-%d")
            now = datetime.now().strftime("%H:%M:%S")

            df = pd.read_csv(ATTENDANCE_FILE)
            record = df[(df["Name"] == matched_name) & (df["Date"] == today)]

            if record.empty:
                df.loc[len(df)] = [matched_name, today, now, ""]
                st.success(f"✅ Punch In recorded for {matched_name}")
            else:
                idx = record.index[0]
                df.at[idx, "Punch Out"] = now
                st.success(f"✅ Punch Out recorded for {matched_name}")

            df.to_csv(ATTENDANCE_FILE, index=False)
        else:
            st.error("❌ Face not recognized")
