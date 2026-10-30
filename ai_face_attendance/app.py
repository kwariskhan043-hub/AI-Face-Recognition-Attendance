import os
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from attendance import (
    DATA_DIR, STUDENTS_DIR, MODEL_PATH, LABELS_PATH, ATTENDANCE_PATH,
    ensure_dirs, load_labels, train_model, recognize_face, mark_attendance,
    get_today_attendance, save_student_image, valid_student_id
)

st.set_page_config(page_title="AttendAI | Smart Attendance", page_icon="🎓", layout="wide")
ensure_dirs()

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(115deg,#102a43,#176b87); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#d9edf7; margin:.45rem 0 0 0;}
.small-note {color: #6b7280; font-size: .88rem;}
</style>
<div class="hero">
<h1>🎓 AttendAI — Smart Face Recognition Attendance</h1>
<p>Computer vision • Automated attendance • Daily reports</p>
</div>
""", unsafe_allow_html=True)

labels = load_labels()
student_count = len(labels)
today_df = get_today_attendance()

c1, c2, c3 = st.columns(3)
c1.metric("Registered students", student_count)
c2.metric("Today's attendance", len(today_df))
c3.metric("Date", date.today().strftime("%d %b %Y"))

tab_att, tab_register, tab_report, tab_admin = st.tabs(
    ["📸 Mark Attendance", "➕ Register Student", "📊 Reports", "⚙️ Admin & Model"]
)

with tab_att:
    st.subheader("Mark attendance")
    st.write("Take a clear, front-facing photo in good lighting. The system compares it with enrolled faces.")
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.warning("No trained face model found. Register students and train the model in the Admin tab first.")
    else:
        photo = st.camera_input("Take a photo")
        uploaded = st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"], key="attendance_upload")
        image_source = photo if photo is not None else uploaded
        threshold = st.slider("Recognition strictness", 35, 100, 65, help="Lower values are stricter for LBPH distance.")
        if image_source is not None:
            image = Image.open(image_source).convert("RGB")
            st.image(image, caption="Captured image", width=320)
            if st.button("Identify & mark attendance", type="primary"):
                result = recognize_face(np.array(image), threshold=threshold)
                if result["status"] == "ok":
                    status = mark_attendance(result["student_id"], result["name"])
                    st.success(f"Recognized: {result['name']} ({result['student_id']})")
                    if status == "already":
                        st.info("Attendance was already recorded for this student today.")
                    else:
                        st.success("Attendance marked successfully.")
                elif result["status"] == "multiple":
                    st.warning("More than one face detected. Please submit an image with only one person.")
                elif result["status"] == "no_face":
                    st.error("No face detected. Try better lighting and a front-facing photo.")
                else:
                    st.error("Face not confidently recognized. Please try again or contact the administrator.")

with tab_register:
    st.subheader("Register a student")
    st.info("For a useful model, enroll each student with 5–10 clear photos captured from slightly different angles.")
    with st.form("register_form"):
        sid = st.text_input("Student ID", placeholder="e.g. BSCS-2026-014")
        name = st.text_input("Full name", placeholder="e.g. Ali Ahmed")
        pics = st.file_uploader("Upload face photos (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Save student photos")
    if submitted:
        if not valid_student_id(sid):
            st.error("Use a student ID containing only letters, numbers, hyphens or underscores.")
        elif not name.strip():
            st.error("Please enter the student's name.")
        elif len(pics) < 3:
            st.error("Please upload at least 3 photos for more reliable recognition.")
        else:
            saved, rejected = save_student_image(sid.strip(), name.strip(), pics)
            if saved:
                st.success(f"Saved {saved} face photo(s) for {name.strip()}.")
            if rejected:
                st.warning(f"{rejected} photo(s) did not contain exactly one detectable face and were skipped.")
            st.caption("Next: open Admin & Model and train/retrain the model.")

with tab_report:
    st.subheader("Attendance reports")
    df = pd.read_csv(ATTENDANCE_PATH) if ATTENDANCE_PATH.exists() else pd.DataFrame(columns=["date","time","student_id","name"])
    if not df.empty:
        df["date"] = df["date"].astype(str)
        min_d, max_d = pd.to_datetime(df["date"]).min().date(), pd.to_datetime(df["date"]).max().date()
        selected = st.date_input("Filter date range", value=(min_d, max_d))
        if isinstance(selected, tuple) and len(selected) == 2:
            start, end = selected
            view = df[(pd.to_datetime(df["date"]).dt.date >= start) & (pd.to_datetime(df["date"]).dt.date <= end)]
        else:
            view = df[df["date"] == str(selected)]
        st.dataframe(view.sort_values(["date","time"], ascending=False), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "attendance_report.csv", "text/csv")
        st.subheader("Daily attendance count")
        daily = view.groupby("date").size().reset_index(name="attendance_count")
        st.bar_chart(daily.set_index("date"))
    else:
        st.info("No attendance records yet. Mark attendance to populate reports.")

with tab_admin:
    st.subheader("Model management")
    st.write("The model uses OpenCV Haar face detection and LBPH face recognition. Train after adding or updating student photos.")
    if st.button("Train / Retrain recognition model", type="primary"):
        with st.spinner("Training model from enrolled student photos..."):
            count, errors = train_model()
        if count:
            st.success(f"Training complete. Processed {count} face image(s) across {len(load_labels())} student(s).")
        else:
            st.error("No usable face images found. Register students with at least 3 clear photos.")
        if errors:
            st.warning(f"{errors} image(s) were skipped because no single face could be detected.")
    if MODEL_PATH.exists():
        st.success("A trained model is available.")
    else:
        st.warning("No trained model available yet.")
    st.markdown("---")
    st.caption("Admin setup: set ADMIN_PASSWORD in Streamlit Secrets. The admin section can be protected further before production use.")
    with st.expander("Privacy, limitations & responsible use"):
        st.markdown("""
- Obtain informed consent before collecting or processing face images.
- Store student photos and model files securely; restrict access to authorized staff.
- Face recognition can make mistakes. Keep a manual attendance correction process.
- Use this as an academic prototype, not as the sole source for high-stakes attendance decisions.
- The app stores attendance in a local CSV. Streamlit Community Cloud storage may reset on redeploy/restart; use a persistent database for institutional deployment.
""")

st.markdown("---")
st.markdown('<p class="small-note">AttendAI academic prototype • Built with Python, OpenCV, Streamlit and Pandas</p>', unsafe_allow_html=True)
