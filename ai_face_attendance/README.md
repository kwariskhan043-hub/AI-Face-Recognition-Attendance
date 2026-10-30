# AttendAI — Smart Face Recognition Attendance System

An academic computer-vision project built with Python, OpenCV (Haar Cascade + LBPH), Streamlit and Pandas.

## Features
- Student enrollment using multiple face images
- Face detection and LBPH-based recognition
- Duplicate attendance prevention per day
- Daily attendance dashboard and date-filtered CSV export
- Retrainable local recognition model

## Run locally
Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Register each student with at least 3 (ideally 5–10) clear photos, then use **Admin & Model → Train / Retrain**.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload this project.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch `main`, and main file `app.py`.
4. Deploy. The dependencies are listed in `requirements.txt`.
5. Test the deployed app with consented sample images.

## Important deployment note
Streamlit Community Cloud's local filesystem is not guaranteed to be persistent. This prototype stores enrollment photos, the trained model and attendance CSV under `data/`. For a real institution, connect persistent storage/database (for example, PostgreSQL or a managed database) and add authenticated administrator access before use.

## Privacy and responsible use
Face images are biometric data. Obtain explicit informed consent, limit collection, secure access, define a retention/deletion policy, and provide a manual attendance alternative. Do not use this prototype as the sole basis for disciplinary or high-stakes decisions. Recognition performance depends on image quality, lighting, camera angle and the training set.

## Technical overview
1. Haar Cascade locates a face in each uploaded/captured image.
2. The face is converted to grayscale, histogram-equalized and resized to 200×200.
3. LBPH learns texture-pattern histograms for enrolled identities.
4. At check-in, the closest LBPH prediction is accepted only under a configurable distance threshold.
5. Attendance is stored once per student per calendar day.

## Project structure
```
ai_face_attendance/
├── app.py
├── attendance.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
└── data/
    ├── students/
    └── models/
```

## Academic disclaimer
This is a learning prototype, not a certified biometric access-control system. Validate accuracy on a separate test set and compare results with manual ground truth before presenting performance claims.
