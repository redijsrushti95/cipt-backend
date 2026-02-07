from flask import Flask, request, jsonify, session
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
import io
from PIL import Image
import threading
import queue
import os
import json
from datetime import datetime
from pathlib import Path
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production
#CORS(app)  # Enable CORS for React frontend
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ------------------ Body Posture Logic ------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_body_posture(image_data):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lt_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                rt_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Calculate angle
                angle = calculate_angle(elbow, lt_shoulder, rt_shoulder)
                
                # Determine posture status
                movement = "Optimal"
                if angle > 110:
                    movement = "Over Extended"
                elif angle < 90:
                    movement = "Crossed Arms"
                
                return {
                    "shoulder_angle": round(angle, 1),
                    "posture_status": movement,
                    "posture_score": min(100, max(0, 100 - abs(angle - 100))),
                    "confidence": 85,
                    "success": True
                }
    except Exception as e:
        print(f"Body posture error: {e}")
    
    return {"success": False, "error": "No pose detected"}

# ------------------ Eye Contact Logic ------------------
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    return math.sqrt((x1 - x)**2 + (y1 - y)**2)

def analyze_eye_contact(image_data):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        map_face_mesh = mp.solutions.face_mesh
        with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                img_height, img_width = frame.shape[:2]
                mesh_coords = [(int(point.x * img_width), int(point.y * img_height))
                             for point in results.multi_face_landmarks[0].landmark]
                
                # Calculate blink ratio
                rh_right = mesh_coords[RIGHT_EYE[0]]
                rh_left = mesh_coords[RIGHT_EYE[8]]
                rv_top = mesh_coords[RIGHT_EYE[12]]
                rv_bottom = mesh_coords[RIGHT_EYE[4]]
                
                lh_right = mesh_coords[LEFT_EYE[0]]
                lh_left = mesh_coords[LEFT_EYE[8]]
                lv_top = mesh_coords[LEFT_EYE[12]]
                lv_bottom = mesh_coords[LEFT_EYE[4]]
                
                rh_distance = euclidean_distance(rh_right, rh_left)
                rv_distance = euclidean_distance(rv_top, rv_bottom)
                lv_distance = euclidean_distance(lv_top, lv_bottom)
                lh_distance = euclidean_distance(lh_right, lh_left)
                
                re_ratio = rh_distance / rv_distance if rv_distance > 0 else 0
                le_ratio = lh_distance / lv_distance if lv_distance > 0 else 0
                ratio = (re_ratio + le_ratio) / 2 if (re_ratio + le_ratio) > 0 else 0
                
                # Simple gaze detection (simplified)
                eye_center_x = (mesh_coords[LEFT_EYE[0]][0] + mesh_coords[RIGHT_EYE[8]][0]) / 2
                face_center_x = img_width / 2
                
                gaze = "Center"
                if eye_center_x < face_center_x - 50:
                    gaze = "Left"
                elif eye_center_x > face_center_x + 50:
                    gaze = "Right"
                
                # Estimate eye contact percentage
                eye_contact = 75  # Base value
                if gaze == "Center":
                    eye_contact = 85
                elif ratio > 4:  # Blinking
                    eye_contact = 60
                
                return {
                    "gaze_direction": gaze,
                    "eye_contact_percentage": eye_contact,
                    "blink_ratio": round(ratio, 2),
                    "engagement_score": min(100, eye_contact + 10),
                    "success": True
                }
    except Exception as e:
        print(f"Eye contact error: {e}")
    
    return {"success": False, "error": "No face detected"}

# ------------------ Emotion Detection Logic ------------------
def analyze_emotion(image_data):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
            
            # Heuristic emotion detection
            emotion = "Neutral"
            if len(eyes) >= 2 and len(mouth) > 0:
                emotion = "Happy"
            elif len(eyes) >= 2 and len(mouth) == 0:
                emotion = "Fear"
            elif len(eyes) < 2 and len(mouth) == 0:
                emotion = "Sad"
            elif len(eyes) < 2 and len(mouth) > 0:
                emotion = "Angry"
            
            # Calculate confidence based on detections
            confidence = min(95, 70 + (len(eyes) * 10) + (len(mouth) * 5))
            
            return {
                "dominant_emotion": emotion,
                "confidence": round(confidence),
                "eyes_detected": len(eyes),
                "mouth_detected": len(mouth) > 0,
                "success": True
            }
    except Exception as e:
        print(f"Emotion detection error: {e}")
    
    return {"success": False, "error": "No face detected"}

# ------------------ API Routes ------------------
@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.json
    username = data.get("username")
    password = data.get("password")

  # For now, accept any login (in production, validate against database)
    if username and password:
        session['username'] = username
        session['logged_in'] = True
        return jsonify({"success": True, "username": username})
    else:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    # ✅ Handle CORS preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    # ✅ Handle actual register POST request
    data = request.json
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")

    # For now, accept any registration (in production, save to database)
    if username and password:
        session['username'] = username
        session['logged_in'] = True
        # --- your existing DB insert logic here ---
        return jsonify({"success": True, "username": username})
    else:
        return jsonify({"success": False, "error": "Registration failed"}), 400

@app.route("/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    session.clear()
    return jsonify({"success": True})


@app.route('/api/analyze/posture', methods=['POST'])
def analyze_posture():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    result = analyze_body_posture(data['image'])
    return jsonify(result)

@app.route('/api/analyze/eye', methods=['POST'])
def analyze_eye():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    result = analyze_eye_contact(data['image'])
    return jsonify(result)

@app.route('/api/analyze/emotion', methods=['POST'])
def analyze_emotion_route():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    result = analyze_emotion(data['image'])
    return jsonify(result)

@app.route('/api/analyze/voice', methods=['POST'])
def analyze_voice():
    # For voice analysis, you'd need audio data
    # This is a simplified version
    data = request.json
    if not data or 'audio' not in data:
        return jsonify({"error": "No audio data provided"}), 400
    
    # Simulate voice analysis
    return jsonify({
        "pitch": 180 + np.random.randint(-20, 20),
        "clarity": 85 + np.random.randint(-10, 10),
        "speech_rate": 150 + np.random.randint(-30, 30),
        "volume": 75 + np.random.randint(-15, 15),
        "success": True
    })

# ------------------ Report Management ------------------
REPORTS_DIR = Path("professional_reports")
REPORTS_DIR.mkdir(exist_ok=True)

def get_user_reports(username):
    """Get all PDF reports for a specific user"""
    user_reports = []
    if not REPORTS_DIR.exists():
        return user_reports

    # Look for PDF files that contain the username
    for pdf_file in REPORTS_DIR.glob("*.pdf"):
        if username.lower() in pdf_file.name.lower():
            # Extract timestamp from filename or use file modification time
            try:
                # Try to parse timestamp from filename (format: Name_Report_YYYYMMDD_HHMMSS.pdf)
                parts = pdf_file.stem.split('_')
                if len(parts) >= 3:
                    timestamp_str = f"{parts[-2]}_{parts[-1]}"
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                else:
                    timestamp = datetime.fromtimestamp(pdf_file.stat().st_mtime)
            except:
                timestamp = datetime.fromtimestamp(pdf_file.stat().st_mtime)

            user_reports.append({
                "name": pdf_file.stem.replace('_', ' '),
                "url": f"/reports/{pdf_file.name}",
                "createdAt": int(timestamp.timestamp() * 1000)  # milliseconds
            })

    # Sort by creation time (newest first)
    user_reports.sort(key=lambda x: x["createdAt"], reverse=True)
    return user_reports

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all reports for the current user"""
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session['username']
    reports = get_user_reports(username)
    return jsonify(reports)

@app.route('/api/analyze/generate-report', methods=['POST'])
def generate_report():
    """Generate an average PDF report for all user answers"""
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session['username']

    # Get all videos for this user
    if username not in user_videos or not user_videos[username]:
        return jsonify({"error": "No videos found for analysis"}), 404

    user_videos_list = user_videos[username]

    try:
        # Import the analysis module
        import sys
        sys.path.append('..')
        from advanced_report import EnhancedProfessionalVideoAnalyzer, EnhancedProfessionalReportGenerator

        all_results = []
        total_videos = len(user_videos_list)

        print(f"Analyzing {total_videos} videos for user {username}")

        # Analyze each video
        for i, video_info in enumerate(user_videos_list):
            try:
                video_path = video_info['localPath']
                print(f"Analyzing video {i+1}/{total_videos}: {video_path}")

                if not Path(video_path).exists():
                    print(f"Video file not found: {video_path}")
                    continue

                analyzer = EnhancedProfessionalVideoAnalyzer(video_path)
                results = analyzer.analyze_all()
                all_results.append(results)

            except Exception as e:
                print(f"Error analyzing video {video_info['filename']}: {str(e)}")
                continue

        if not all_results:
            return jsonify({"error": "No videos could be analyzed"}), 500

        # Calculate averages across all videos
        averaged_results = calculate_average_results(all_results)

        # Generate PDF report with averaged data
        user_info = {
            'name': username,
            'role': 'Interview Candidate',
            'total_videos_analyzed': len(all_results),
            'analysis_type': 'Average Performance Report'
        }

        generator = EnhancedProfessionalReportGenerator(averaged_results, user_info)
        pdf_path = generator.generate_report()

        if pdf_path:
            # Return the report info
            report_info = {
                "name": f"{username}_Average_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "url": f"/reports/{Path(pdf_path).name}",
                "createdAt": int(datetime.now().timestamp() * 1000)
            }
            return jsonify({"success": True, "report": report_info})
        else:
            return jsonify({"error": "Failed to generate report"}), 500

    except Exception as e:
        print(f"Report generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

def calculate_average_results(all_results):
    """Calculate average results across multiple video analyses"""
    if not all_results:
        return {}

    # Initialize averages
    averaged = {
        'posture': {'total_score': 0, 'count': 0, 'analysis_type': 'Average Analysis'},
        'facial': {'total_score': 0, 'count': 0, 'analysis_type': 'Average Analysis'},
        'eye_contact': {'total_score': 0, 'count': 0, 'analysis_type': 'Average Analysis'},
        'voice': {'total_score': 0, 'count': 0, 'analysis_type': 'Average Analysis'},
        'overall': {'total_score': 0, 'count': 0}
    }

    # Aggregate scores
    for result in all_results:
        if 'posture' in result and result['posture'].get('score'):
            averaged['posture']['total_score'] += result['posture']['score']
            averaged['posture']['count'] += 1

        if 'facial' in result and result['facial'].get('score'):
            averaged['facial']['total_score'] += result['facial']['score']
            averaged['facial']['count'] += 1

        if 'eye_contact' in result and result['eye_contact'].get('score'):
            averaged['eye_contact']['total_score'] += result['eye_contact']['score']
            averaged['eye_contact']['count'] += 1

        if 'voice' in result and result['voice'].get('score'):
            averaged['voice']['total_score'] += result['voice']['score']
            averaged['voice']['count'] += 1

        if 'overall' in result and result['overall'].get('score'):
            averaged['overall']['total_score'] += result['overall']['score']
            averaged['overall']['count'] += 1

    # Calculate averages
    for category in averaged:
        if averaged[category]['count'] > 0:
            averaged[category]['score'] = round(averaged[category]['total_score'] / averaged[category]['count'], 1)
        else:
            averaged[category]['score'] = 0

    # Add summary information
    averaged['summary'] = {
        'videos_analyzed': len(all_results),
        'average_overall_score': averaged['overall']['score'],
        'best_category': max(
            [('Posture', averaged['posture']['score']),
             ('Facial Expression', averaged['facial']['score']),
             ('Eye Contact', averaged['eye_contact']['score']),
             ('Voice', averaged['voice']['score'])],
            key=lambda x: x[1]
        )[0] if any([averaged['posture']['score'], averaged['facial']['score'],
                    averaged['eye_contact']['score'], averaged['voice']['score']]) else 'N/A'
    }

    return averaged

@app.route('/reports/<filename>', methods=['GET'])
def serve_report(filename):
    """Serve PDF reports"""
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session['username']
    file_path = REPORTS_DIR / filename

    # Security check: ensure the file belongs to the current user
    if not file_path.exists() or username.lower() not in filename.lower():
        return jsonify({"error": "Report not found"}), 404

    from flask import send_file
    return send_file(file_path, as_attachment=False, mimetype='application/pdf')

# ------------------ Video Upload and Management ------------------
ANSWERS_DIR = Path("answers")
ANSWERS_DIR.mkdir(exist_ok=True)

# Store user video sessions (in production, use database)
user_videos = {}

@app.route('/upload-answer', methods=['POST'])
def upload_answer():
    """Upload user answer video"""
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session['username']

    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    question = request.form.get('question', 'unknown')
    domain = request.form.get('domain', 'unknown')

    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{username}_{domain}_q{question}_{timestamp}_{unique_id}.webm"
    file_path = ANSWERS_DIR / filename

    try:
        video_file.save(str(file_path))

        # Store video info for this user
        if username not in user_videos:
            user_videos[username] = []

        video_info = {
            'filename': filename,
            'localPath': str(file_path),
            'question': question,
            'domain': domain,
            'timestamp': timestamp,
            'username': username
        }

        user_videos[username].append(video_info)

        # Keep only the latest 10 videos per user
        if len(user_videos[username]) > 10:
            user_videos[username] = user_videos[username][-10:]

        return jsonify({
            "success": True,
            "message": "Video uploaded successfully",
            "filename": filename,
            "path": str(file_path)
        })

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/get-latest-video', methods=['GET'])
def get_latest_video():
    """Get the latest uploaded video for the current user"""
    if 'username' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    username = session['username']

    if username not in user_videos or not user_videos[username]:
        return jsonify({"error": "No videos found"}), 404

    # Get the most recent video
    latest_video = max(user_videos[username], key=lambda x: x['timestamp'])

    return jsonify(latest_video)

@app.route('/videos/<path:filename>', methods=['GET'])
def serve_video(filename):
    """Serve video files"""
    from flask import send_file
    video_path = Path("backend/videos") / filename
    if not video_path.exists():
        video_path = Path("videos") / filename

    if not video_path.exists():
        return jsonify({"error": "Video not found"}), 404

    return send_file(str(video_path), mimetype='video/mp4')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Python backend is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
