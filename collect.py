import cv2
import mediapipe as mp
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

csv_file = "../data/garba.csv"
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = []
        for i in range(33):
            headers.extend([f"x{i}", f"y{i}", f"z{i}", f"conf{i}"])
        headers.append("label")
        writer.writerow(headers)

cap = cv2.VideoCapture(0)
current_step = None
frame_count = 0

print("\n=== GARBA POSE COLLECTION ===")
print("Press 't' for TAALI")
print("Press 'd' for DODHIYU")
print("Press 'h' for HINCH")
print("Press 'q' to QUIT")
print("Press 'r' to RESET\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    status_text = f"RECORDING: {current_step.upper()} | Frames: {frame_count}" if current_step else "WAITING FOR INPUT | Press T/D/H"
    color = (0, 255, 0) if current_step else (0, 165, 255)
    
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, "T=Taali  D=Dodhiyu  H=Hinch  Q=Quit  R=Reset", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Garba Pose Collection", frame)
    
    if results.pose_landmarks and current_step:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = []
            for landmark in results.pose_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            row.append(current_step)
            writer.writerow(row)
        frame_count += 1
        print(f"Recorded {current_step}: Frame {frame_count}")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        current_step = None
        frame_count = 0
        print(">>> RESET! Press T/D/H to start recording\n")
    elif key == ord('t'):
        current_step = "taali"
        frame_count = 0
        print("\n>>> START RECORDING TAALI\n")
    elif key == ord('d'):
        current_step = "dodhiyu"
        frame_count = 0
        print("\n>>> START RECORDING DODHIYU\n")
    elif key == ord('h'):
        current_step = "hinch"
        frame_count = 0
        print("\n>>> START RECORDING HINCH\n")

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Recording complete! Total frames: {frame_count}")