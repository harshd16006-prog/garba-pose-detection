import numpy as np
import joblib

def angle_between(a, b, c):
    """Calculate angle at point b between points a and c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def normalize_pose(pose):
    """Normalize pose by hip center and shoulder distance"""
    pose = pose.reshape(33, -1)
    hip_center = (pose[23, :2] + pose[24, :2]) / 2
    pose[:, :2] = pose[:, :2] - hip_center
    shoulder_dist = np.linalg.norm(pose[11, :2] - pose[12, :2])
    if shoulder_dist > 0:
        pose[:, :2] = pose[:, :2] / shoulder_dist
    return pose

# Landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

def score_and_feedback(user_pose, ref_pose, tolerance=15):
    """Compare user pose with reference pose"""
    user_pose = normalize_pose(user_pose)
    ref_pose = normalize_pose(ref_pose)
    
    user_pose = user_pose.reshape(33, -1)
    ref_pose = ref_pose.reshape(33, -1)
    
    joints = {
        "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        "left_shoulder": (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
        "right_shoulder": (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
        "left_knee": (LEFT_HIP, LEFT_KNEE, 27),
        "right_knee": (RIGHT_HIP, RIGHT_KNEE, 28),
    }
    
    total_score = 0
    feedback = []
    
    for joint_name, (p1, p2, p3) in joints.items():
        user_angle = angle_between(user_pose[p1, :2], user_pose[p2, :2], user_pose[p3, :2])
        ref_angle = angle_between(ref_pose[p1, :2], ref_pose[p2, :2], ref_pose[p3, :2])
        
        angle_diff = abs(user_angle - ref_angle)
        score = max(0, 1 - angle_diff / 90)
        total_score += score
        
        if angle_diff > tolerance:
            side, part = joint_name.split("_")
            if angle_diff > 20:
                if "elbow" in part:
                    msg = f"{'Straighten' if user_angle < ref_angle else 'Bend'} your {side} elbow"
                elif "shoulder" in part:
                    msg = f"{'Raise' if user_angle < ref_angle else 'Lower'} your {side} arm"
                else:
                    msg = f"{'Straighten' if user_angle < ref_angle else 'Bend'} your {side} leg"
                feedback.append(msg)
    
    final_score = round((total_score / len(joints)) * 100)
    return final_score, feedback

if __name__ == "__main__":
    # Test
    ref_poses = joblib.load("reference_poses.pkl")
    print("Feedback system ready!")
    print(f"Reference steps: {list(ref_poses.keys())}")