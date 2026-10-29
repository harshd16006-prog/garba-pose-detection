import numpy as np
import joblib

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
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

TOLERANCE = 15       # itne degree tak galti maaf
SCORE_RANGE = 60     # itne degree ki galti par score 0 ho jata hai
MAX_TIPS = 3         # ek baar mein zyada se zyada kitne tips


def angle_between(a, b, c):
    """Calculate angle at point b between points a and c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def normalize_pose(pose):
    """Normalize pose by hip center and shoulder distance (original ko nahi badalta)"""
    pose = np.array(pose, dtype=float).reshape(33, -1).copy()
    hip_center = (pose[LEFT_HIP, :2] + pose[RIGHT_HIP, :2]) / 2
    pose[:, :2] = pose[:, :2] - hip_center
    shoulder_dist = np.linalg.norm(pose[LEFT_SHOULDER, :2] - pose[RIGHT_SHOULDER, :2])
    if shoulder_dist > 0:
        pose[:, :2] = pose[:, :2] / shoulder_dist
    return pose


def score_and_feedback(user_pose, ref_pose, tolerance=TOLERANCE):
    """Compare user pose with reference pose. Returns (score 0-100, feedback list)"""
    user_pose = normalize_pose(user_pose)
    ref_pose = normalize_pose(ref_pose)

    joints = {
        "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        "left_shoulder": (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
        "right_shoulder": (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
        "left_knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "right_knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    }

    total_score = 0
    problems = []

    for joint_name, (p1, p2, p3) in joints.items():
        user_angle = angle_between(user_pose[p1, :2], user_pose[p2, :2], user_pose[p3, :2])
        ref_angle = angle_between(ref_pose[p1, :2], ref_pose[p2, :2], ref_pose[p3, :2])

        angle_diff = abs(user_angle - ref_angle)
        total_score += max(0.0, 1 - angle_diff / SCORE_RANGE)

        if angle_diff > tolerance:
            side, part = joint_name.split("_")
            if part == "elbow":
                msg = f"{'Straighten' if user_angle < ref_angle else 'Bend'} your {side} elbow"
            elif part == "shoulder":
                msg = f"{'Raise' if user_angle < ref_angle else 'Lower'} your {side} arm"
            else:
                msg = f"{'Straighten' if user_angle < ref_angle else 'Bend'} your {side} leg"
            problems.append((angle_diff, msg))

    # Sabse badi galti pehle, zyada se zyada MAX_TIPS
    problems.sort(reverse=True)
    feedback = [msg for _, msg in problems[:MAX_TIPS]]
    if not feedback:
        feedback = ["Great job! Pose matches well"]

    final_score = round((total_score / len(joints)) * 100)
    return final_score, feedback


if __name__ == "__main__":
    ref_poses = joblib.load("reference_poses.pkl")
    print("Feedback system ready!")
    print(f"Reference steps: {list(ref_poses.keys())}")