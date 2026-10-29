from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from feedback import score_and_feedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and references
model = joblib.load("garba_model.pkl")
reference_poses = joblib.load("reference_poses.pkl")

print("✓ Model loaded")
print(f"✓ Steps available: {list(reference_poses.keys())}")

@app.get("/")
def read_root():
    return {"message": "Garba Pose Detection Server Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        landmarks = np.array(data.get("landmarks", [])).reshape(33, 4)
        
        # Extract x, y, z
        pose = landmarks[:, :3].flatten()
        
        # Predict step
        step = model.predict([landmarks.flatten()])[0]
        
        # Get score and feedback
        score, feedback = score_and_feedback(pose, reference_poses[step])
        
        return {
            "step": step,
            "score": score,
            "feedback": feedback,
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)