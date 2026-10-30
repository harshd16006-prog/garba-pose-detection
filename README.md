# garba-pose-detection 
 
AI Garba pose detection with 96.21% accuracy 
 
Detects 3 Garba steps: taali, dodhiyu, hinch. Gives feedback and a score out of 100. 
 
## How it works 
1. Flutter app uses Google ML Kit to get 33 body landmarks from the camera. 
2. App sends landmarks to the FastAPI server at /predict. 
3. Random Forest model predicts the step; feedback.py compares with reference poses for feedback and score. 
 
## Model 
Random Forest, 100 trees, 80/20 stratified train/test split. Test accuracy: 96.21 percent. 
 
## Live server 
https://garba-pose-detection.onrender.com/docs 
Render free tier sleeps after 15 min idle; first request can take 30-60 seconds. 
 
## Requirements status 
1. ML Kit pose landmarks: done 
2. Dataset garba_dataset.csv: done 
3. Preprocessing, normalization in train.py: done 
4. Random Forest, scikit-learn: done 
5. Testing on unseen 20 percent data: done 
6. Model saved as garba_model.pkl: done 
7. Feedback system, feedback.py: done 
8. Pose matching score out of 100: done 
9. Flutter app integration: done 
10. GitHub and Render deployment: done 
 
## Run server locally 
python -m venv venv 
venv\Scripts\activate 
pip install -r requirements.txt 
uvicorn server:app --host 0.0.0.0 --port 8000 
 
## Build app 
cd flutter_app 
flutter pub get 
flutter build apk 
Set kServerUrl in lib/main.dart to your server URL.
