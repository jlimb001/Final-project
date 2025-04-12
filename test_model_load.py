import pickle
import os

model_path = 'model/best_svm_model.pkl'

print(f"Working directory: {os.getcwd()}")
print(f"Looking for model at: {model_path}")

if os.path.exists(model_path):
    print("Model file found.")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
else:
    print("❌ Model file does not exist.")
