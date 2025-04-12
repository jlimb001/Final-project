/_ README.md _/

# Hypertension Prediction Web Application

This Flask application predicts hypertension risk based on physical measurements using machine learning models trained on NHANES 2017-2020 data.

## Project Structure

```
hypertension-predictor/
├── app.py                 # Main Flask application
├── train_model.py         # Script to preprocess data and train the model
├── static/
│   ├── css/
│   │   └── style.css      # Custom CSS
│   └── feature_importance.png  # Generated from train_model.py
├── templates/
│   ├── index.html         # Home page with input form
│   ├── result.html        # Results page
│   ├── about.html         # Information about the application
│   └── error.html         # Error page
├── model/
│   ├── hypertension_model.pkl  # Trained model
│   └── scaler.pkl         # Feature scaler
└── requirements.txt       # Python dependencies
```

## Setup and Installation

1. Clone this repository:

```
git clone <repository-url>
cd hypertension-predictor
```

2. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Prepare your NHANES data:

   - Download the NHANES 2017-2020 data
   - Update the `data_path` in `train_model.py`

5. Train the model:

```
python train_model.py
```

6. Run the Flask application:

```
python app.py
```

7. Open your browser and go to `http://127.0.0.1:5000/`

## Requirements

Create a `requirements.txt` file with the following content:

```
Flask==2.3.3
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```
