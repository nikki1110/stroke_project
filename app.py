from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#Load models and columns
try:
    lr = joblib.load("model_lr.pkl")
    dt = joblib.load("model_dt.pkl")
    rf = joblib.load("model_rf.pkl")
    ensemble = joblib.load("model_ensemble.pkl")
    scaler = joblib.load("scaler.pkl") 
    model_columns = joblib.load("model_columns.pkl")
    models = {"Ensemble": ensemble}
    
except Exception as e:
    print("Model loading error:", e)
    lr = dt = rf = ensemble = None
    scaler = None
    models = {}
    model_columns = []

def map_risk(prob):
    if prob < 0.05:
        return f"✅ Very Low Risk (Probability: {prob:.2f})"
    elif prob < 0.15:
        return f"✅ Low Risk (Probability: {prob:.2f})"
    elif prob < 0.30:
        return f"⚠️ Medium Risk (Probability: {prob:.2f})"
    elif prob < 0.50:
        return f"⚠️ High Risk (Probability: {prob:.2f})"
    else:
        return f"🚨 Very High Risk (Probability: {prob:.2f})"
# ------------------ Format Feature Names ------------------
def format_feature(feature):
    if "_" in feature:
        parts = feature.split("_")

        if parts[0] == "gender":
            return f"Gender: {parts[1]}"
        elif parts[0] == "work":
            return f"Work Type: {parts[-1]}"
        elif parts[0] == "Residence":
            return f"Residence: {parts[-1]}"
        elif parts[0] == "smoking":
            return f"Smoking Status: {' '.join(parts[2:])}"

    mapping = {
        "age": "Age",
        "bmi": "BMI",
        "avg_glucose_level": "Glucose Level",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease"
    }

    return mapping.get(feature, feature)

# ------------------ Recommendations ------------------
def get_recommendations(risk_text):
    if "Very Low" in risk_text or "Low" in risk_text:
        return [
            "Maintain a balanced diet",
            "Exercise regularly",
            "Continue healthy lifestyle habits"
        ]

    elif "Medium" in risk_text:
        return [
            "Monitor blood pressure regularly",
            "Reduce sugar and salt intake",
            "Increase physical activity"
        ]

    elif "High" in risk_text or "Very High" in risk_text:
        return [
            "Consult a doctor immediately",
            "Control hypertension and glucose levels",
            "Avoid smoking and alcohol",
            "Maintain strict diet and exercise routine"
        ]

    return []

# Prediction function
def make_prediction(data):
    try:
        input_dict = {
            'age': float(data.get('age', 0)),
            'hypertension': 1 if str(data.get('hypertension','No')).lower() in ['yes','1','true'] else 0,
            'heart_disease': 1 if str(data.get('heart_disease','No')).lower() in ['yes','1','true'] else 0,
            'avg_glucose_level': float(data.get('glucose', 0)),
            'bmi': float(data.get('bmi', 0)),
            'gender': data.get('gender','Other'),
            'ever_married': data.get('ever_married','No'),
            'work_type': data.get('work_type','Private'),
            'Residence_type': data.get('residence_type','Urban'),
            'smoking_status': data.get('smoking_status','never smoked')
        }

        df = pd.DataFrame([input_dict])
        df = pd.get_dummies(df)

        if model_columns:
            df = df.reindex(columns=model_columns, fill_value=0)

<<<<<<< HEAD
        # Apply scaling
        df_scaled = scaler.transform(df) if scaler is not None else df

        # Use Ensemble Model
        model = models.get("Ensemble")
        if not model:
            return "❌ Model not loaded.", [], []

        prob = model.predict_proba(df_scaled)[0][1]
        print("Probability:", prob)
        risk_text = map_risk(prob)

        # ------------------ Explainable AI ------------------
        top_features = []
        if rf is not None:
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-3:][::-1]
            top_features = [model_columns[i] for i in indices]

        # Convert to readable names
        top_features = [format_feature(f) for f in top_features]

        # ------------------ Recommendations ------------------
        recommendations = get_recommendations(risk_text)

        return risk_text, top_features, recommendations

    except Exception as e:
        return f"❌ Error: {e}", [], []
# ------------------ Routes ------------------
@app.route('/')
=======
        model_choice = data.get('model','Random Forest')
        model = models.get(model_choice, models.get('Random Forest'))

        if not model:
            return "❌ Model not loaded.", []

        prob = model.predict_proba(df)[0][1]
        risk_text = map_risk(prob)

        # Using Explainable AI 
        top_features = []

        if model_choice in ["Random Forest", "Decision Tree"]:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-3:][::-1]
            top_features = [model_columns[i] for i in indices]

        elif model_choice == "Logistic Regression":
            coefs = model.coef_[0]
            indices = np.argsort(np.abs(coefs))[-3:][::-1]
            top_features = [model_columns[i] for i in indices]

        return risk_text, top_features

    except Exception as e:
        return f"❌ Error: {e}", []


# Routes 
app.route('/')
>>>>>>> f9cc5cb7663bfea0b75bfad4065c862e1839bfa1
def landing():
    return render_template('landing.html')

@app.route('/user_type')
def user_type():
    return render_template('user_type.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Patient Prediction 
@app.route('/predict', methods=['GET','POST'])
@app.route('/predict_patient', methods=['GET','POST'])
def predict_patient():
    result = None
    form_data = {}
    top_features = []

# Doctor Prediction 
@app.route('/predict_doctor', methods=['GET','POST'])
def predict_doctor():
    result = None
    tables = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            try:
                df_csv = pd.read_csv(file)

                if df_csv.empty:
                    result = "❌ CSV is empty."
                else:
                    df_processed = pd.get_dummies(df_csv)

                    if model_columns:
                        df_processed = df_processed.reindex(columns=model_columns, fill_value=0)

                    tables = []

                    for idx in range(len(df_csv)):
                        row_result = {}

                        try:
                            # Select a single row and ensure correct feature alignment
                            row_df = df_processed.iloc[[idx]]
                            row_df = row_df.reindex(columns=model_columns, fill_value=0)

                            # Apply scaling if the scaler is available
                            row_scaled = scaler.transform(row_df) if scaler is not None else row_df

                            # Predict using the ensemble model
                            prob = ensemble.predict_proba(row_scaled)[0][1]
                            row_result['Ensemble_Prediction'] = map_risk(prob)

                        except Exception:
                            row_result['Ensemble_Prediction'] = "❌ Could not predict"

                        # Combine original data with prediction
                        tables.append({**df_csv.iloc[idx].to_dict(), **row_result})

            except Exception as e:
                result = f"❌ CSV Error: {e}"

    return render_template('index_doctor.html', result=result, results=tables)
if __name__ == '__main__':
    app.run(debug=True)