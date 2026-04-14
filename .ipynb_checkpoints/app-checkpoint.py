from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ------------------ Load models and columns ------------------
try:
    lr = joblib.load("model_lr.pkl")
    dt = joblib.load("model_dt.pkl")
    rf = joblib.load("model_rf.pkl")
    ensemble = joblib.load("model_ensemble.pkl")
    scaler = joblib.load("scaler.pkl") 
    model_columns = joblib.load("model_columns.pkl")
    models = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}
except Exception as e:
    print("Model loading error:", e)
    lr = dt = rf = None
    scaler = None
    models = {}
    model_columns = []

# ------------------ Map probability to risk ------------------
def map_risk(prob):
    if prob < 0.03: return f"✅ Very Low Risk (Probability: {prob:.2f})"
    elif prob < 0.07: return f"✅ Low Risk (Probability: {prob:.2f})"
    elif prob < 0.12: return f"⚠️ Medium Risk (Probability: {prob:.2f})"
    elif prob < 0.16: return f"⚠️ High Risk (Probability: {prob:.2f})"
    else: return f"🚨 Very High Risk (Probability: {prob:.2f})"

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

# ------------------ Prediction function ------------------
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

        df_scaled = scaler.transform(df) if scaler is not None else df

        model_choice = data.get('model','Random Forest')
        model = models.get(model_choice, models.get('Random Forest'))

        if not model:
            return "❌ Model not loaded.", [], []

        prob = model.predict_proba(df_scaled)[0][1]
        print("Probability:", prob)
        risk_text = map_risk(prob)

        # ------------------ Explainable AI ------------------
        top_features = []

        if model_choice in ["Random Forest", "Decision Tree"]:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-3:][::-1]
            top_features = [model_columns[i] for i in indices]

        elif model_choice == "Logistic Regression":
            coefs = model.coef_[0]
            indices = np.argsort(np.abs(coefs))[-3:][::-1]
            top_features = [model_columns[i] for i in indices]

        # Convert to readable names
        top_features = [format_feature(f) for f in top_features]

        # Get recommendations
        recommendations = get_recommendations(risk_text)

        return risk_text, top_features, recommendations

    except Exception as e:
        return f"❌ Error: {e}", [], []

# ------------------ Routes ------------------
@app.route('/')
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

# ------------------ Patient Prediction ------------------
@app.route('/predict', methods=['GET','POST'])
@app.route('/predict_patient', methods=['GET','POST'])
def predict_patient():
    result = None
    form_data = {}
    top_features = []
    recommendations = []

    if request.method == 'POST':
        form_data = request.form.to_dict()
        result, top_features, recommendations = make_prediction(request.form)

    return render_template('index.html',
                           result=result,
                           form_data=form_data,
                           top_features=top_features,
                           recommendations=recommendations)

# ------------------ Doctor Prediction ------------------
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

                        for model_name, model in models.items():
                            try:
                                prob = model.predict_proba(df_processed.iloc[[idx]])[0][1]
                                row_result[model_name + '_Prediction'] = map_risk(prob)
                            except:
                                row_result[model_name + '_Prediction'] = "❌ Could not predict"

                        tables.append({**df_csv.iloc[idx].to_dict(), **row_result})

            except Exception as e:
                result = f"❌ CSV Error: {e}"

    return render_template('index_doctor.html', result=result, results=tables)

if __name__ == '__main__':
    app.run(debug=True)