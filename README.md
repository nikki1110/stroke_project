𝐒𝐭𝐫𝐨𝐤𝐞 𝐑𝐢𝐬𝐤 𝐏𝐫𝐞𝐝𝐢𝐜𝐭𝐢𝐨𝐧 𝐒𝐲𝐬𝐭𝐞𝐦

This project is a web-based application that predicts the risk of stroke using machine learning algorithms. It is designed to help both PATIENTS and DOCTORS quickly assess stroke risk based on health parameters.

𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐎𝐯𝐞𝐫𝐯𝐢𝐞𝐰

Stroke is one of the leading causes of death, and early prediction can help in prevention.
This system uses multiple machine learning models and combines them to give a more accurate prediction.

The application provides:

1. Individual patient prediction
2. Doctor dashboard for batch prediction (CSV upload)
3. Risk level classification with recommendations

𝐓𝐞𝐜𝐡 𝐒𝐭𝐚𝐜𝐤

- Frontend: HTML, CSS
- Backend: Flask (Python)
- Machine Learning: Scikit-learn
- Libraries Used: Pandas, NumPy, Joblib

𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐌𝐨𝐝𝐞𝐥𝐬 𝐔𝐬𝐞𝐝

• Logistic Regression
• Decision Tree
• Random Forest

These models are combined using ensemble learning (voting) to improve accuracy.
𝐅𝐞𝐚𝐭𝐮𝐫𝐞𝐬

• Predict stroke risk based on user input
• Categorizes risk into:

  ‣ Very Low
  ‣ Low
  ‣ Medium
  ‣ High
  ‣ Very High
 Shows top factors influencing prediction and also provides health recommendations accordingly.
 Doctor dashboard can be used for multiple patient predictions using CSV files

𝐈𝐧𝐩𝐮𝐭 𝐏𝐚𝐫𝐚𝐦𝐞𝐭𝐞𝐫𝐬

● Age
● Gender
● Hypertension
● Heart Disease
● Average Glucose Level
● BMI
● Marital Status
● Work Type
● Residence Type
● Smoking Status

𝐇𝐨𝐰 𝐭𝐨 𝐑𝐮𝐧 𝐭𝐡𝐞 𝐏𝐫𝐨𝐣𝐞𝐜𝐭

1. Clone the repository:

   git clone https://github.com/your-username/stroke_project.git
   

2. Navigate to the project folder:

cd stroke_project
   

3. Install required libraries:

   pip install -r requirements.txt


4. Run the Flask app:

   python app.py

5. Open browser and go to:

   http://127.0.0.1:5000/
𝐏𝐫𝐨𝐣𝐞𝐜𝐭 𝐒𝐭𝐫𝐮𝐜𝐭𝐮𝐫𝐞

stroke_project/
│── app.py
│── model_lr.pkl
│── model_dt.pkl
│── model_rf.pkl
│── model_ensemble.pkl
│── scaler.pkl
│── model_columns.pkl
│── templates/
│── static/
│── project_stroke.ipynb

𝐅𝐮𝐭𝐮𝐫𝐞 𝐢𝐦𝐩𝐫𝐨𝐯𝐞𝐦𝐞𝐧𝐭𝐬

• Improve model accuracy with more data
• Add user authentication
• Deploy on cloud (Heroku / Render)
• Add real-time health tracking


## 👩‍💻 Author

Developed as a mini-project for academic purposes.

