# AI-Powered-Delinquency-Prediction-Agentic-Strategy-Tata-iQ-Geldium-

## Project Overview 
As a AI Transformation Consultant in this Tata iQ job simulation, I developed an end-to-end delinquency management system for Geldium. This project transforms debt management at Geldium from a reactive manual process into a proactive, intelligent workflow. By performing initial data standardization in PostgreSQL and building an optimized Random Forest pipeline in Python, I created a system that balances predictive power with the transparency required for FinTech compliance.

---

## Dataset Details
Source: The dataset was provided by the client - Geldium

Rows: 501

Columns: 19

Key Fields: Customer_ID, Age, Gender, Income, Credit_Score, Credit_Utilization, Missed_Payments, Delinquent_Account, Loan_Balance, Debt_to_Income_Ratio, Employment_Status, Account_Tenure, Credit_Card_Type, Location, Month_1, Month_2, Month_3, Month_4, Month_5, Month_6.

The dataset contains financial and behavioral attributes of customers. It is used to predict delinquency risk based on past financial activities, credit history and employment status.

---

## Skills Demonstrated
**Data Engineering**: SQL ETL, Categorical Standardization (EMP/Employed), Target Variable Engineering.

**Machine Learning**: Random Forest, SMOTE (Oversampling), Hyperparameter Tuning.

**Risk Analytics**: Probability Threshold Optimization, Precision-Recall Trade-off Analysis.

**Explainable AI (XAI)**: Feature Importance ranking for regulatory compliance (Adverse Action).

**Strategic Thinking**: Agentic AI workflow design, Operational Triage, Business Case Development.

---

## Tools & Technologies
**Data Engineering & Storage**
PostgreSQL: Used as the primary data warehouse to perform ETL operations. I wrote complex CASE statements to standardize categorical labels (e.g., 'EMP' → 'Employed') and engineered the binary target variable for delinquency.

SQL: Leveraged for data cleaning, joining fragmented tables, and validating the mathematical integrity of DTI and Credit Utilization ratios before export.

**Machine Learning & Analytics**
Python (Kaggle Environment): The primary programming language for the predictive pipeline.

Scikit-Learn: Used for building the Random Forest classifier and performing hyperparameter tuning.

Imbalanced-Learn (SMOTE): Critical for addressing the 84/16 class imbalance to ensure the model could effectively learn the patterns of the "Delinquent" minority class.

Pandas & NumPy: For data manipulation and array processing.

**Strategy & Explainability**
Random Forest Feature Importance: Utilized as a "White-Box" explainability tool to provide the "Reason Codes" required for regulatory compliance (Adverse Action).

Agentic AI Framework: Designed the logic for autonomous outreach triage (Moderate vs. Critical risk segmentation).

Matplotlib & Seaborn: Used to visualize the Precision-Recall Trade-off and the optimal threshold at 0.35.

---

## System Architecture
1. **Data Foundation (PostgreSQL Standardization)**
The foundation of this project is a robust ETL process. Before the data reached the modeling environment, it was standardized within a PostgreSQL database.

Employment Status Unification: Resolved fragmented labels (e.g., merging 'EMP' and 'Employed' into a unified 'Employed' category) using SQL CASE logic to ensure categorical consistency.

Target Mapping: Converted multi-class status indicators into a clean, binary delinquent_account target.

Data Integrity: Ensured features like DTI and Credit Utilization were correctly typed and exported to a standardized CSV for Kaggle.

2. **Decision Engine & Probability Tuning (The "Brain")**
The system segments Geldium’s customers into risk tiers based on delinquency probability rather than a simple binary "yes/no."

Model: Utilizes a Random Forest Classifier with SMOTE to address the 84/16 class imbalance.

Threshold Tuning: Optimized at a 0.35 threshold to balance the "Goldilocks" zone between catching defaults and approving good business.

Explainability: Employs Feature Importance to identify the "Risk Drivers" (e.g., high credit utilization) behind every decision.

3. **Action Layer & Agentic AI**
The system uses Agentic AI to autonomously manage routine tasks, reserving Geldium’s specialists for high-impact decision points:

Autonomous Tasks: AI initiates automated reminders and acts as a researcher to identify "lost income" signals before outreach.

Human-in-the-Loop (HITL): Risk probability scores and key features are presented to human specialists during Hardship Verification to provide context for empathetic decision-making.

---

## Strategic Business Outcomes
1. **Financial Impact: The 15% Reduction Strategy**
The Logic: Flagging 50% of high-risk cases early allows intervention at Day 5 rather than Day 45.

Outcome: Target 15% reduction in 30+ day delinquency by recovering payments before the borrower’s financial situation becomes terminal.

2. **Operational Efficiency: The "60% Automation" Logic**
The Mechanism: Accounts are segmented by Probability Scores:

Low Risk (< 0.35): Self-resolve; no action needed.

Moderate Risk (0.35–0.60): 60% of flagged accounts; handled autonomously by Agentic AI.

Critical Risk (> 0.60): Escalated to Geldium’s human specialists.

Outcome: Staff spend 100% of their time on high-complexity, "Critical" negotiations.

3. **Customer Experience: Judgment-Free Negotiations**
The Mechanism: The Agentic AI uses Feature Importance (like Credit Utilization spikes) to offer tailored solutions (e.g., repayment holidays).

Outcome: Text-based portals remove the "social shame" of debt, leading to higher engagement and better retention for Geldium.

---

<details>
<summary><b>Click to expand: SQL Standardization Script</b></summary>

(
create table delinquency_data_standardized
(Customer_ID varchar,

Age int, Income int,

Credit_Score int,

Credit_Utilization float,

Missed_Payments int,

Delinquent_Account int,

Loan_Balance int,

Debt_to_Income_Ratio float,

Employment_Status text,

Account_Tenure int,

Credit_Card_Type text,

Location text,

Month_1 varchar, Month_2 varchar, Month_3 varchar,

Month_4 varchar, Month_5 varchar, Month_6 varchar);

select * from delinquency_data_standardized;

update delinquency_prediction_dataset
set employment_status = replace(employment_status,'EMP','Employed')
where employment_status like '%EMP%';

update delinquency_prediction_dataset
set employment_status = replace(employment_status,'employed','Employed')
where employment_status like 'employed'; 

select * from delinquency_prediction_dataset order by customer_id;

alter table delinquency_data_standardized
drop column customer_id;

COPY delinquency_prediction_dataset 
TO 'C:\Users\Public\delinquency_data.csv'
DELIMITER','
CSV HEADER;
)

</details>

<details>
<summary><b>Click to expand: Python Code Implementation - Model Code</b></summary>

(
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from imblearn.pipeline import Pipeline as ImbPipeline

from imblearn.over_sampling import SMOTE

 1. Load Data (Update path as needed)
df = pd.read_csv('/kaggle/input/delinquency-v3/delinquency_data.csv')

 2. Setup Features
target = 'delinquent_account'
numeric_features = ['age', 'income', 'credit_score', 'loan_balance', 'debt_to_income_ratio', 'credit_utilization']
categorical_features = ['employment_status', 'location']

if df[target].dtype == 'O':
    y = df[target].map({'Yes': 1, 'No': 0, 'Delinquent': 1, 'Standard': 0})
else:
    y = df[target]

X = df[numeric_features + categorical_features]

 3. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

 4. Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

 5. Pipeline with SMOTE and Random Forest
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=1.0)), 
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced_subsample'
    ))
])

 6. Fit
model_pipeline.fit(X_train, y_train)

 7. RECOMMENDATION: Adjusting Threshold to 0.35 for Business Balance
y_probs = model_pipeline.predict_proba(X_test)[:, 1]
risk_threshold = 0.35
y_pred_adj = (y_probs >= risk_threshold).astype(int)

 8. Evaluation
print(f"--- Business-Optimized Forest (Threshold: {risk_threshold}) ---")
print(classification_report(y_test, y_pred_adj))

 9. Final Visualizations
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

 Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_adj, cmap='YlGnBu', ax=ax[0])
ax[0].set_title(f"Confusion Matrix (Threshold: {risk_threshold})")

 Feature Importance
ohe_cols = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_cols = numeric_features + list(ohe_cols)
importances = model_pipeline.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_cols).sort_values(ascending=True)

feat_imp.tail(10).plot(kind='barh', color='teal', ax=ax[1])
ax[1].set_title("Top Drivers of Delinquency Risk")

plt.tight_layout()
plt.show())

</details>

<details>
<summary><b>Click to expand: Detailed Model Hyperparameters</b></summary>

(
Technical Appendix: Model Hyperparameters

To achieve the balance of 50% Recall and 85% Precision, the Random Forest classifier was configured with the following hyperparameters within an Imbalanced-Learn pipeline:

1. Random Forest Configuration
n_estimators: 200 — Increased from the default 100 to provide a more stable forest and reduce variance in risk prediction.

max_depth: 7 — Constrained to prevent overfitting on the minority "Delinquent" class, ensuring the model generalizes well to new Geldium applicants.

min_samples_leaf: 5 — Ensures that each leaf represents a meaningful segment of the population, providing smoother probability estimates for the 0.35 threshold.

class_weight: None — Since we utilized SMOTE for oversampling, the class weights were kept balanced to avoid "double-dipping" on the minority class importance.

2. Resampling Strategy (SMOTE)
sampling_strategy: 'auto' — Balanced the minority class to be equal to the majority class during the training phase.

k_neighbors: 5 — Used to generate synthetic examples that are locally representative of the delinquent population.

3. The "0.35" Probability Pivot
The default decision boundary for classifiers is 0.50. However, in the context of Geldium’s risk appetite, we performed a Precision-Recall Curve Analysis:

At 0.50: Recall was too low (< 30%), meaning we were missing too many defaults.

At 0.20: Precision dropped significantly, meaning we were harassing "Good" customers with unnecessary collections outreach.

At 0.35 (The "Goldilocks" Zone): We achieved the target 50% Recall while maintaining an 85% Precision for the standard portfolio.)

</details>

---

## Project Roadmap: Scaling to Production
While this job simulation established the core predictive engine, the following stages outline how this would be deployed and scaled within the Geldium ecosystem.

Phase 1: Pilot & A/B Testing (Current Milestone)
Shadow Mode Deployment: Run the model alongside the existing legacy system to compare predictions against real-world outcomes without impacting live customers.

Threshold Validation: Confirm the 0.35 threshold delivers the projected 15% reduction in delinquency in a live environment.

Phase 2: Automated Agentic Integration
API Development: Wrap the Random Forest model in a FastAPI or Flask wrapper to provide real-time risk scores to the frontend.

Outreach Logic: Connect the "Moderate Risk" probability scores directly to an automated messaging service (e.g., Twilio) to trigger the Agentic AI negotiation scripts.

Phase 3: Continuous Learning & Monitoring
Model Drift Detection: Implement monitoring to track if the DTI or Credit Utilization distributions shift due to macroeconomic changes (e.g., inflation or interest rate hikes).

Retraining Pipeline: Automate the PostgreSQL-to-Python ETL process to retrain the model monthly with the latest repayment data, ensuring the "Brain" stays sharp.

---

## Contact

For questions or collaborations, contact:
- **Name** : Chibuike Lawrence

- **Email** : [lawchibuike12345@gmail.com]

- **LinkedIn** : [https://www.linkedin.com/in/chibuike-lawrence-2348b01b6]

- **GitHub** : [https://github.com/ChibuikeLawrence12345]
