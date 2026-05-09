import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve, auc,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score)

import shap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory for CSVs
import os
os.makedirs('results_csv', exist_ok=True)


# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("="*70)
print("STEP 1: LOADING DATASET")
print("="*70)

data_path = "./data/default of credit card clients.xls"
df = pd.read_excel(data_path, header=1)

df.rename(columns={'default payment next month': 'default'}, inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean()*100:.2f}%")


# ============================================================================
# 2. FIGURE 1 DATA: Class Imbalance Statistics
# ============================================================================
print("\n" + "="*70)
print("FIGURE 1 DATA: Class Imbalance")
print("="*70)

class_counts = df['default'].value_counts().reset_index()
class_counts.columns = ['Default_Status', 'Count']
class_counts['Percentage'] = (class_counts['Count'] / class_counts['Count'].sum() * 100).round(2)
class_counts['Default_Status'] = class_counts['Default_Status'].map({0: 'No Default', 1: 'Default'})

print(class_counts)
class_counts.to_csv('results_csv/figure1_class_distribution.csv', index=False)
print("✓ Saved: results_csv/figure1_class_distribution.csv")


# ============================================================================
# 3. FIGURE 2 DATA: Correlation Matrix
# ============================================================================
print("\n" + "="*70)
print("FIGURE 2 DATA: Correlation with Default")
print("="*70)

corr_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

corr_with_default = df[corr_features + ['default']].corr()['default'].drop('default').sort_values(ascending=False)
corr_df = corr_with_default.reset_index()
corr_df.columns = ['Feature', 'Correlation_with_Default']
print(corr_df)
corr_df.to_csv('results_csv/figure2_correlation_with_default.csv', index=False)
print("✓ Saved: results_csv/figure2_correlation_with_default.csv")


# ============================================================================
# 4. FIGURE 3 DATA: Demographic Analysis
# ============================================================================
print("\n" + "="*70)
print("FIGURE 3 DATA: Demographic Analysis")
print("="*70)

# Gender
gender_stats = df.groupby('SEX')['default'].agg(['count', 'mean']).reset_index()
gender_stats.columns = ['Gender_Code', 'Count', 'Default_Rate']
gender_stats['Default_Rate_Pct'] = (gender_stats['Default_Rate'] * 100).round(1)
gender_stats['Gender'] = gender_stats['Gender_Code'].map({1: 'Male', 2: 'Female'})
print("\n--- Gender Analysis ---")
print(gender_stats[['Gender', 'Count', 'Default_Rate_Pct']])

# Education
edu_mapping = {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others', 
               5: 'Unknown', 6: 'Unknown', 0: 'Unknown'}
edu_stats = df.groupby('EDUCATION')['default'].agg(['count', 'mean']).reset_index()
edu_stats.columns = ['Education_Code', 'Count', 'Default_Rate']
edu_stats['Education'] = edu_stats['Education_Code'].map(edu_mapping)
edu_stats['Default_Rate_Pct'] = (edu_stats['Default_Rate'] * 100).round(1)
print("\n--- Education Analysis ---")
print(edu_stats[['Education', 'Count', 'Default_Rate_Pct']])

# Age Groups
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 60, 100], 
                          labels=['20-29', '30-39', '40-49', '50-59', '60+'])
age_stats = df.groupby('AGE_GROUP', observed=True)['default'].agg(['count', 'mean']).reset_index()
age_stats.columns = ['Age_Group', 'Count', 'Default_Rate']
age_stats['Default_Rate_Pct'] = (age_stats['Default_Rate'] * 100).round(1)
print("\n--- Age Group Analysis ---")
print(age_stats)

# Credit Limit Quantiles
df['LIMIT_BAL_GROUP'] = pd.qcut(df['LIMIT_BAL'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
limit_stats = df.groupby('LIMIT_BAL_GROUP', observed=True)['default'].agg(['count', 'mean']).reset_index()
limit_stats.columns = ['Credit_Limit_Group', 'Count', 'Default_Rate']
limit_stats['Default_Rate_Pct'] = (limit_stats['Default_Rate'] * 100).round(1)
print("\n--- Credit Limit Analysis ---")
print(limit_stats)

# Combine all demographic results
demographic_results = {
    'gender': gender_stats[['Gender', 'Count', 'Default_Rate_Pct']],
    'education': edu_stats[['Education', 'Count', 'Default_Rate_Pct']],
    'age': age_stats,
    'credit_limit': limit_stats
}

# Save combined
with pd.ExcelWriter('results_csv/figure3_demographic_analysis.xlsx') as writer:
    gender_stats[['Gender', 'Count', 'Default_Rate_Pct']].to_excel(writer, sheet_name='Gender', index=False)
    edu_stats[['Education', 'Count', 'Default_Rate_Pct']].to_excel(writer, sheet_name='Education', index=False)
    age_stats.to_excel(writer, sheet_name='Age_Group', index=False)
    limit_stats.to_excel(writer, sheet_name='Credit_Limit', index=False)
print("✓ Saved: results_csv/figure3_demographic_analysis.xlsx")


# ============================================================================
# 5. FIGURE 4 DATA: Payment History Impact
# ============================================================================
print("\n" + "="*70)
print("FIGURE 4 DATA: Payment History Impact")
print("="*70)

# PAY_0 analysis
pay_status_names = {
    -2: 'No Consumption', -1: 'Pay Duly', 0: 'Minimum Payment',
    1: 'Delay 1 Month', 2: 'Delay 2 Months', 3: 'Delay 3 Months',
    4: 'Delay 4 Months', 5: 'Delay 5 Months', 6: 'Delay 6 Months',
    7: 'Delay 7 Months', 8: 'Delay 8+ Months'
}

pay0_stats = df.groupby('PAY_0')['default'].agg(['count', 'mean']).reset_index()
pay0_stats.columns = ['PAY_0_Code', 'Count', 'Default_Rate']
pay0_stats['Payment_Status'] = pay0_stats['PAY_0_Code'].map(pay_status_names)
pay0_stats['Default_Rate_Pct'] = (pay0_stats['Default_Rate'] * 100).round(1)
pay0_stats = pay0_stats.sort_values('PAY_0_Code')
print("\n--- Default Rate by Payment Status (PAY_0) ---")
print(pay0_stats[['Payment_Status', 'Count', 'Default_Rate_Pct']])

# Payment trend over months
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
pay_months = ['Sep 05', 'Aug 05', 'Jul 05', 'Jun 05', 'May 05', 'Apr 05']

trend_defaulters = df[df['default'] == 1][pay_cols].mean()
trend_non_defaulters = df[df['default'] == 0][pay_cols].mean()

payment_trend = pd.DataFrame({
    'Month': pay_months,
    'Defaulters_Avg_Status': trend_defaulters.values.round(2),
    'Non_Defaulters_Avg_Status': trend_non_defaulters.values.round(2),
    'Difference': (trend_defaulters.values - trend_non_defaulters.values).round(2)
})
print("\n--- Payment Status Trend ---")
print(payment_trend)

payment_trend.to_csv('results_csv/figure4_payment_trend.csv', index=False)
pay0_stats.to_csv('results_csv/figure4_pay0_default_rates.csv', index=False)
print("✓ Saved: results_csv/figure4_pay0_default_rates.csv")
print("✓ Saved: results_csv/figure4_payment_trend.csv")


# ============================================================================
# 6. DATA PREPARATION FOR MODELING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: PREPARING DATA FOR MODELING")
print("="*70)

X = df.drop(['default', 'AGE_GROUP', 'LIMIT_BAL_GROUP'], axis=1, errors='ignore')
y = df['default']

if 'ID' in X.columns:
    X = X.drop('ID', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} samples (default rate: {y_train.mean()*100:.2f}%)")
print(f"Test set: {X_test.shape[0]} samples (default rate: {y_test.mean()*100:.2f}%)")

# Save split info
split_info = pd.DataFrame({
    'Dataset': ['Training', 'Test'],
    'Samples': [X_train.shape[0], X_test.shape[0]],
    'Default_Rate_Pct': [y_train.mean()*100, y_test.mean()*100]
})
split_info.to_csv('results_csv/train_test_split.csv', index=False)


# ============================================================================
# 7. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3: TRAINING MODELS")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Training set: {X_train_balanced.shape[0]} samples")

results = {}
roc_data = {}
pr_data = {}
confusion_data = {}

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_data[name] = {
        'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1]
    }
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': results[name]['roc_auc']}
    
    # PR curve data
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    pr_data[name] = {'recall': recall_vals, 'precision': precision_vals, 'pr_auc': pr_auc}
    results[name]['pr_auc'] = pr_auc
    
    print(f"\n{name}:")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1-Score:  {results[name]['f1']:.4f}")
    print(f"  MCC:       {results[name]['mcc']:.4f}")
    print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}")
    print(f"  PR-AUC:    {results[name]['pr_auc']:.4f}")

# Save model results
model_results_df = pd.DataFrame(results).T.reset_index()
model_results_df.columns = ['Model', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC_AUC', 'PR_AUC']
model_results_df.to_csv('results_csv/model_results_summary.csv', index=False)
model_results_df.to_csv('results_csv/figure5_model_comparison.csv', index=False)
print("\n✓ Saved: results_csv/model_results_summary.csv")
print("✓ Saved: results_csv/figure5_model_comparison.csv")

# Save confusion matrices
confusion_df = pd.DataFrame(confusion_data).T.reset_index()
confusion_df.columns = ['Model', 'True_Negatives', 'False_Positives', 'False_Negatives', 'True_Positives']
confusion_df.to_csv('results_csv/figure6_confusion_matrices.csv', index=False)
print("✓ Saved: results_csv/figure6_confusion_matrices.csv")


# ============================================================================
# 8. FIGURE 7 & 8 DATA: ROC and PR Curves
# ============================================================================
print("\n" + "="*70)
print("FIGURES 7 & 8 DATA: ROC and PR Curves")
print("="*70)

# Save ROC curve points for each model
for name, data in roc_data.items():
    roc_points = pd.DataFrame({
        'False_Positive_Rate': data['fpr'],
        'True_Positive_Rate': data['tpr']
    })
    roc_points.to_csv(f'results_csv/figure7_roc_curve_{name.replace(" ", "_")}.csv', index=False)
    print(f"  Saved: figure7_roc_curve_{name.replace(' ', '_')}.csv")

# Save PR curve points for each model
for name, data in pr_data.items():
    pr_points = pd.DataFrame({
        'Recall': data['recall'],
        'Precision': data['precision']
    })
    pr_points.to_csv(f'results_csv/figure8_pr_curve_{name.replace(" ", "_")}.csv', index=False)
    print(f"  Saved: figure8_pr_curve_{name.replace(' ', '_')}.csv")


# ============================================================================
# 9. SHAP ANALYSIS DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SHAP ANALYSIS DATA")
print("="*70)

best_model = results['XGBoost']  # Using XGBoost for SHAP
best_model_obj = models['XGBoost']
best_model_obj.fit(X_train_balanced, y_train_balanced)

X_sample = X_test.sample(n=100, random_state=42)
explainer = shap.TreeExplainer(best_model_obj)
shap_values = explainer.shap_values(X_sample)

# FIGURE 9 & 10 DATA: SHAP values
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\n--- SHAP Feature Importance (Top 15) ---")
print(shap_importance.head(15))
shap_importance.to_csv('results_csv/figure10_shap_feature_importance.csv', index=False)
print("✓ Saved: results_csv/figure10_shap_feature_importance.csv")

# FIGURE 11 DATA: Single prediction explanation
defaulter_idx = y_test[y_test == 1].index[0]
defaulter_instance = X_test.loc[defaulter_idx]

shap_single = explainer.shap_values(defaulter_instance.values.reshape(1, -1))[0]

shap_single_df = pd.DataFrame({
    'Feature': X.columns,
    'Feature_Value': defaulter_instance.values,
    'SHAP_Value': shap_single
}).sort_values('SHAP_Value', ascending=False)

print("\n--- SHAP Explanation for a Defaulting Client ---")
print(shap_single_df.head(10))
shap_single_df.to_csv('results_csv/figure11_shap_waterfall_data.csv', index=False)
print("✓ Saved: results_csv/figure11_shap_waterfall_data.csv")

# Base value and prediction
base_value = explainer.expected_value
prediction_prob = best_model_obj.predict_proba(defaulter_instance.values.reshape(1, -1))[0, 1]
shap_explanation = pd.DataFrame({
    'Metric': ['Base_Value', 'Predicted_Probability', 'f(x)'],
    'Value': [base_value, prediction_prob, prediction_prob]
})
shap_explanation.to_csv('results_csv/figure11_base_and_prediction.csv', index=False)


# ============================================================================
# 10. FIGURE 12 DATA: Feature Importance Comparison
# ============================================================================
print("\n" + "="*70)
print("FIGURE 12 DATA: Feature Importance Comparison")
print("="*70)

# XGBoost built-in importance
builtin_importance = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost_Builtin_Importance': best_model_obj.feature_importances_
}).sort_values('XGBoost_Builtin_Importance', ascending=False)

# Merge with SHAP importance
importance_comparison = builtin_importance.merge(shap_importance, on='Feature')
importance_comparison = importance_comparison.sort_values('Mean_Abs_SHAP', ascending=False)

print("\n--- Top 10 Features by Both Metrics ---")
print(importance_comparison.head(10))
importance_comparison.to_csv('results_csv/figure12_feature_importance_comparison.csv', index=False)
print("✓ Saved: results_csv/figure12_feature_importance_comparison.csv")


# ============================================================================
# 11. ADDITIONAL: EDA Statistics for Paper Tables
# ============================================================================
print("\n" + "="*70)
print("ADDITIONAL STATISTICS FOR PAPER")
print("="*70)

# Dataset overview
dataset_overview = pd.DataFrame({
    'Metric': ['Total Samples', 'Default Rate (%)', 'Features Count', 
               'Training Samples', 'Test Samples', 'Default Rate in Train (%)', 
               'Default Rate in Test (%)'],
    'Value': [len(df), f"{df['default'].mean()*100:.2f}", X.shape[1],
              len(X_train), len(X_test), f"{y_train.mean()*100:.2f}", f"{y_test.mean()*100:.2f}"]
})
dataset_overview.to_csv('results_csv/dataset_overview.csv', index=False)
print("✓ Saved: results_csv/dataset_overview.csv")

# MARRIAGE analysis
marriage_mapping = {1: 'Married', 2: 'Single', 3: 'Other'}
marriage_stats = df.groupby('MARRIAGE')['default'].agg(['count', 'mean']).reset_index()
marriage_stats.columns = ['Marriage_Code', 'Count', 'Default_Rate']
marriage_stats['Marital_Status'] = marriage_stats['Marriage_Code'].map(marriage_mapping)
marriage_stats['Default_Rate_Pct'] = (marriage_stats['Default_Rate'] * 100).round(1)
print("\n--- Marital Status Analysis (Key SHAP Feature) ---")
print(marriage_stats[['Marital_Status', 'Count', 'Default_Rate_Pct']])
marriage_stats.to_csv('results_csv/marital_status_analysis.csv', index=False)

print("\n" + "="*70)
print("ALL CSV FILES SAVED TO './results_csv/' DIRECTORY")
print("="*70)

# List all generated files
print("\nGenerated CSV files:")
for f in sorted(os.listdir('results_csv')):
    print(f"  - {f}")