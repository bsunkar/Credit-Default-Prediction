import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, roc_curve, auc,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef, ConfusionMatrixDisplay)

import shap

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def save_figure(base_name, dpi=300, bbox_inches='tight'):
    png_path = f"{base_name}.png"
    eps_dir = 'figures_eps'
    os.makedirs(eps_dir, exist_ok=True)
    eps_path = os.path.join(eps_dir, f"{base_name}.eps")
    plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(eps_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {eps_path}")


# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("="*70)
print("STEP 1: LOADING DATASET")
print("="*70)

# Local dataset path in project data folder
data_path = "./data/default of credit card clients.xls"
df = pd.read_excel(data_path, header=1)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# Rename target column for clarity
df.rename(columns={'default payment next month': 'default'}, inplace=True)

print(f"\nDataset info:")
print(df.info())

print(f"\nTarget variable distribution (default=1 means defaulted):")
print(df['default'].value_counts())
print(f"\nDefault rate: {df['default'].mean()*100:.2f}%")


# ============================================================================
# 2. FIGURE 1: TARGET VARIABLE DISTRIBUTION (Class Imbalance)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors = ['#2ecc71', '#e74c3c']
explode = (0, 0.05)
df['default'].value_counts().plot(
    kind='pie', ax=axes[0], autopct='%1.1f%%', 
    colors=colors, explode=explode, shadow=True,
    labels=['No Default (0)', 'Default (1)'],
    textprops={'fontsize': 12}
)
axes[0].set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
axes[0].set_ylabel('')

# Bar plot
counts = df['default'].value_counts()
bars = axes[1].bar(['No Default\n(0)', 'Default\n(1)'], counts.values, 
                    color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_title('Class Distribution - Count', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Clients')
axes[1].set_xlabel('Default Status')

# Add count labels on bars
for bar, count in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'n={count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('Figure 1: Class Imbalance in Credit Card Default Dataset', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure('figure1_class_imbalance')
plt.show()


# ============================================================================
# 3. FIGURE 2: CORRELATION HEATMAP
# ============================================================================
# Select key features for correlation analysis
corr_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                 'default']

corr_matrix = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 9})
ax.set_title('Figure 2: Feature Correlation Matrix\n(Darker red = stronger positive correlation with default)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure('figure2_correlation_heatmap')
plt.show()


# ============================================================================
# 4. FIGURE 3: DEFAULT RATE BY DEMOGRAPHIC FACTORS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Default rate by SEX
sex_default = df.groupby('SEX')['default'].mean() * 100
sex_labels = ['Female', 'Male']
bars = axes[0, 0].bar(sex_labels, sex_default.values, color=['#e74c3c', '#3498db'])
axes[0, 0].set_title('Default Rate by Gender', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Default Rate (%)')
axes[0, 0].set_ylim(0, 30)
for bar, val in zip(bars, sex_default.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Default rate by EDUCATION
edu_default = df.groupby('EDUCATION')['default'].mean() * 100
edu_mapping = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Others', 
               5: 'Unknown', 6: 'Unknown 2', 0: 'Unknown 0'}
edu_labels = [edu_mapping.get(int(i), f'Unknown {i}') for i in edu_default.index]
bars = axes[0, 1].bar(range(len(edu_default)), edu_default.values, color='coral')
axes[0, 1].set_title('Default Rate by Education Level', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(len(edu_default)))
axes[0, 1].set_xticklabels(edu_labels, rotation=45, ha='right')
axes[0, 1].set_ylabel('Default Rate (%)')
for bar, val in zip(bars, edu_default.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10)

# Default rate by AGE groups
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 60, 100], 
                          labels=['20-29', '30-39', '40-49', '50-59', '60+'])
age_default = df.groupby('AGE_GROUP')['default'].mean() * 100
axes[1, 0].plot(range(len(age_default)), age_default.values, 'o-', 
                linewidth=2, markersize=10, color='#2c3e50')
axes[1, 0].fill_between(range(len(age_default)), age_default.values, alpha=0.3)
axes[1, 0].set_xticks(range(len(age_default)))
axes[1, 0].set_xticklabels(age_default.index)
axes[1, 0].set_title('Default Rate by Age Group', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Default Rate (%)')
axes[1, 0].set_ylim(0, 35)
for i, val in enumerate(age_default.values):
    axes[1, 0].annotate(f'{val:.1f}%', (i, val), textcoords="offset points", 
                        xytext=(0, 10), ha='center')

# Default rate by LIMIT_BAL (Credit Limit) quantiles
df['LIMIT_BAL_GROUP'] = pd.qcut(df['LIMIT_BAL'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
limit_default = df.groupby('LIMIT_BAL_GROUP')['default'].mean() * 100
bars = axes[1, 1].bar(range(len(limit_default)), limit_default.values, color='teal')
axes[1, 1].set_title('Default Rate by Credit Limit (Quantiles)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(len(limit_default)))
axes[1, 1].set_xticklabels(limit_default.index, rotation=45)
axes[1, 1].set_ylabel('Default Rate (%)')
for bar, val in zip(bars, limit_default.values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10)

plt.suptitle('Figure 3: Default Rate Analysis by Demographic Factors', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure('figure3_demographic_analysis')
plt.show()


# ============================================================================
# 5. FIGURE 4: PAYMENT HISTORY IMPACT
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Payment status values: -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8
pay_status = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
pay_labels = ['No Cons', 'Pay Duly', 'Min Pay', 'Delay 1M', 'Delay 2M', 
              'Delay 3M', 'Delay 4M', 'Delay 5M', 'Delay 6M', 'Delay 7M', 'Delay 8M+']

# Default rate by PAY_0 (most recent payment status)
pay0_default = df.groupby('PAY_0')['default'].mean() * 100
bars = axes[0].bar(range(len(pay_labels)), 
                   [pay0_default.get(status, 0) for status in pay_status], 
                   color='crimson', alpha=0.7)
axes[0].set_title('Default Rate by Most Recent Payment Status (PAY_0)', 
                  fontsize=12, fontweight='bold')
axes[0].set_xticks(range(len(pay_labels)))
axes[0].set_xticklabels(pay_labels, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('Default Rate (%)')
axes[0].axhline(y=df['default'].mean()*100, color='blue', linestyle='--', 
                label=f"Overall Avg: {df['default'].mean()*100:.1f}%")
axes[0].legend()

# Payment history trend (PAY_0 to PAY_6)
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
pay_months = ['Sep 05', 'Aug 05', 'Jul 05', 'Jun 05', 'May 05', 'Apr 05']

# Calculate average payment status for defaulters vs non-defaulters
defaulters = df[df['default'] == 1][pay_cols].mean()
non_defaulters = df[df['default'] == 0][pay_cols].mean()

x = range(len(pay_months))
axes[1].plot(x, defaulters.values, 'o-', linewidth=2, markersize=8, 
             label='Defaulters', color='red')
axes[1].plot(x, non_defaulters.values, 's--', linewidth=2, markersize=8, 
             label='Non-Defaulters', color='green')
axes[1].set_xticks(x)
axes[1].set_xticklabels(pay_months)
axes[1].set_title('Payment Status Trend (Lower = Better Payment Behavior)', 
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Payment Status Score')
axes[1].set_xlabel('Month')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)

plt.suptitle('Figure 4: Impact of Payment History on Default Risk', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure('figure4_payment_history_impact')
plt.show()


# ============================================================================
# 6. DATA PREPARATION FOR MODELING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: PREPARING DATA FOR MODELING")
print("="*70)

# Separate features and target
X = df.drop(['default', 'AGE_GROUP', 'LIMIT_BAL_GROUP'], axis=1, errors='ignore')
y = df['default']

# Remove ID column if exists
if 'ID' in X.columns:
    X = X.drop('ID', axis=1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Default rate: {y.mean()*100:.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training default rate: {y_train.mean()*100:.2f}%")
print(f"Test default rate: {y_test.mean()*100:.2f}%")


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

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training set: {X_train_balanced.shape[0]} samples")
print(f"Balanced default rate: {y_train_balanced.mean()*100:.2f}%")

# Train models
results = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    print(f"\n{name} Results:")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1-Score:  {results[name]['f1']:.4f}")
    print(f"  MCC:       {results[name]['mcc']:.4f}")


# ============================================================================
# 8. FIGURE 5: MODEL COMPARISON BAR CHART
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

metrics = ['precision', 'recall', 'f1', 'mcc']
x = np.arange(len(metrics))
width = 0.25
colors_model = ['#3498db', '#2ecc71', '#e74c3c']

for i, (name, color) in enumerate(zip(results.keys(), colors_model)):
    values = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width, values, width, label=name, color=color, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 5: Model Performance Comparison (After SMOTE Balancing)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'MCC'])
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_figure('figure5_model_comparison')
plt.show()


# ============================================================================
# 9. FIGURE 6: CONFUSION MATRICES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, color) in enumerate(zip(results.keys(), colors_model)):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(f'{name}\n(TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})', 
                      fontsize=11)

plt.suptitle('Figure 6: Confusion Matrices for All Models', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure('figure6_confusion_matrices')
plt.show()


# ============================================================================
# 10. FIGURE 7: ROC CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Figure 7: ROC Curves for All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure('figure7_roc_curves')
plt.show()


# ============================================================================
# 11. FIGURE 8: PRECISION-RECALL CURVES (Better for Imbalanced Data)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for name in results.keys():
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, results[name]['y_pred_proba'])
    pr_auc = auc(recall_vals, precision_vals)
    ax.plot(recall_vals, precision_vals, linewidth=2, 
            label=f'{name} (PR-AUC = {pr_auc:.3f})')

# Baseline (proportion of positives)
baseline = y_test.mean()
ax.axhline(y=baseline, color='gray', linestyle='--', 
           label=f'Baseline (Default Rate = {baseline:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Figure 8: Precision-Recall Curves\n(Better metric for imbalanced data)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure('figure8_precision_recall_curves')
plt.show()


# ============================================================================
# 12. SHAP ANALYSIS FOR XGBOOST (Interpretability)
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
print("="*70)

# Get the best model (XGBoost)
best_model = results['XGBoost']['model']

# Create SHAP explainer
X_sample = X_test.sample(n=100, random_state=42)  # Sample for faster computation
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Get feature names
feature_names = X.columns.tolist()

# FIGURE 9: SHAP Summary Plot (Global Feature Importance)
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.title('Figure 9: SHAP Summary Plot - Global Feature Importance\n(Features pushing predictions toward default shown in red)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure('figure9_shap_summary_plot')
plt.show()

# FIGURE 10: SHAP Bar Plot (Mean Absolute SHAP)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                  plot_type="bar", show=False)
plt.title('Figure 10: Mean Absolute SHAP Values\n(Top Features by Impact on Model Output)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure('figure10_shap_bar_plot')
plt.show()

# FIGURE 11: SHAP Waterfall Plot for Single Prediction
fig, ax = plt.subplots(figsize=(12, 6))
# Get a defaulter from test set
defaulter_idx = y_test[y_test == 1].index[0]
defaulter_instance = X_test.loc[[defaulter_idx]]

shap_values_single = explainer.shap_values(defaulter_instance)
shap.waterfall_plot(shap.Explanation(values=shap_values_single[0], 
                                      base_values=explainer.expected_value,
                                      data=defaulter_instance.iloc[0].values,
                                      feature_names=feature_names),
                    show=False, max_display=10)
plt.title('Figure 11: SHAP Waterfall Plot - Explanation for a Defaulting Client\n(Red = pushes toward default, Blue = pushes away from default)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
save_figure('figure11_shap_waterfall')
plt.show()


# ============================================================================
# 13. FIGURE 12: FEATURE IMPORTANCE COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# XGBoost built-in feature importance
importance_xgb = best_model.feature_importances_
sorted_idx = np.argsort(importance_xgb)[-10:]
axes[0].barh(range(10), importance_xgb[sorted_idx])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[0].set_xlabel('XGBoost Feature Importance (Gain)')
axes[0].set_title('XGBoost Built-in Feature Importance', fontsize=12, fontweight='bold')

# SHAP-based feature importance (mean absolute SHAP values)
shap_importance = np.abs(shap_values).mean(axis=0)
shap_sorted_idx = np.argsort(shap_importance)[-10:]
axes[1].barh(range(10), shap_importance[shap_sorted_idx])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([feature_names[i] for i in shap_sorted_idx])
axes[1].set_xlabel('Mean |SHAP Value|')
axes[1].set_title('SHAP-Based Feature Importance', fontsize=12, fontweight='bold')

plt.suptitle('Figure 12: Feature Importance Comparison\n(PAY_0 = Most Recent Payment Status emerges as top predictor)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_figure('figure12_feature_importance_comparison')
plt.show()


# ============================================================================
# 14. SUMMARY TABLE (For Results Section)
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY TABLE")
print("="*70)

summary_table = pd.DataFrame({
    'Model': list(results.keys()),
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'MCC': [results[m]['mcc'] for m in results.keys()]
})

print("\n", summary_table.to_string(index=False))

# Export to CSV
summary_table.to_csv('model_results_summary.csv', index=False)
print("\n✓ Results saved to: model_results_summary.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated Visualizations (12 figures):")
print("  1. figure1_class_imbalance.png")
print("  2. figure2_correlation_heatmap.png")
print("  3. figure3_demographic_analysis.png")
print("  4. figure4_payment_history_impact.png")
print("  5. figure5_model_comparison.png")
print("  6. figure6_confusion_matrices.png")
print("  7. figure7_roc_curves.png")
print("  8. figure8_precision_recall_curves.png")
print("  9. figure9_shap_summary_plot.png")
print(" 10. figure10_shap_bar_plot.png")
print(" 11. figure11_shap_waterfall.png")
print(" 12. figure12_feature_importance_comparison.png")
print("\nAdditional outputs:")
print("  - model_results_summary.csv")