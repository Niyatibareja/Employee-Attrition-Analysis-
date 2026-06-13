# -*- coding: utf-8 -*-
"""
Fixed modeling section -- handles class imbalance with class_weight + threshold tuning,
and fixes feature importance for Logistic Regression via coefficients.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
import warnings, os, textwrap
warnings.filterwarnings("ignore")

OUT = "/home/claude/attrition_output"
os.makedirs(OUT, exist_ok=True)

NAVY=  "#1E2761"; MID= "#2D3A8C"; ACCENT="#F7941D"
TEAL=  "#0D9488"; RED= "#E53E3E"; GREEN= "#38A169"
LIGHT= "#CADCFC"; GRAY="#64748B"; BG="#F0F4FF"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})

def save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK]  {name}.png")

def makeShadow(): return {}   # not needed here

# -- Load dataset ----------------------------------------------------------
df_raw = pd.read_csv(f"{OUT}/ibm_hr_attrition.csv")

# -- Preprocess ------------------------------------------------------------
df_model = df_raw.copy()
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Attrition")
cat_cols = [c for c in cat_cols if c != "RiskLevel"] if "RiskLevel" in cat_cols else cat_cols
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])
df_model["Attrition"] = (df_model["Attrition"] == "Yes").astype(int)

drop_cols = [c for c in ["AttritionRiskScore","RiskLevel"] if c in df_model.columns]
df_model.drop(columns=drop_cols, inplace=True)

X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]
print(f"  Class distribution: {y.value_counts().to_dict()}")
print(f"  Attrition rate: {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -- Train models WITH class_weight='balanced' -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42,
                                               class_weight="balanced", C=0.1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42,
                                                   class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=8,
                                                   random_state=42, n_jobs=-1,
                                                   class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                       learning_rate=0.05,
                                                       max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    # For GB use sample_weight to simulate balanced
    if name == "Gradient Boosting":
        sw = np.where(y_train == 1,
                      len(y_train) / (2 * y_train.sum()),
                      len(y_train) / (2 * (len(y_train) - y_train.sum())))
        model.fit(Xtr, y_train, sample_weight=sw)
    else:
        model.fit(Xtr, y_train)

    y_prob = model.predict_proba(Xte)[:, 1]
    # Use 0.3 threshold instead of 0.5 to improve recall for minority class
    threshold = 0.30
    y_pred = (y_prob >= threshold).astype(int)

    cv = cross_val_score(model, Xtr, y_train,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
    results[name] = {
        "model": model, "Xte": Xte,
        "accuracy":  accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "recall":    recall_score(y_test, y_pred, zero_division=0) * 100,
        "f1":        f1_score(y_test, y_pred, zero_division=0) * 100,
        "auc":       roc_auc_score(y_test, y_prob) * 100,
        "cv_auc":    cv.mean() * 100,
        "y_pred": y_pred, "y_prob": y_prob,
    }
    print(f"  {name:<26} Acc={results[name]['accuracy']:.1f}%  "
          f"AUC={results[name]['auc']:.1f}%  "
          f"Recall={results[name]['recall']:.1f}%  F1={results[name]['f1']:.1f}%")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n  *  Best model: {best_name}  AUC={best['auc']:.1f}%  F1={best['f1']:.1f}%")

# -- Feature Importance ----------------------------------------------------
best_model = best["model"]
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    # Logistic Regression -- use abs(coef)
    fi = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)

# -- Plot 1: Model Comparison ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Predictive Model Comparison (Balanced Classes)",
             fontsize=14, fontweight="bold", color=NAVY)

metrics_keys   = ["accuracy","precision","recall","f1","auc"]
metrics_labels = ["Accuracy","Precision","Recall","F1","AUC"]
model_names    = list(results.keys())
x  = np.arange(len(model_names))
w  = 0.15
colors_m = [NAVY, ACCENT, TEAL, RED, GREEN]

ax = axes[0]
for i, (mk, ml) in enumerate(zip(metrics_keys, metrics_labels)):
    vals = [results[n][mk] for n in model_names]
    ax.bar(x + i*w, vals, w, label=ml, color=colors_m[i])
ax.set_xticks(x + w*2)
ax.set_xticklabels([n.replace(" ","\n") for n in model_names], fontsize=9)
ax.set_ylabel("Score (%)"); ax.set_ylim(0, 115)
ax.set_title("All Metrics by Model", fontweight="bold", color=NAVY)
ax.legend(loc="upper right", fontsize=8)
ax.axhline(70, color=GRAY, linestyle="--", lw=0.8, alpha=0.5)

ax = axes[1]
color_list = [NAVY, ACCENT, TEAL, RED]
for (name, res), color in zip(results.items(), color_list):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name.split()[0]} ({res['auc']:.1f}%)")
ax.plot([0,1],[0,1],"--", color=GRAY, lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves -- All Models", fontweight="bold", color=NAVY)
ax.legend(fontsize=9)
save(fig, "04_model_comparison")

# -- Plot 2: Best Model Detail ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model: {best_name} -- Confusion Matrix & Feature Importances",
             fontsize=13, fontweight="bold", color=NAVY)

ax = axes[0]
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", linewidths=1,
            xticklabels=["Stay","Leave"], yticklabels=["Stay","Leave"],
            annot_kws={"size":14,"fontweight":"bold"})
ax.set_xlabel("Predicted", fontweight="bold"); ax.set_ylabel("Actual", fontweight="bold")
ax.set_title("Confusion Matrix", fontweight="bold", color=NAVY)

ax = axes[1]
top15 = fi.sort_values(ascending=True).tail(15)
bar_cols = [RED if v > top15.quantile(0.7) else TEAL for v in top15.values]
ax.barh(top15.index, top15.values, color=bar_cols, height=0.6)
ax.set_xlabel("Importance Score")
ax.set_title(f"Top 15 Features -- {best_name}", fontweight="bold", color=NAVY)
save(fig, "05_best_model_details")

print(f"\n  Classification Report -- {best_name} (threshold=0.30):")
print(classification_report(y_test, best["y_pred"], target_names=["Stay","Leave"]))

# -- Risk Scoring ----------------------------------------------------------
Xall_sc = scaler.transform(X)
Xall    = Xall_sc if best_name == "Logistic Regression" else X
probs_all = best_model.predict_proba(Xall)[:, 1]

df_out = df_raw.copy()
df_out["AttritionRiskScore"] = (probs_all * 100).round(1)
df_out["RiskLevel"] = pd.cut(df_out["AttritionRiskScore"],
                              bins=[0,30,60,100], labels=["Low","Medium","High"])
df_out.to_csv(f"{OUT}/employees_with_risk_scores.csv", index=False, encoding="utf-8")

print(f"\n  Risk distribution:")
print(df_out["RiskLevel"].value_counts().to_string())

attr_rate = (df_raw["Attrition"] == "Yes").mean() * 100

# -- BI Dashboard (regenerate with correct data) ---------------------------
dept_rate = df_raw.groupby("Department")["Attrition"].apply(
    lambda x: (x=="Yes").mean()*100).sort_values(ascending=False)

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(NAVY)

header_ax = fig.add_axes([0, 0.93, 1, 0.07])
header_ax.set_facecolor(NAVY); header_ax.axis("off")
header_ax.text(0.5, 0.55, "HR ANALYTICS DASHBOARD -- Employee Attrition Intelligence",
               ha="center", va="center", fontsize=18, fontweight="bold", color="white")
header_ax.text(0.5, 0.1, f"IBM HR Analytics Dataset  -  1,470 Employees  -  Best Model: {best_name} (AUC {best['auc']:.0f}%)",
               ha="center", va="center", fontsize=10, color=LIGHT, style="italic")

kpis = [
    ("Total Employees",     "1,470",                                  MID,   "white"),
    ("Overall Attrition",   f"{attr_rate:.1f}%",                      RED,   "white"),
    ("High-Risk Employees", str(int((df_out['RiskLevel']=='High').sum())), ACCENT,"#1E2761"),
    ("Best Model AUC",      f"{best['auc']:.1f}%",                    GREEN, "white"),
    ("Best Recall",         f"{best['recall']:.1f}%",                 TEAL,  "white"),
]
for i, (title, val, bg, fg) in enumerate(kpis):
    ax = fig.add_axes([0.02 + i*0.196, 0.81, 0.175, 0.10])
    ax.set_facecolor(bg); ax.axis("off")
    ax.text(0.5, 0.68, val,   ha="center", va="center", fontsize=22, fontweight="bold", color=fg)
    ax.text(0.5, 0.18, title, ha="center", va="center", fontsize=9,  color=fg, alpha=0.85)

# Panel A - Attrition by Dept
ax1 = fig.add_axes([0.02, 0.53, 0.29, 0.25]); ax1.set_facecolor(BG)
bars = ax1.bar(range(len(dept_rate)), dept_rate.values, color=[RED,ACCENT,TEAL], width=0.5)
ax1.set_xticks(range(len(dept_rate)))
ax1.set_xticklabels([textwrap.fill(l,12) for l in dept_rate.index], fontsize=8)
ax1.set_ylabel("Attrition %", fontsize=9)
ax1.set_title("Attrition by Department", fontweight="bold", color=NAVY, fontsize=11)
for b in bars:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{b.get_height():.1f}%", ha="center", fontsize=9, color=NAVY)

# Panel B - Risk Donut
ax2 = fig.add_axes([0.34, 0.53, 0.18, 0.25]); ax2.set_facecolor(BG)
risk_counts = df_out["RiskLevel"].value_counts()
wedges, texts, autotexts = ax2.pie(
    risk_counts.values, labels=risk_counts.index, autopct="%1.0f%%",
    colors=[RED,ACCENT,GREEN][:len(risk_counts)], startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2,"width":0.6})
for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
ax2.set_title("Risk Distribution", fontweight="bold", color=NAVY, fontsize=11)

# Panel C - Income distribution
ax3 = fig.add_axes([0.55, 0.53, 0.21, 0.25]); ax3.set_facecolor(BG)
for label, color in [("No",GREEN),("Yes",RED)]:
    vals = df_raw[df_raw["Attrition"]==label]["MonthlyIncome"]
    ax3.hist(vals, bins=20, alpha=0.65, color=color, label=label, edgecolor="white")
ax3.set_xlabel("Monthly Income ($)", fontsize=9); ax3.set_ylabel("Count", fontsize=9)
ax3.set_title("Income Distribution by Attrition", fontweight="bold", color=NAVY, fontsize=11)
ax3.legend(title="Attrition", fontsize=8)

# Panel D - Model AUC
ax4 = fig.add_axes([0.79, 0.53, 0.19, 0.25]); ax4.set_facecolor(BG)
sorted_aucs = dict(sorted({k:v["auc"] for k,v in results.items()}.items(), key=lambda x:x[1]))
bar_cols4   = [GREEN if k==best_name else TEAL for k in sorted_aucs]
bars4 = ax4.barh([n.replace(" ","\n") for n in sorted_aucs],
                  list(sorted_aucs.values()), color=bar_cols4, height=0.5)
ax4.set_xlabel("AUC (%)", fontsize=9); ax4.set_xlim(50,110)
ax4.set_title("Model AUC Scores", fontweight="bold", color=NAVY, fontsize=11)
for b in bars4:
    ax4.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
             f"{b.get_width():.1f}%", va="center", fontsize=8, color=NAVY)

# Panel E - Satisfaction x WLB heatmap
ax5 = fig.add_axes([0.02, 0.24, 0.29, 0.26]); ax5.set_facecolor(BG)
pivot = df_raw.pivot_table(values="Attrition", index="JobSatisfaction",
                            columns="WorkLifeBalance",
                            aggfunc=lambda x:(x=="Yes").mean()*100)
sns.heatmap(pivot, ax=ax5, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.5, annot_kws={"size":9})
ax5.set_xlabel("Work-Life Balance ->", fontsize=9); ax5.set_ylabel("Job Satisfaction ^", fontsize=9)
ax5.set_title("Attrition % Heatmap\n(Satisfaction x Work-Life Balance)",
              fontweight="bold", color=NAVY, fontsize=11)

# Panel F - Feature Importances
ax6 = fig.add_axes([0.34, 0.24, 0.30, 0.26]); ax6.set_facecolor(BG)
top10 = fi.sort_values(ascending=True).tail(10)
bc6 = [RED if v>top10.quantile(0.7) else TEAL for v in top10.values]
ax6.barh(top10.index, top10.values, color=bc6, height=0.6)
ax6.set_xlabel("Importance Score", fontsize=9)
ax6.set_title(f"Top 10 Features -- {best_name}", fontweight="bold", color=NAVY, fontsize=11)

# Panel G - ROC curves
ax7 = fig.add_axes([0.67, 0.24, 0.30, 0.26]); ax7.set_facecolor(BG)
for (name, res), col in zip(results.items(), [NAVY,ACCENT,TEAL,RED]):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax7.plot(fpr, tpr, color=col, lw=2,
             label=f"{name.split()[0]} ({res['auc']:.0f}%)")
ax7.plot([0,1],[0,1],"--", color=GRAY, lw=1)
ax7.set_xlabel("FPR", fontsize=9); ax7.set_ylabel("TPR", fontsize=9)
ax7.set_title("ROC Curves", fontweight="bold", color=NAVY, fontsize=11)
ax7.legend(fontsize=8)

# Panel H - High-risk by dept
ax8 = fig.add_axes([0.02, 0.04, 0.45, 0.17]); ax8.set_facecolor(BG)
high_risk_dept = df_out[df_out["RiskLevel"]=="High"].groupby("Department").size()
total_dept     = df_out.groupby("Department").size()
hr_pct         = (high_risk_dept / total_dept * 100).fillna(0).sort_values(ascending=False)
bars8 = ax8.bar(hr_pct.index, hr_pct.values,
                color=[RED,ACCENT,TEAL][:len(hr_pct)], width=0.4)
ax8.set_ylabel("High-Risk %", fontsize=9)
ax8.set_xticklabels([textwrap.fill(l,14) for l in hr_pct.index], fontsize=9)
ax8.set_title("High-Risk % by Department", fontweight="bold", color=NAVY, fontsize=11)
for b in bars8:
    ax8.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{b.get_height():.1f}%", ha="center", fontsize=9, color=NAVY)

# Panel I - Risk score histogram
ax9 = fig.add_axes([0.52, 0.04, 0.45, 0.17]); ax9.set_facecolor(BG)
ax9.hist(df_out["AttritionRiskScore"], bins=30, color=MID, edgecolor="white", alpha=0.85)
ax9.axvline(30, color=GREEN, linestyle="--", lw=1.5, label="Low/Med boundary")
ax9.axvline(60, color=RED,   linestyle="--", lw=1.5, label="Med/High boundary")
mean_s = df_out["AttritionRiskScore"].mean()
ax9.axvline(mean_s, color=ACCENT, lw=2, label=f"Mean={mean_s:.1f}%")
ax9.set_xlabel("Attrition Risk Score (%)", fontsize=9)
ax9.set_ylabel("Employees", fontsize=9)
ax9.set_title("Employee Risk Score Distribution", fontweight="bold", color=NAVY, fontsize=11)
ax9.legend(fontsize=8)

save(fig, "06_bi_dashboard")

# -- Updated Findings Report -----------------------------------------------
top5 = fi.sort_values(ascending=False).head(5)
report = f"""
+==============================================================================+
|            EMPLOYEE ATTRITION FORECASTING -- FINAL FINDINGS REPORT          |
+==============================================================================+

---------------------------------------
  DATASET SUMMARY
---------------------------------------
  Total Employees         : 1,470
  Features Analyzed       : {X.shape[1]}
  Overall Attrition Rate  : {attr_rate:.1f}%
  Missing Values          : 0
  Class Imbalance Fix     : class_weight=balanced + threshold=0.30

---------------------------------------
  EDA KEY FINDINGS
---------------------------------------
  1. Department highest attrition : {dept_rate.idxmax()} ({dept_rate.max():.1f}%)
  2. OverTime 'Yes' attrition rate: {df_raw[df_raw['OverTime']=='Yes']['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%
     OverTime 'No'  attrition rate: {df_raw[df_raw['OverTime']=='No']['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%
  3. Single employees attrition   : {df_raw[df_raw['MaritalStatus']=='Single']['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%
  4. Frequent travellers face highest attrition risk
  5. Low Job Satisfaction (1) -> ~2x attrition vs High (4)

---------------------------------------
  MODEL PERFORMANCE (threshold = 0.30)
---------------------------------------
  {"Model":<26} {"Acc":>7} {"AUC":>7} {"Recall":>8} {"F1":>7}
  {"-"*58}"""
for name, res in results.items():
    star = "  *" if name == best_name else ""
    report += (f"\n  {name:<26} {res['accuracy']:>6.1f}%"
               f" {res['auc']:>6.1f}%"
               f" {res['recall']:>7.1f}%"
               f" {res['f1']:>6.1f}%{star}")

report += f"""

---------------------------------------
  TOP 5 PREDICTIVE FEATURES -- {best_name}
---------------------------------------"""
for feat, score in top5.items():
    report += f"\n  - {feat:<30}  {score:.4f}"

report += f"""

---------------------------------------
  RISK SEGMENTATION
---------------------------------------
  High Risk  (>60%) : {int((df_out['RiskLevel']=='High').sum()):>4} employees
  Medium Risk(30-60%): {int((df_out['RiskLevel']=='Medium').sum()):>4} employees
  Low Risk   (<30%) : {int((df_out['RiskLevel']=='Low').sum()):>4} employees

---------------------------------------
  STRATEGIC RECOMMENDATIONS
---------------------------------------
  1. OVERTIME POLICY     -- Cap mandatory overtime; pilot flexible schedules
  2. CAREER PATHING      -- Structured promotions for employees at 3-5 yr mark
  3. COMPENSATION REVIEW -- Benchmark pay for Sales & HR departments
  4. PULSE SURVEYS       -- Monthly check-ins to catch satisfaction drops early
  5. TARGETED RETENTION  -- Personalised packages for high-risk employees
  6. TRAVEL INCENTIVES   -- Compensate or reduce frequent-travel assignments

+==============================================================================+
"""
print(report)
with open(f"{OUT}/findings_report.txt","w", encoding="utf-8") as f: f.write(report)
print("  All files updated. [OK]")
