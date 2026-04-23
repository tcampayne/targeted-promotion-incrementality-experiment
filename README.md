# 🧪 A/B Promotion Incrementality Analysis

## 📌 Overview

This project evaluates whether offering a **10% discount to high-LTV users** generates **true incremental revenue** or primarily subsidizes existing purchasing behavior.

The analysis combines **randomized A/B testing**, **panel methods**, and **causal inference techniques** to assess both **average impact** and **model sensitivity**.

---

## ❓ Business Question

> Does a 10% discount drive incremental revenue, or does it reduce margin by discounting purchases that would have occurred anyway?

---

## 📊 Key Results

- **+$8.50 incremental revenue per user**  
- **95% CI:** [7.83, 9.16]  
- **-$2.99 net impact per user** after discount cost  

### ✅ Conclusion

While the promotion increases revenue, it is **not profitable** at a 10% blanket discount level.

This suggests:
- Value in **targeted discounting**
- Risk of **subsidizing existing behavior**

---

## 🚀 Interactive App

👉 **Live Streamlit App:**  
https://targeted-promotion-incrementality-experiment-ltfx4qbfgnsn5ey2x.streamlit.app  

The app includes:
- A/B test results with confidence intervals  
- Event study (TWFE) visualization  
- Model comparison (Naive, FE, TWFE, Weighted DiD)  
- Business impact breakdown  
- Heterogeneous treatment effects  

---

## 🧠 Experimental Design

- Randomized controlled experiment  
- ~61.9K users  
- Treatment: 10% discount  
- Control: no discount  
- Multi-week panel dataset  

---

## 🏗️ Data Pipeline

Built a reproducible SQL pipeline:
staging → core → marts

- Cleaned and structured raw event data  
- Created user-level and panel-level datasets  
- Ensured experiment-ready schema for analysis  

---

## 🔬 Methodology

### 1. A/B Test (Primary Estimate)
- Post-period difference in means  
- Interpreted as **Average Treatment Effect (ATE)**  

---

### 2. Difference-in-Differences

Estimated treatment effects using:

- Naive DiD  
- User fixed effects  
- Two-way fixed effects (user + time)  

---

### 3. Event Study Analysis

- Estimated dynamic treatment effects over time  
- Used to validate the **parallel trends assumption**  

### ⚠️ Finding:
Pre-treatment trends differ between treatment and control groups, indicating a **violation of parallel trends**.

---

### 4. Weighted DiD (Robustness Check)

- Applied reweighting to improve comparability  
- Adjusts for observable differences between groups  

**Note:** Improves balance but does not fully resolve violations of parallel trends.

---

### 5. Heterogeneous Treatment Effects

- Implemented **causal forests (EconML)**  
- Estimated user-level treatment effects  

### Finding:
Users with higher baseline spend show **larger absolute revenue lift**, suggesting potential for targeted discount strategies.

---

## ⚠️ Limitations

- **Parallel trends assumption violated**, weakening DiD interpretation  
- Revenue is simulated (not real transaction data)  
- Discount cost model is simplified  
- Results are sensitive to modeling assumptions  

---

## 💡 Design Improvements

Future experiments could be strengthened by:

- **Stratified randomization** (e.g., by baseline spend / LTV)  
- More granular user segmentation  
- Longer pre-treatment observation windows  

---

## 🔄 Analysis Workflow
Raw Data → SQL Pipeline → Panel Dataset → A/B Test → DiD Models → Event Study → HTE → Business Decision
---

## 🛠️ Tools & Technologies

- SQL (data pipeline)  
- Python (analysis)  
- Pandas, NumPy  
- Statsmodels (regression)  
- EconML (causal forests)  
- Streamlit (interactive app)  

---

## 📂 Repo Structure
data/
raw/
processed/

sql/
staging/
core/
marts/

analysis/
notebooks/

app/
streamlit_app.py

---

## 🎯 Key Takeaways

- Positive revenue lift does **not imply profitability**  
- Randomized ATE is the most reliable estimate  
- DiD estimates depend on assumptions and specification  
- Parallel trends violations must be explicitly diagnosed  
- Targeting is critical for promotion effectiveness  
- Experimental design is as important as modeling  

---
