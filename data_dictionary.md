# Data Dictionary — Gallstone Dataset

This file documents each variable in the dataset, its meaning, and notes from EDA.

> ✅ Note: No missing values were found in this dataset.

| Column Name | Type | Description | Units | Notes (EDA findings) |
|-------------|------|-------------|-------|-----------------------|
| Gallstone Status | Categorical (0/1) | Target variable: gallstones present (1) or absent (0) | - | Range: 0–1, Skewness=0.02 (approx. symmetric) |
| Age | Numeric | Patient age | Years | Range: 20–96, Skewness=0.13 (approx. symmetric) |
| Gender | Categorical | Patient sex (Male/Female) | - | Range: 0–1, Skewness=0.03 (approx. symmetric) |
| Comorbidity | Categorical | Presence of other conditions (e.g. diabetes, hypertension) | - | Range: 0–3, Skewness=1.45 (right-skewed) |
| Coronary Artery Disease (CAD) | Numeric |  |  | Range: 0–1, Skewness=4.88 (right-skewed) |
| Hypothyroidism | Numeric |  |  | Range: 0–1, Skewness=5.73 (right-skewed) |
| Hyperlipidemia | Numeric |  |  | Range: 0–1, Skewness=6.10 (right-skewed) |
| Diabetes Mellitus (DM) | Numeric |  |  | Range: 0–1, Skewness=2.15 (right-skewed) |
| Height | Numeric |  |  | Range: 145–191, Skewness=-0.08 (approx. symmetric) |
| Weight | Numeric |  |  | Range: 42.9–143.5, Skewness=0.43 (approx. symmetric) |
| Body Mass Index (BMI) | Numeric |  |  | Range: 17.4–49.7, Skewness=0.67 (right-skewed) |
| Total Body Water (TBW) | Numeric |  |  | Range: 13.0–66.2, Skewness=0.21 (approx. symmetric) |
| Extracellular Water (ECW) | Numeric |  |  | Range: 9.0–27.8, Skewness=0.02 (approx. symmetric) |
| Intracellular Water (ICW) | Numeric |  |  | Range: 13.8–57.1, Skewness=0.95 (right-skewed) |
| Extracellular Fluid/Total Body Water (ECF/TBW) | Numeric |  |  | Range: 29.23–52.0, Skewness=-0.51 (left-skewed) |
| Total Body Fat Ratio (TBFR) (%) | Numeric |  |  | Range: 6.3–50.92, Skewness=0.13 (approx. symmetric) |
| Lean Mass (LM) (%) | Numeric |  |  | Range: 48.99–93.67, Skewness=-0.13 (approx. symmetric) |
| Body Protein Content (Protein) (%) | Numeric |  |  | Range: 5.56–24.81, Skewness=-0.05 (approx. symmetric) |
| Visceral Fat Rating (VFR) | Numeric |  |  | Range: 1–31, Skewness=0.80 (right-skewed) |
| Bone Mass (BM) | Numeric |  |  | Range: 1.4–4.0, Skewness=0.21 (approx. symmetric) |
| Muscle Mass (MM) | Numeric |  |  | Range: 4.7–78.8, Skewness=-0.10 (approx. symmetric) |
| Obesity (%) | Numeric |  |  | Range: 0.4–1954.0, Skewness=16.87 (right-skewed) |
| Total Fat Content (TFC) | Numeric |  |  | Range: 3.1–62.5, Skewness=0.81 (right-skewed) |
| Visceral Fat Area (VFA) | Numeric |  |  | Range: 0.9–41.0, Skewness=1.06 (right-skewed) |
| Visceral Muscle Area (VMA) (Kg) | Numeric |  |  | Range: 18.9–41.1, Skewness=-0.06 (approx. symmetric) |
| Hepatic Fat Accumulation (HFA) | Numeric |  |  | Range: 0–4, Skewness=0.18 (approx. symmetric) |
| Glucose | Numeric | Fasting blood glucose level | mg/dL | Range: 69.0–575.0, Skewness=5.94 (right-skewed) |
| Total Cholesterol (TC) | Numeric |  |  | Range: 60.0–360.0, Skewness=0.43 (approx. symmetric) |
| Low Density Lipoprotein (LDL) | Numeric |  |  | Range: 11.0–293.0, Skewness=0.54 (right-skewed) |
| High Density Lipoprotein (HDL) | Numeric |  |  | Range: 25.0–273.0, Skewness=6.53 (right-skewed) |
| Triglyceride | Numeric |  |  | Range: 1.39–838.0, Skewness=2.79 (right-skewed) |
| Aspartat Aminotransferaz (AST) | Numeric |  |  | Range: 8.0–195.0, Skewness=6.99 (right-skewed) |
| Alanin Aminotransferaz (ALT) | Numeric |  |  | Range: 3.0–372.0, Skewness=7.28 (right-skewed) |
| Alkaline Phosphatase (ALP) | Numeric |  |  | Range: 7.0–197.0, Skewness=0.80 (right-skewed) |
| Creatinine | Numeric |  |  | Range: 0.46–1.46, Skewness=0.62 (right-skewed) |
| Glomerular Filtration Rate (GFR) | Numeric |  |  | Range: 10.6–132.0, Skewness=-1.81 (left-skewed) |
| C-Reactive Protein (CRP) | Numeric |  |  | Range: 0.0–43.4, Skewness=5.41 (right-skewed) |
| Hemoglobin (HGB) | Numeric |  |  | Range: 8.5–18.8, Skewness=-0.38 (approx. symmetric) |
| Vitamin D | Numeric |  |  | Range: 3.5–53.1, Skewness=0.28 (approx. symmetric) |
