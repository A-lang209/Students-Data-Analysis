# Students-Data-Analysis

# Educational Data Analysis: Uncovering Paths to Graduate School & Career Success

This project analyzes an educational dataset (`Students data.csv`) to investigate the complex factors influencing a student's journey after graduation—specifically focusing on who successfully transitions to graduate school (inland or abroad) versus who enters the job market.

## 📌 Project Overview
Educational data often contains hidden hierarchies, socio-economic privilege indicators, and high multicollinearity among academic metrics. Simple linear models miss these nuances. This project employs `pandas`, `numpy`, `matplotlib`, and `seaborn` to build an analytical pipeline that:
1. **Identifies the profile of a "brilliant student"** capable of handling graduate-level study.
2. **Mitigates multicollinearity** among heavily correlated indicators (e.g., numerous math course scores).
3. **Exposes socioeconomic disparities** and structural privilege that influence international application rates.
4. **Provides data-driven career counseling** for students transitioning directly into the workforce.

## 🗂️ Dataset Insights
The dataset contains demographics, academic variables (`GPA`, multiple advanced math scores like `Algebra`), home location variables, and a family wealth/poverty index (`from4`). 

The target variable (`y`) classifies outcomes into:
* `0`: Transitioning to the job market / preparing to re-apply
* `1`: Successful inland graduate admission
* `2`: Successful international graduate admission

## 🚀 Key Features of the Pipeline
* **Collinearity Control:** Combines highly correlated math indicators into a unified `Math_Average` metric to prevent variance inflation in future predictive models.
* **Privilege Stratification:** Uses stacked visualizations to map out how family wealth indexes (`from4`) heavily skew the likelihood of studying abroad, separating academic capability from financial means.
* **Targeted Career Profiling:** Filters out students entering the workforce ($y=0$) and maps their specific skill strengths (e.g., high probability/statistics scores vs. pure calculus) to recommend roles like Data Analyst, Risk Analyst, or Software Developer.
2. Install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn
