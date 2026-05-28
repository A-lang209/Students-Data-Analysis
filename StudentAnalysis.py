import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plotting style for cleaner visuals
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================

# Load the dataset
df = pd.read_csv("Students data.csv")

# Map target variable for clearer visualizations
target_map = {0: "Failed / Job Search", 1: "Inland Grad", 2: "Abroad Grad"}
df["Outcome"] = df["y"].map(target_map)

# Define math courses for quick reference
math_courses = [
    "Algebra",
    "Calculus1",
    "Calculus2",
    "Statistics",
    "Probability",
    "Measure",
    "Functional_analysis",
]

print("--- Dataset Overview ---")
print(df.info())
print("\nTarget Distribution:\n", df["Outcome"].value_counts())

# ==========================================
# 2. ADDRESSING TIP 1: MULTICOLLINEARITY (The Math Scores)
# ==========================================
print("\n--- Investigating Collinearity among Math Courses ---")

# Calculate correlation matrix for math courses and GPA
corr_matrix = df[["GPA"] + math_courses].corr()

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: GPA & Math Courses (Detecting Collinearity)")
plt.tight_layout()
plt.show()

# Actionable Insight Code: Calculate an "Average Math Score" to combat collinearity
df["Math_Average"] = df[math_courses].mean(axis=1)

# ==========================================
# 3. GOAL 1: WHAT MAKES A BRILLIANT STUDENT? (Prediction Insights)
# ==========================================
print("\n--- Profiling Successful Graduate Applicants ---")

# Check how academic performance relates to the outcome
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x="Outcome", y="GPA", palette="Set2")
plt.title("GPA Distribution by Graduate Outcome")

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x="Outcome", y="Math_Average", palette="Set2")
plt.title("Math Average Distribution by Graduate Outcome")
plt.tight_layout()
plt.show()

# Investigating Heterogeneity: Does 'class' or 'gender' change the outcome?
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x="class", hue="Outcome", palette="viridis")
plt.title("Outcomes Across Different Classes")

plt.subplot(1, 2, 2)
sns.countplot(data=df, x="gender", hue="Outcome", palette="pastel")
plt.title("Outcomes Across Gender")
plt.tight_layout()
plt.show()

# ==========================================
# 4. ADDRESSING TIP 2: PRIVILEGE & BACKGROUND (`from4` wealth index)
# ==========================================
print("\n--- Analyzing the Impact of Background / Privilege ---")

# Cross-tabulate family background (from4: 0=Wealthier, 4=Poorer) with outcomes
pivot_bg = pd.crosstab(
    df["from4"], df["Outcome"], normalize="index"
) * 100
print("\nPercentage of Outcomes based on Family Background (0=Wealthy, 4=Poor):")
print(pivot_bg.round(2))

# Visualize Privilege vs. Success
pivot_bg.plot(kind="bar", stacked=True, colormap="viridis", edgecolor="black")
plt.title("How Family Background (from4) Affects Graduate School Outcomes")
plt.ylabel("Percentage (%)")
plt.xlabel("Poverty Index (0 = Most Wealthy, 4 = Most Poor)")
plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ==========================================
# 5. GOAL 2: ADVICE FOR JOB SEARCHERS (y = 0)
# ==========================================
print("\n--- Generating Insights for Failed Applicants (Job Search Advice) ---")

# Filter for students who failed to get into grad school
failed_students = df[df["y"] == 0]

# Let's see what their strengths are to guide their job search
print(f"Total students needing job advice: {len(failed_students)}")

# Boxplot of all math skills for failed students to see where they peak
plt.figure(figsize=(12, 6))
sns.boxplot(data=failed_students[math_courses])
plt.title("Skill Profile of Students Transitioning to the Job Market")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
