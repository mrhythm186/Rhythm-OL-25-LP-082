import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Load dataset
df = pd.read_csv('cleaned_survey.csv')

# Convert binary columns from 0/1 to Yes/No
if df['treatment'].dtype in [int, float]:
    df['treatment'] = df['treatment'].map({1: 'Yes', 0: 'No'})

if df['benefits'].dtype in [int, float]:
    df['benefits'] = df['benefits'].map({1: 'Yes', 0: 'No'})

# Log transform Age
df['log_age'] = np.log1p(df['Age'])

# Create age groups
df['age_group'] = pd.cut(df['Age'], bins=[17, 25, 35, 50, 80], labels=['<25', '25-35', '36-50', '51+'])

# 1. Log Transformed Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['log_age'], kde=True)
plt.title("Log Transformed Age Distribution")
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(6,6))
df['Gender'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# 4. Self-Employment Distribution
plt.figure(figsize=(6,6))
df['self_employed'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Self-Employed Distribution')
plt.ylabel('')
plt.show()

# 5. Work Interference Distribution
plt.figure(figsize=(6,6))
sns.countplot(x='work_interfere', data=df, order=df['work_interfere'].value_counts().index, palette='pastel')
plt.title('Work Interference with Mental Health')
plt.xlabel('Work Interfere')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 6. Age Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Age'], bins=15, kde=True, color='orange')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 7. Treatment Seeking Distribution
plt.figure(figsize=(6,6))
df['treatment'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Treatment Seeking Distribution')
plt.xlabel('Treatment')
plt.ylabel('Count')
plt.show()

# 8. Treatment Seeking by Gender
plt.figure(figsize=(8,6))
sns.countplot(x='Gender', hue='treatment', data=df, palette='Set2')
plt.title('Treatment Seeking by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Treatment')
plt.show()

# 9. Treatment Seeking by Age Group
plt.figure(figsize=(8,6))
sns.countplot(x='age_group', hue='treatment', data=df, palette='muted')
plt.title('Treatment Seeking by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Treatment')
plt.show()

# 10. Awareness of Benefits
plt.figure(figsize=(6,6))
df['benefits'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Awareness of Mental Health Benefits')
plt.xlabel('Benefits')
plt.ylabel('Count')
plt.show()

# 11. Benefits vs Treatment Seeking
plt.figure(figsize=(8,6))
sns.countplot(x='benefits', hue='treatment', data=df, palette='coolwarm',
              order=df['benefits'].value_counts().index)
plt.title('Impact of Benefits Awareness on Treatment Seeking')
plt.xlabel('Benefits Awareness')
plt.ylabel('Count')
plt.legend(title='Treatment')
plt.show()
