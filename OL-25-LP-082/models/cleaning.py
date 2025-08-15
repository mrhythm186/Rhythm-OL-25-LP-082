import pandas as pd
import numpy as np
df = pd.read_csv('survey.csv')
print(f"Original shape: {df.shape}")
df = df[(df['Age'] > 18) & (df['Age'] < 80)]
df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
gender_map = {
    'm': 'Male', 'male': 'Male', 'maile': 'Male', 'something kinda male?': 'Male',
    'cis male': 'Other', 'mal': 'Male', 'male (cis)': 'Male', 'queer/she/they': 'Female',
    'make': 'Male', 'guy (-ish) ^_^': 'Other', 'man': 'Male', 'male leaning androgynous': 'Other',
    'malr': 'Male', 'cis man': 'Other', 'mail': 'Male', 'nah': 'Other', 'all': 'Unknown',
    'fluid': 'Other', 'p': 'Other', 'neuter': 'Other', 'a little about you': 'Other'
}
df['Gender'] = df['Gender'].map(gender_map).fillna('Female')
fill_unknown_cols = [
    'state', 'self_employed', 'work_interfere', 'tech_company', 'benefits', 
    'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
for col in fill_unknown_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
df['no_employees'] = df['no_employees'].astype(str).str.strip()
employee_map = {'1-5':0, '6-25':1, '26-100':2, '100-500':3, '500-1000':4, 'More than 1000':5}
df['no_employees'] = df['no_employees'].map(employee_map).fillna(-1).astype(int)
self_emp_map = {'Yes':1, 'No':0, 'Unknown':-1}
df['self_employed'] = df['self_employed'].map(self_emp_map).fillna(-1).astype(int)
binary_cols = [
    'tech_company', 'benefits', 'care_options', 'wellness_program',
    'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
    'phys_health_consequence', 'coworkers', 'supervisor',
    'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].replace({'Yes':1, 'No':0, 'Unknown':-1})
df.to_csv('cleaned_survey.csv', index=False)
print(df.head())
print(df['no_employees'].value_counts())