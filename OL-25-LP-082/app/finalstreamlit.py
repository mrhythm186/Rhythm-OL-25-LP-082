import streamlit as st
import numpy as np
import joblib 
import pandas as pd

st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "About",
        "Exploratory Data Analysis",
        "Predict Age",
        "Predicting Treatment Seeking",
        "Persona Clustering"
    ]
)
# About
if menu == "About":
    st.title("OpenLearn 1.0 ML Capstone Project")
    st.divider()
    st.header("Dataset Overview")
    st.markdown("""
    ### Dataset Source: [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    ### Collected by OSMI (Open Sourcing Mental Illness)
   
    """)

    st.header("About the project")
    st.markdown("""
                In this project, we use a dataset of over 1,500 survey entries from the OSMI Mental Health in Tech Survey, 
                which records information about age, gender, country, employment type, work environment, remote work frequency, 
               workplace support policies, past mental health history, family history, and willingness to discuss mental health. 
             These factors provide a comprehensive view of the demographic, personal, and workplace influences 
             on mental wellness in the technology sector.

        ### Contents:
        * **Exploratory Data Analysis**
        * **Supervised Learning**:
            * *Classification task*: Predict whether a person is likely to seek mental health treatment (treatment column: yes/no)
            * *Regression task*: Predict the respondent's age
        * **Unsupervised Learning**: Cluster tech workers into mental health personas
        * **Streamlit App Deployment**
    """)

  

elif menu =="Exploratory Data Analysis":
    st.title("📊 Data Analysis and Observations")
    st.divider()
    st.write("This dataset had many anomalies, null values, outliers, and imbalanced data in columns like `Gender`, `Age`, `Country`, etc. which needed to be cleaned" \
    "and standardised and following are the observations that were made.")
 
    st.divider()
    
    st.image("OL-25-LP-082/Images/gender distri.png", caption="Gender Distribution", use_container_width=True)
    
    st.markdown("""
        ### A **significant majority** of respondents identify as **Male**, making up the largest proportion of the dataset.
     - **Female** respondents represent a smaller but still notable portion of the population.
     - A minimal percentage of participants identify as **Other** or non-binary genders.
     - This imbalance suggests potential gender representation bias in the dataset, which should be considered when interpreting results.
        """)


    st.image("OL-25-LP-082/Images/work inter.png", caption="Work Interference", use_container_width=True)

    st.markdown("""The majority of respondents report mental health interference at work occurring 'Sometimes', with fewer experiencing it 'Often'.  
        This suggests that while workplace impact on mental health exists, it is not persistently high for most employees.
      """)
    st.image("OL-25-LP-082/Images/age distri.png", caption="Age Distribution", use_container_width=True)
    st.markdown("""Most participants are aged 25–35, indicating the dataset primarily reflects early to mid-career professionals.  
     Older age groups are underrepresented, with very few respondents over 50.
     """)

    st.divider()
   
    st.image("OL-25-LP-082/Images/treatment.png", caption="Treatment Seeking", use_container_width=True)
    st.markdown("""
        The split between those seeking and not seeking treatment is nearly balanced, with a slight tilt toward treatment seekers.  
        This reflects moderate openness toward addressing mental health but also highlights hesitation among many.
        """)

    st.image("OL-25-LP-082/Images/treatment gender.png", caption="Treatment Seeking By Gender", use_container_width=True)
    st.markdown("""
     Males form the largest group overall, but proportionally, females are more likely to seek treatment.  
     Other genders have low representation but appear in both seeking and non-seeking categories.
      """)

    
    st.image("OL-25-LP-082/Images/treatment age.png", caption="Treatment Seeking By Age", use_container_width=True)
    st.markdown("""The 25–35 age group dominates in both treatment-seeking and non-seeking categories.  
     A notable trend is that individuals aged 36–50 show a stronger tendency to seek treatment,  
     while younger participants (<25) are evenly split in their responses.
    
     """)
    st.image("OL-25-LP-082/Images/benifits awareness.png", caption="Awareness about Benefits", use_container_width=True)
    st.markdown("""The largest proportion of respondents are aware of their mental health benefits (category **1**).  
     - A significant portion of participants are unsure about whether they have such benefits (“don’t know”).  
     - The smallest group reported not having mental health benefits (category **0**).  
     - Awareness levels are not overwhelmingly high, as the combined “don’t know” and “no” categories are nearly equal to the “yes” category.  
     - The uncertainty among a large number of respondents suggests that communication about benefits may be lacking in workplaces.  
     
     """)

    st.image("OL-25-LP-082/Images/impact benifits.png", caption="Treatment by Awareness about Benefits", use_container_width=True)
    st.markdown("""
     - A large majority of respondents in the **“Don’t know”** category still chose to seek treatment, with around 250 people opting for it out of approximately 300.  
     - Only about 50 individuals in the **“Don’t know”** category avoided treatment, showing that uncertainty about benefits does not heavily discourage seeking help.  
     - Factors other than benefits awareness may play a stronger role in influencing treatment decisions.  
     - Lack of clarity about benefits could still contribute to treatment avoidance for a small portion of individuals.  
     - Clearer workplace communication about healthcare benefits may help ensure that even the hesitant minority feels supported to seek care.  

     """)


  
  

elif menu == 'Predict Age':
    st.set_page_config(page_title="Age Prediction", layout="centered")
    st.title("📊 Age Prediction")
    MODEL_PATH = "OL-25-LP-082/app/reg_model.pkl"
    model_wrap = joblib.load(MODEL_PATH)
    estimator = getattr(model_wrap, "best_estimator_", model_wrap)
    preprocessor = estimator.named_steps['preprocessor']
    feature_names = list(preprocessor.feature_names_in_)
    numeric_cols = []
    categorical_cols = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num' or str(name).lower().startswith('num'):
            cols_names = [feature_names[i] if isinstance(i, int) else i for i in cols]
            numeric_cols.extend(cols_names)
        elif name == 'cat' or str(name).lower().startswith('cat'):
            cols_names = [feature_names[i] if isinstance(i, int) else i for i in cols]
            categorical_cols.extend(cols_names)

    for f in feature_names:
        if f not in numeric_cols and f not in categorical_cols:
            categorical_cols.append(f)

    yes_no_unknown = ['Unknown', 'Yes', 'No']
    gender_opts = ['Male', 'Female', 'Other', 'Unknown']
    work_interfere_opts = ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown']
    no_employees_opts = ['Unknown', '1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'] 

    input_values = {}

    for feat in feature_names:
        if feat in numeric_cols:
            if feat == 'self_employed':
                val = st.selectbox("Are you self-employed?", yes_no_unknown, key=feat)
                input_values[feat] = {'Yes':1,'No':0,'Unknown':-1}[val]
            elif feat == 'no_employees':
                val = st.selectbox("Number of employees (company size):", no_employees_opts, key=feat)
                emp_map = {'Unknown':-1,'1-5':0,'6-25':1,'26-100':2,'100-500':3,'500-1000':4,'More than 1000':5}
                input_values[feat] = emp_map[val]
            else:
                val = st.selectbox(f"{feat} (Yes/No/Unknown):", yes_no_unknown, key=feat)
                input_values[feat] = {'Yes':1,'No':0,'Unknown':-1}[val]
        else: 
            if feat == 'Gender':
                val = st.selectbox("Gender", gender_opts, key=feat)
            elif feat == 'work_interfere':
                val = st.selectbox("Work interference", work_interfere_opts, key=feat)
            elif feat == 'family_history':
                val = st.selectbox("Family history of mental illness?", yes_no_unknown, key=feat)
            elif feat == 'remote_work':
                val = st.selectbox("Do you work remotely at least 50% of the time?", yes_no_unknown, key=feat)
            else:
                val = st.selectbox(
                    feat,
                    ['Unknown','Yes','No','Maybe','Not sure',"Don't know",
                     'Some of them','Often','Rarely','Never','Sometimes',
                     'Very easy','Somewhat easy','Somewhat difficult','Very difficult'],
                    key=feat
                )
            input_values[feat] = val

    input_df = pd.DataFrame([input_values], columns=feature_names)

    if st.button("Predict"):
        pred = estimator.predict(input_df)
        val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        st.success(f"Predicted Age: {val:.1f} years")




elif menu == "Predicting Treatment Seeking":
    st.header("Treatment Prediction")
    st.subheader('Predicting whether an employee is likely to seek mental health treatment')
    st.caption("Model Used: RandomForestClassifier")

    # --- User Inputs ---
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment?', ['Yes', 'No'])
    work_interfere = st.selectbox('Does condition interfere with work?', ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    remote_work = st.selectbox('Work remotely at least 50%?', ['Yes', 'No'])
    benefits = st.selectbox('Employer provides mental health benefits?', ["Don't know", 'Yes', 'No'])
    care_options = st.selectbox('Know options for mental health care?', ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox('Employer discussed mental health in wellness program?', ["Don't know", 'Yes', 'No'])
    seek_help = st.selectbox('Employer provides resources to learn about mental health?', ['Yes', 'No'])
    leave = st.selectbox('Ease of taking mental health leave?', ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    mental_health_consequence = st.selectbox('Negative consequences of discussing mental health?', ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Discuss mental health with coworkers?', ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Bring up mental health in interview?', ['No', 'Yes'])
    supervisor = st.selectbox('Discuss mental health with supervisor(s)?', ['No', 'Maybe', 'Yes'])

    # --- Convert inputs to numeric encoding matching training ---
    yes_no_unknown_map = {'Yes':1, 'No':0, 'Unknown':-1, "Don't know":-1}
    some_map = {'Some of them':1, 'No':0, 'Yes':2, 'Maybe':-1, 'Not sure':-1}
    work_interfere_map = {'Often':3, 'Rarely':1, 'Never':0, 'Sometimes':2, 'Unknown':-1}
    leave_map = {'Very easy':3, 'Somewhat easy':2, 'Somewhat difficult':1, 'Very difficult':0, "Don't know":-1}

    input_df = pd.DataFrame([{
        'Gender': gender,
        'self_employed': yes_no_unknown_map[self_employed],
        'family_history': yes_no_unknown_map[family_history],
        'treatment': yes_no_unknown_map[treatment],
        'work_interfere': work_interfere_map[work_interfere],
        'remote_work': yes_no_unknown_map[remote_work],
        'benefits': yes_no_unknown_map[benefits],
        'care_options': some_map[care_options],
        'wellness_program': some_map[wellness_program],
        'seek_help': yes_no_unknown_map[seek_help],
        'leave': leave_map[leave],
        'mental_health_consequence': some_map[mental_health_consequence],
        'coworkers': some_map[coworkers],
        'mental_health_interview': yes_no_unknown_map[mental_health_interview],
        'supervisor': some_map[supervisor]
    }])

    # --- Prediction ---
    if st.button('Predict'):
        clf = joblib.load('clf_model.pkl')  # pipeline or preprocessor must be included in this model
        predicted_treatment = clf.predict(input_df)
        if predicted_treatment[0] == 1:
            st.success('Yes, likely to seek treatment')
        else:
            st.warning('No, unlikely to seek treatment')

  

    






elif menu =="Persona Clustering":
    st.title("📊 Clustering Analysis")
    st.divider()
    st.markdown("The objective of this task is to make clusters and group tech workers according to their mental health personas. Below are some of the techniques and algorithms applied for the same.")
    st.write("The columns `Age`, `Country`, `Gender`, `no_employees`, `wellness_program`, `care_options`, `mental_health_consequence`, `benefits` were dropped due to their less contribution to the overall cluster making. These features" \
    "somewhere get covered in the rest of the questionnaire filled by the respondents.")

    # Clustering techniques
    st.subheader("Techniques Used: ")
    st.write(" - Principal Component Analysis (PCA)\n - t-distributed Stochastic Neighbor Embedding (t-SNE)")

    st.divider()

    st.image(
    "OL-25-LP-082/Images/Screenshot 2025-08-15 154009.png", 
    caption="Clusters formed by the models", 
    use_container_width=True
 )

    st.divider()

    st.markdown("###  These are the different Personas the respondents can be classified into ⬇️")

    tab1, tab2, tab3, tab4 = st.tabs([
    " Cluster 1", " Cluster 2", " Cluster 3",
    " Cluster 4"
    ])

    with tab1:
        st.markdown("""
        ###  Cluster 1:"Disengaged Strugglers" (Top-Left)

           Individuals with significant needs (e.g., stress, untreated conditions) but minimal interaction with support systems.

          High distress, low visibility—likely to avoid seeking help due to stigma or lack of awareness.
        """)

    with tab2:
        st.markdown("""
        ###  Cluster 2:"Proactive Engagers" (Top-Right)

         Active participants who utilize available resources and advocate for themselves or others.

         High awareness and agency, but variability suggests diverse engagement levels (e.g., some may burnout from over-involvement).
        """)

    with tab3:
        st.markdown("""
        ###  Cluster 3: System-Dependent" (Bottom-Left)

          Reliant on existing structures (e.g., workplace benefits, routine care) without seeking additional support.

         Stable but vulnerable—may collapse if systems fail them.
        """)

    with tab4:
        st.markdown("""
        ###  Cluster 4:"Passive Observers" (Bottom-Right)

      Aware of resources but disinclined to act unless compelled (e.g., by crisis or external pressure).

      Majority group representing untapped potential for intervention.


        """)

















