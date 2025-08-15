import streamlit as st
import numpy as np
import joblib 
import pandas as pd

st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "About",
        "Exploratory Data Analysis and model report",
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

  

elif menu =="Exploratory Data Analysis and model report":
    st.title("📊 Data Analysis and Observations")
    st.divider()
    st.write("This dataset had many anomalies, null values, outliers, and imbalanced data in columns like `Gender`, `Age`, `Country`, etc. which needed to be cleaned" \
    "and standardised and following are the observations that were made.")
 
    st.divider()
    
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 154050.png", caption="Gender Distribution", use_container_width=True)
    
    st.markdown("""
        ### A **significant majority** of respondents identify as **Male**, making up the largest proportion of the dataset.
     - **Female** respondents represent a smaller but still notable portion of the population.
     - A minimal percentage of participants identify as **Other** or non-binary genders.
     - This imbalance suggests potential gender representation bias in the dataset, which should be considered when interpreting results.
        """)


    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181826.png", caption="Work Interference", use_container_width=True)

    st.markdown("""The majority of respondents report mental health interference at work occurring 'Sometimes', with fewer experiencing it 'Often'.  
        This suggests that while workplace impact on mental health exists, it is not persistently high for most employees.
      """)
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181833.png", caption="Age Distribution", use_container_width=True)
    st.markdown("""Most participants are aged 25–35, indicating the dataset primarily reflects early to mid-career professionals.  
     Older age groups are underrepresented, with very few respondents over 50.
     """)

    st.divider()
   
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181838.png", caption="Treatment Seeking", use_container_width=True)
    st.markdown("""
        The split between those seeking and not seeking treatment is nearly balanced, with a slight tilt toward treatment seekers.  
        This reflects moderate openness toward addressing mental health but also highlights hesitation among many.
        """)

    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181844.png", caption="Treatment Seeking By Gender", use_container_width=True)
    st.markdown("""
     Males form the largest group overall, but proportionally, females are more likely to seek treatment.  
     Other genders have low representation but appear in both seeking and non-seeking categories.
      """)

    
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181851.png", caption="Treatment Seeking By Age", use_container_width=True)
    st.markdown("""The 25–35 age group dominates in both treatment-seeking and non-seeking categories.  
     A notable trend is that individuals aged 36–50 show a stronger tendency to seek treatment,  
     while younger participants (<25) are evenly split in their responses.
    
     """)
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181857.png", caption="Awareness about Benefits", use_container_width=True)
    st.markdown("""The largest proportion of respondents are aware of their mental health benefits (category **1**).  
     - A significant portion of participants are unsure about whether they have such benefits (“don’t know”).  
     - The smallest group reported not having mental health benefits (category **0**).  
     - Awareness levels are not overwhelmingly high, as the combined “don’t know” and “no” categories are nearly equal to the “yes” category.  
     - The uncertainty among a large number of respondents suggests that communication about benefits may be lacking in workplaces.  
     
     """)

    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 181905.png", caption="Treatment by Awareness about Benefits", use_container_width=True)
    st.markdown("""
     - A large majority of respondents in the **“Don’t know”** category still chose to seek treatment, with around 250 people opting for it out of approximately 300.  
     - Only about 50 individuals in the **“Don’t know”** category avoided treatment, showing that uncertainty about benefits does not heavily discourage seeking help.  
     - Factors other than benefits awareness may play a stronger role in influencing treatment decisions.  
     - Lack of clarity about benefits could still contribute to treatment avoidance for a small portion of individuals.  
     - Clearer workplace communication about healthcare benefits may help ensure that even the hesitant minority feels supported to seek care.  

     """)


  
  
  
    st.title("Classification Task")
    st.markdown("""### Classification Task

       In this part of the project, the goal was to predict whether someone will seek mental health treatment based on their survey responses.  
     The outcome we’re trying to predict is simple — **"Yes" or "No"** for treatment.  

      I looked at different factors such as:
      - Age and gender
      - Whether they’re self-employed or working remotely
     - If they have a family history of mental health issues
     - How much work interferes with their mental health
     - Company size
     - Awareness about workplace benefits

     I tried different machine learning models, and **Logistic Regression** performed the best for our dataset.  
     It gave us strong results while also being simple to interpret — we could see how each factor influenced the likelihood of seeking treatment.  
     I  split the data into training and testing sets and evaluated the model using accuracy, ROC-AUC score, precision, recall, and F1-score.  
     The results can help identify people who might avoid treatment so that awareness and support programs can reach them before problems worsen.

     """)
    
    st.divider()
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 155746.png", caption="Logistic Classifier", use_container_width=False)
    st.markdown("""- Best performer with ROC-AUC **0.8974**, indicating strong discriminatory power.  
                   - Simple and efficient, making it suitable for quick, interpretable predictions. 
     """)
    st.divider()

    
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 155813.png", caption="Random Forest Classifier", use_container_width=True)

    
    st.markdown("""
        - ROC-AUC **0.8288**, slightly lower than Logistic Regression.  
        - Captures non-linear relationships but less interpretable. 
        """)

    st.divider()
    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 155949.png", caption="XGB Classifier", use_container_width=True)
    st.markdown("""- ROC-AUC **0.8899**, close to Random Forest.  
        - Performs well but needs more tuning for optimal results.  """)
    st.divider()

    st.title("Regression Task")
    st.markdown("""### Classification Task

       In this part of the project, the goal was to predict a **numerical outcome** — the age of the survey respondents — based on their answers.  
      Instead of predicting "Yes" or "No", here the task was to estimate an exact number.

     I considered similar factors as in the classification task, such as:
     - Gender
     - Family history of mental health issues
     - Workplace size and remote work
     - Self-employment status
     - Awareness of workplace benefits

     I experimented with different machine learning models, and **Random Forest Regressor** performed slightly better than Linear Regression for our dataset.  
     Although the R² scores were low, Random Forest captured more variability and gave a lower Mean Squared Error compared to Linear Regression.  
     I split the data into training and testing sets and evaluated the models using metrics like **Mean Squared Error (MSE)** and **R² Score**.  
     The results show that predicting age from these survey factors is challenging, suggesting that age is not strongly determined by the features provided.

     """)

    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 155718.png", caption="Linear Regressor", use_container_width=False)
    st.markdown("""
     ###   - MSE **43.80** with R² **0.0367**, lowest performance among tested models.  
     - Simple baseline but struggles with capturing variability in the data.  
     """)
    st.divider()


    st.image("C:/Users/LOQ/OneDrive/Desktop/OL-25-LP-082/Images/Screenshot 2025-08-15 155843.png", caption="Random Forest Classifier", use_container_width=False)
    st.markdown("""
     ###   - MSE **41.97** with R² **0.0769**, showing limited predictive strength.  
     - Captures complex patterns better than Linear Regression in this case.
     """)
    st.divider()


elif menu =='Predict Age':
    st.header("📊 Age Prediction")
    st.subheader(".")

    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                  ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    remote_work = st.selectbox('Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
    benefits = st.selectbox('Does your employer provide mental health benefits?', ["Don't know", 'Yes', 'No'])
    care_options = st.selectbox('Do you know the options for mental health care your employer provides?',
                                ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox('Has your employer ever discussed mental health as part of a wellness program?',
                                    ["Don't know", 'Yes', 'No'])
    seek_help = st.selectbox('Does your employer provide resources to learn about mental health and seeking help?',
                             ['Yes', 'No'])
    leave = st.selectbox('How easy is it for you to take mental health leave?',
                         ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                             ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                             ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                           [ 'No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])
    
   
    input_df = pd.DataFrame([{
    'Gender': gender,
    'self_employed': self_employed,
    'family_history': family_history,
    'treatment': treatment,
    'work_interfere': work_interfere,
    'remote_work': remote_work,
    'benefits': benefits,
    'care_options': care_options,
    'wellness_program': wellness_program,
    'seek_help': seek_help,
    'leave': leave,
    'mental_health_consequence': mental_health_consequence,
    'coworkers': coworkers,
    'mental_health_interview': mental_health_interview,
    'supervisor': supervisor
}])
   
    
    if st.button('Predict'):
        model = joblib.load('reg_model.pkl')
        predicted_age = model.predict(input_df)
        st.write(f"Predicted Age: {np.expm1(predicted_age)} years")






if menu =="Predicting Treatment Seeking":
    st.header("Treatment Prediction")
    st.subheader('Predicting whether a employee is likely to seek mental health treatment')
    st.caption("Model Used : RandomForestClassifier")

    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                  ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    remote_work = st.selectbox('Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
    benefits = st.selectbox('Does your employer provide mental health benefits?', ["Don't know", 'Yes', 'No'])
    care_options = st.selectbox('Do you know the options for mental health care your employer provides?',
                                ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox('Has your employer ever discussed mental health as part of a wellness program?',
                                    ["Don't know", 'Yes', 'No'])
    seek_help = st.selectbox('Does your employer provide resources to learn about mental health and seeking help?',
                             ['Yes', 'No'])
    leave = st.selectbox('How easy is it for you to take mental health leave?',
                         ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                             ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                             ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                           [ 'No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])
    
    input_df = pd.DataFrame([{
    'Gender': gender,
    'self_employed': self_employed,
    'family_history': family_history,
    'treatment': treatment,
    'work_interfere': work_interfere,
    'remote_work': remote_work,
    'benefits': benefits,
    'care_options': care_options,
    'wellness_program': wellness_program,
    'seek_help': seek_help,
    'leave': leave,
    'mental_health_consequence': mental_health_consequence,
    'coworkers': coworkers,
    'mental_health_interview': mental_health_interview,
    'supervisor': supervisor
    }])

    
    if st.button('Predict'):
        clf = joblib.load('clf_model.pkl')
        predicted_treatment = clf.predict(input_df)
        if predicted_treatment == 1 :
            st.write('Yes')
        else :
            st.write("No")






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



