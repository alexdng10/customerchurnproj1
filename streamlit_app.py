# Modified
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv  # Make sure this is present
import utils as utilities  # Changed alias to make it more descriptive
load_dotenv()
# Initialize OpenAI client with the environment variable
openai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv('GROQ_API_KEY')  # Using os.getenv for consistency in variable retrieval
)

# Modified: Streamlined model loading
def load_trained_model(filepath):
    with open(filepath, "rb") as model_file:
        return pickle.load(model_file)
# Load all models using the modified load_trained_model function
xgboost_model = load_trained_model('xgb_model.pkl')
native_bayes_model = load_trained_model('nb_model.pkl')
random_forest_model = load_trained_model('rf_model.pkl')
decision_tree_model = load_trained_model('dt_model.pkl')
svm_model = load_trained_model('SVM_model.pkl')
knn_model = load_trained_model('knn_model.pkl')
voting_classifier_model = load_trained_model('voting_clf.pkl')
xgboost_SMOTE_model = load_trained_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_trained_model('xgboost-featureEngineered.pkl')


def generate_input_dataframe(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_features = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),   # Use HasCrCard
        'IsActiveMember': int(is_active_member),  # Use IsActiveMember
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }

    # Returns DataFrame and dictionary for consistency
    return pd.DataFrame([input_features]), input_features




def compute_and_display_churn_risk(input_dataframe, input_details):
    # Filter DataFrame columns required for prediction
    input_dataframe = input_dataframe[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
                                       'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 
                                       'Geography_Spain', 'Gender_Female', 'Gender_Male']]

    # Generate model probabilities
    model_predictions = {
        'XGBoost': xgboost_model.predict_proba(input_dataframe)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_dataframe)[0][1],
        'KNN': knn_model.predict_proba(input_dataframe)[0][1]
    }

    # Calculate average churn risk probability
    average_churn_risk = np.mean(list(model_predictions.values()))

    col_left, col_right = st.columns(2)

    with col_left:
        gauge_chart = utilities.create_gauge_chart(average_churn_risk)
        st.plotly_chart(gauge_chart, use_container_width=True, config={'displayModeBar': False})
        st.write(f"Customer has a {average_churn_risk:.2%} chance of churning.")

    with col_right:
        probability_chart = utilities.create_model_probability_chart(model_predictions)
        st.plotly_chart(probability_chart, use_container_width=True, config={'displayModeBar': False})

    return average_churn_risk


def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank. You specialize in interpreting and explaining predictions of machine learning models, particularly customer churn predictions. 
    Your task is to explain the following prediction to a non-technical customer service representative.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning based on the information provided below.

    Please provide:
    1. **An overview** of the prediction, including why the customer might churn.
    2. **A detailed breakdown** of the key features (from the model's top 10 most important features) that most contribute to this prediction.
    3. **Actionable advice** for the customer service representative on how to reduce the churn risk based on the provided features.

    Here's the customer's information:
    {input_dict}

    Here are the top 10 most important features from the machine learning model, ranked by importance:

    Feature | Importance
    ----------------------
    NumOfProducts    | 0.323888
    IsActiveMember   | 0.164146
    Age              | 0.109559
    Geography_Germany| 0.095376
    Balance          | 0.057875
    Geography_France | 0.052467
    Gender_Female    | 0.049893
    EstimatedSalary  | 0.031940
    HasCrCard        | 0.030954
    Tenure           | 0.030054
    Gender_Male      | 0.000000

    Based on this, generate a detailed 3-sentence explanation of why {surname} might churn, as well as suggestions to reduce churn risk.
    """

    raw_response = openai_client.chat.completions.create(
        model="llama-3.2-3b-preview",  
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content



def compose_customer_email(churn_risk, customer_data, risk_factors, last_name):
    prompt_text = f"""
    As a relationship manager at Gucci Bank, your responsibility is to ensure customer satisfaction and loyalty. Compose a professional, personalized email for our valued customer, {last_name}, offering services to address their needs and encourage continued loyalty.
    My name is Hung Dang
    Analysis indicates that {last_name} may have a {round(churn_risk * 100, 1)}% chance of churning based on recent insights. To reinforce our commitment, express our appreciation for their loyalty and propose personalized incentives to enhance their experience.

    Customer Details:
    {customer_data}

    Risk Factors Identified for {last_name}:
    {risk_factors}

    Email Guidelines:
    1. **Express Appreciation**: Recognize their loyalty without becoming overly personal, acknowledging their ongoing relationship with PAK Bank.
    2. **Offer Personalized Incentives**: Propose relevant benefits, such as exclusive rates, tailored promotions, or enhanced services based on their profile.
    3. **Encourage Dialogue**: Suggest a follow-up conversation to better understand their needs, emphasizing PAK Bank’s commitment to supporting their financial goals.
    4. **Maintain a Professional Tone**: Balance the message with professionalism, avoiding any overly personal or emotional language.

    Suggested Incentives to Include:
    - Exclusive interest rates on savings or loan products.
    - Enhanced rewards programs and banking benefits.
    - Personalized support for managing their financial objectives.

    Avoid any mention of predictive analysis or machine learning models. Focus on the tangible benefits of staying with PAK Bank, offering clear and actionable incentives to retain their business.
    """

    # Generating the email content using the OpenAI client
    response = openai_client.chat.completions.create(
        model="llama-3.1-8b-instant",  
        messages=[{
            "role": "user",
            "content": prompt_text
        }],
    )

    print("\n\nGenerated Email Prompt:", prompt_text)  # To review the prompt in logs if needed

    return response.choices[0].message.content






st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])

selected_surname = selected_customer_option.split(" - ")[1]

selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=int(selected_customer['CreditScore'])
    )

    location = st.selectbox(
        "Location", ["Spain", "France", "Germany"],
        index=["Spain", "France", "Germany"].index(selected_customer['Geography'])
    )

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer['Gender'] == 'Male' else 1)

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=int(selected_customer['Age'])
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=50,
        value=int(selected_customer['Tenure'])
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=float(selected_customer['Balance'])
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=int(selected_customer['NumOfProducts'])
    )

    has_credit_card = st.checkbox(
        "Has Credit Card",
        value=bool(selected_customer['HasCrCard'])
    )

    is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer['IsActiveMember'])
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary'])
    )


input_df, input_dict = generate_input_dataframe(credit_score, location, gender, age, tenure, balance,
                                     num_products, has_credit_card, is_active_member, estimated_salary)

avg_probability =  compute_and_display_churn_risk(input_df, input_dict)


explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

st.markdown("---")
st.subheader("Explanation of Prediction")
st.markdown(explanation)


email = compose_customer_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

st.markdown("---")

st.subheader("Personalized Email")

st.markdown(email)
