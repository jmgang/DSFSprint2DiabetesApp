from typing import Set
import ast
import json
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from core import chat_with_specialist, run_llm
import streamlit as st
from streamlit_chat import message


model = pickle.load(open('models/logistic_smoteenn_top10feats_final.pkl', 'rb'))
scaler : StandardScaler = pickle.load(open('models/logistic_smoteenn_top10feats_final_scaler.pkl', 'rb'))

st.set_page_config(page_title="Tim's Team's Diabetes Predictor")
st.set_option('deprecation.showfileUploaderEncoding', False)
title = '<h2 style="font-family:Arial; background: #1abc9c; color: white; padding: 20px; text-align: center; width: 100%">Tim\'s Team\'s Diabetes Predictor</h2>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "<span style='color:red'>NOTE: This is only an MVP that aims to predict if a person is prediabetic or not based on a trained model. "
    "Do NOT use the results as medical advice and please consult a medical professional if you have any symptoms. </span>",
    unsafe_allow_html=True)

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["collected_information"] = []


# all_needed_information = ['Age', 'Sex', 'BMI', 'Have you smoked at least 100 cigarettes in your entire life?',
#                            'Do you have high bp', 'Do you have high chol',
#                           'Have you had a stroke?', 'Have you had a heart disease?',
#                           'Have you had phys activity in the past 30 days?', 'Have you consumed fruits 1 to more times a day?',
#                           'Have you had a chol check in the last 5 yrs?', 'Have you consumed veggies 1 to more times a day?',
#                           'How many alcoholic drinks have you consumed per week? (adult men >=14, women >= 7)',
#                           'Have any kind of health care coverage, including health insurance?',
#                           'Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?',
#                           'Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor',
#                           'physical illness or injury days in past 30 days scale 1-30',
#                           'days of poor mental health scale 1-30 days',
#                           'Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes',
#                             'What is your Education level',
#                           'What is your annual income in USD'
#                           ]

all_needed_information = ['Age', 'Sex', 'BMI',
                           'Do you have high bp', 'Do you have high chol',
                          'Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor',
                          'For how many days do you have physical illness or injury in past 30 days, scale 1-30',
                          'For how many days do you have poor mental health (i.e. depression), scale 1-30 days',
                            'From a scale of 1-8, give me the range of your annual income in USD (in thousands). 1.0: <10k, 2.0: 10k-15k, 3.0: 15k-20k, 4.0: 20k-25k, 5.0: 25k-35k, 6.0: 35k-50k, 7.0: 50k-75k, 8.0: >75k',
                            'How many alcoholic drinks have you consumed per week?',
                          ]

remaining_patient_information = all_needed_information
question_with_code = {'Do you have high bp' : 'HighBP',
                      'Do you have high chol' : 'HighChol',
                      'Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor' : 'GenHlth',
                      'For how many days do you have physical illness or injury in past 30 days, scale 1-30' : 'PhysHlth',
                      'For how many days do you have poor mental health (i.e. depression), scale 1-30 days' : 'MentHlth',
                      'From a scale of 1-8, give me the range of your annual income in USD (in thousands). 1.0: <10k, 2.0: 10k-15k, 3.0: 15k-20k, 4.0: 20k-25k, 5.0: 25k-35k, 6.0: 35k-50k, 7.0: 50k-75k, 8.0: >75k' : 'Income',
                      'How many alcoholic drinks have you consumed per week?' : 'HvyAlcoholConsump'
                      }

def predict(df):
    scaled_unseen_data = scaler.transform(df)

    prediction = model.predict(scaled_unseen_data)

    print(prediction)

    return 'Not diabetic' if prediction[0] == 0 else 'Prediabetic/Diabetic'


def give_predict_result(collected_information, prediction):
    prompt = """
        You are a diabetes specialist with about 15 years of experience. A machine learning model that is trained on diabetes data has just predicted that based on
        the collected information of {collected_information}, The predicted result is that the patient is {prediction}. 
        Your job is to  Summarize all the collected information from the patient and tell the person that he/she is prediabetic or not. But also please tell him/her
        to consult a real medical professional since this was only based on a trained machine learning model. 
        """

    result = run_llm(prompt, ['collected_information', 'prediction'], {
        'collected_information': collected_information,
        'prediction': prediction
    })

    print(result)

    return result


def map_and_predict(collected_information, question_with_code):
    prompt = """From the collected JSON patient information: {collected_information}, create a python dictionary matching the information key to the code as shown by {question_with_code}. 
    The key would be the dictionary's keys while the answer would be the dictionary's value. If you can't find a code, then set the key as is with the JSON key. 
    Set yes/no answers to 1/0 integers respectively. 
    For the age, bin it to the following: 1 Age 18 to 24 , 2 Age 25 to 29 , 3 Age 30 to 34 , 4 Age 35 to 39 , 5 Age 40 to 44 , 6 Age 45 to 49 , 7 Age 50 to 54 , 8 Age 55 to 59 , 
    9 Age 60 to 64 , 10 Age 65 to 69 , 11 Age 70 to 74 , 12 Age 75 to 79 , 13 Age 80 or older.
    For the alcoholic drinks, if adult men >=14 or women >= 7 then indicate 1 otherwise 0.
    For the sex, set 0 = female and 1 = male.
    For the income, if the answer is not a single digit, bin it to the following: 1.0: <10k, 2.0: 10k-15k, 3.0: 15k-20k, 4.0: 20k-25k, 5.0: 25k-35k, 6.0: 35k-50k, 7.0: 50k-75k, 8.0: >75k.
    Set the order of columns as follows: ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth', 'HighBP', 'Income', 'HighChol', 'Sex', 'HvyAlcoholConsump']
    Just return one line, the pandas dictionary as a string. No more no less. Don't give any programming line. Just return ONE LINE.
    """

    result = run_llm(prompt, ['collected_information', 'question_with_code'], {
        'collected_information' : collected_information,
        'question_with_code' : question_with_code
    })

    print(result)

    evaluated_result = json.loads(result.strip('"').replace("'", '"'))
    print(type(evaluated_result))

    df = pd.DataFrame([evaluated_result])
    print(df)

    return predict(df)


def can_ai_stop_asking(collected_information, all_information):
    prompt = """
        From the set of collected information here: {collected_information} that is in JSON, detect whether all the information have been gathered or answered based on
        this: {all_information}. If all the information has been sufficiently provided, answer yes. otherwise answer no. If you see an empty json or empty list, then
        answer no. Do not use programming, just answer either yes or no. No more no less.
        """

    result = run_llm(prompt, ['collected_information', 'all_information'], {
        'collected_information': collected_information,
        'all_information': all_information
    })

    print(result)

def plot_pie(df, len_preds):
    st.markdown('<p style="font-family:Arial; background: #1abc9c; font-size: 20px; color: white; '
                'text-align: center;">Prediction of ' + str(len_preds) + ' patients pie chart</p>',
                unsafe_allow_html=True)
    fig = px.pie(data_frame=df, values='Count', names='Predicted',
                 color_discrete_sequence=["green", "red", "yellow", "orange", "blue"], width=800, height=700)
    st.write(fig)

def run_chat():
    st.write(
        "NOTE: Only the top 10 most important features (according to SHAP) are trained with this model for brevity.")

    prompt = st.text_input("Talk with our AI Diabetes specialist",
                           placeholder="Enter your message here...") or st.button(
        "Submit"
    )

    if prompt:
        with st.spinner("Generating response..."):

            collected_information = st.session_state['collected_information']

            print(f'Collected information before prompt: {collected_information}')

            generated_response, parsed_response_dict = chat_with_specialist(query=prompt, patient_information=all_needed_information,
                                                                            collected_patient_information=collected_information)

            # print(f'Generated response: {generated_response}')
            print(f'Parsed response dict: {parsed_response_dict}')

            can_stop_asking = parsed_response_dict['can_stop_asking']

            st.session_state.collected_information = str(parsed_response_dict['collected_patient_information'])
            print(f'Collected information: {st.session_state["collected_information"]}')

           # stop_asking = can_ai_stop_asking(st.session_state["collected_information"], all_needed_information)
            if can_stop_asking :
                prediction = map_and_predict(st.session_state["collected_information"], question_with_code)
                response = give_predict_result(st.session_state["collected_information"], prediction)

                st.session_state.chat_history.append((prompt, response))
                st.session_state.user_prompt_history.append(prompt)
                st.session_state.chat_answers_history.append(response)

                st.write('How we came up with your prediction...')

            else:
                st.session_state.chat_history.append((prompt, parsed_response_dict))
                st.session_state.user_prompt_history.append(prompt)
                st.session_state.chat_answers_history.append(parsed_response_dict['AI response'])

    message("Hi! I am DiabetesGPT. Let me get your patient information so that I can predict if you are prediabetic or not. "
            "Please answer the following questions:\n1. What is your sex?\n2. How old are you?\n3. "
            "What is your BMI? (If you don't know, you can just provide your height and weight)\n4. Do you have high blood pressure?\n"
            "5. Do you have high cholesterol?\n6. On a scale of 1-5, how would you rate your general health? (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)"
            "\n7. For how many days have you had physical illness or injury in the past 30 days? (Scale 1-30 days)\n8. For how many days have you had poor mental health (i.e., depression) in the past 30 days? (Scale 1-30 days)\n9. From a scale of 1-8, what is the range of your annual income in USD (in thousands)? (1.0: <10k, 2.0: 10k-15k, 3.0: 15k-20k, 4.0: 20k-25k, 5.0: 25k-35k, 6.0: 35k-50k, 7.0: 50k-75k, 8.0: >75k)\n10. How many alcoholic drinks have you consumed per week?")

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(
                user_query, is_user=True
            )
            message(generated_response)

# Create sidebar with buttons
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Choose a page", ["DiabetesGPT", "Data Upload"])

# Display content based on selection
if page == "DiabetesGPT":
    st.header("DiabetesGPT")
    run_chat()

elif page == "Data Upload":
    st.header("Data Upload")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        model = pickle.load(open('models/logistic_smoteenn_allfeats_final.pkl', 'rb'))
        scaler: StandardScaler = pickle.load(open('models/logistic_smoteenn_allfeats_final_scaler.pkl', 'rb'))

        demo_dataset = pd.read_csv(uploaded_file)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:Arial; background: #1abc9c; font-size: 20px; color: white; text-align: center; width: 100%">Given Data</p>',
            unsafe_allow_html=True)
        st.dataframe(demo_dataset)
        st.markdown('<hr><br>', unsafe_allow_html=True)

        X_scaled = scaler.transform(demo_dataset)
        y_real_predict = model.predict(X_scaled)

        predicted_mappings = []
        for prediction in y_real_predict:
            predicted_mappings.append('Not diabetic' if prediction else 'Prediabetic/Diabetic')

        demo_dataset["Predicted"] = predicted_mappings

        st.markdown(
            '<p style="font-family:Arial; background: #1abc9c; font-size: 20px; color: white; text-align: center;">'
            'Table of patients with prediction</p>',
            unsafe_allow_html=True)
        st.dataframe(demo_dataset)
        st.markdown('<hr><br>', unsafe_allow_html=True)

        array = np.array(predicted_mappings)
        unique, counts = np.unique(array, return_counts=True)
        result = pd.DataFrame(np.column_stack((unique, counts)), columns=['Predicted', 'Count'])

        plot_pie(result, len(demo_dataset))





