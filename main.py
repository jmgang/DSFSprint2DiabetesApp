from typing import Set
import ast
from core import chat_with_specialist
import streamlit as st
from streamlit_chat import message


st.header("Tim's Team's DiabetesGPT")
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["collected_information"] = []


prompt = st.text_input("Talk with our AI Diabetes specialist", placeholder="Enter your message here...") or st.button(
    "Submit"
)

all_needed_information = ['Age', 'Sex', 'BMI', 'Have you smoked at least 100 cigarettes in your entire life?',
                           'Do you have high bp', 'Do you have high chol',
                          'Have you had a stroke?', 'Have you had a heart disease?',
                          'Have you had phys activity in the past 30 days?', 'Have you consumed fruits 1 to more times a day?',
                          'Have you had a chol check in the last 5 yrs?', 'Have you consumed veggies 1 to more times a day?',
                          'How many alcoholic drinks have you consumed per week? (adult men >=14, women >= 7)',
                          'Have any kind of health care coverage, including health insurance?',
                          'Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?',
                          'Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor',
                          'physical illness or injury days in past 30 days scale 1-30',
                          'days of poor mental health scale 1-30 days',
                          'Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes',
                            'What is your Education level',
                          'What is your annual income in USD'
                          ]

remaining_patient_information = all_needed_information
first = True


def get_keys_from_list(dict_list_str):
    if not dict_list_str:
        return []

    keys_list = []

    # Convert each string to a dictionary and extract the keys
    for dict_str in dict_list_str:
        dictionary = ast.literal_eval(dict_str)
        keys = list(dictionary.keys())
        keys_list.append(keys)

    return keys_list


if prompt:
    with st.spinner("Generating response..."):

        collected_information = st.session_state['collected_information']

        print(f'Collected information before prompt: {collected_information}')

        generated_response, parsed_response_dict = chat_with_specialist(query=prompt, patient_information=all_needed_information,
                                                                        collected_patient_information=collected_information)

        # print(f'Generated response: {generated_response}')
        print(f'Parsed response dict: {parsed_response_dict}')

        st.session_state.chat_history.append((prompt, parsed_response_dict))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(parsed_response_dict['AI response'])
        st.session_state.collected_information = str(parsed_response_dict['collected_patient_information'])

        first = False


        print(f'Collected information: {st.session_state["collected_information"]}')


message('Hi! I am DiabetesGPT. Let me get your patient information so that I can predict if you are prediabetic or not.')

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(
            user_query, is_user=True
        )
        message(generated_response)


