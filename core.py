import os
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv, find_dotenv

from ExtendedConversationBufferMemory import  ExtendedConversationSummaryMemory

_ = load_dotenv(find_dotenv()) # read local .env file


def chat(query: str, llm_model="gpt-3.5-turbo"):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model=llm_model
    )

    template = """The following is a friendly conversation between a human patient and an AI doctor. The AI is friendly and caring towards a patient. If the AI does not know the answer to a question, it truthfully says it does not know.
    The doctor would continue to ask about the patient's information {all_needed_information}.
    The remaining patient information that is needed to be asked are {remaining_patient_information}.
    Format the output as a JSON wherein the response key is your response, and the remaining_patient_information key to store the remaining patient 
    information that you needed to ask from the patient. 
    
    Current conversation:
    {history}
    AI Doctor: Let me get your patient information so that I can diagnose if you are prediabetes or not.
    Human: {input}
    AI Doctor: 
    {format_instructions}
    """

    patient_information_schema = ResponseSchema(name="remaining_patient_information",
                                                description="This is the remaining patient information that is needed to be asked.")



    response_schemas = [patient_information_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["history", "input", "all_needed_information", "remaining_patient_information"],
        template=template
        , partial_variables={"format_instructions": format_instructions})
    # now initialize the conversation chain
    conversation = ConversationChain(llm=llm, prompt=prompt, verbose=True,
                                     memory=ExtendedConversationSummaryMemory(llm=llm, ai_prefix="AI Doctor",
                                                                              extra_variables=["all_needed_information", "remaining_patient_information"]))

    convo = conversation({'input' : 'Hi Doctor, I am a 23 year old male. I want to know if i have diabetes or not.',
                          'remaining_patient_information' : 'BMI, Sex, Age, Has a history of Heart disease',
                          'all_needed_information' : 'BMI, Sex, Age, Has a history of Heart disease'})
    response_as_dict = output_parser.parse(convo['response'])
    print(convo['response'])

    remaining_needed_information = response_as_dict['remaining_patient_information']

    convo = conversation({'input': 'I have a height of 160cm and weight of 85kg',
                          'remaining_patient_information': ', '.join(remaining_needed_information),
                          'all_needed_information': 'BMI, Sex, Age, Has a history of Heart disease'})

    print(convo['response'])

def run_llm(query: str, llm_model="gpt-3.5-turbo"):

    llm = ChatOpenAI(temperature=0.1, model=llm_model, openai_api_key=os.environ['OPENAI_API_KEY'])

    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template(
        "You are a medical receptionist and you want to take a patient's information. "
        "You already got the patient's {got_information} but you still need {more_information}. "
        "Ask a question to the patient to provide the needed {more_information} information to proceed to a diagnostic. If one of the {more_information} can be computed from the patient's other information then automatically compute that. (For example if BMI is needed but the patient can only provide height and weight). Take note that the patient already knows that you are not a doctor nor a medical professional so you need not remind him. "
    )
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                         output_key="English_Review"
                         )

    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:"
        "\n\n{English_Review}"
    )
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm=llm, prompt=second_prompt,
                         output_key="summary"
                         )

    # prompt template 3: translate to english
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                           output_key="language"
                           )

    # prompt template 4: follow up message
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                          output_key="followup_message"
                          )

    # overall_chain: input= Review
    # and output= English_Review,summary, followup_message
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three, chain_four],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True
    )

    result = overall_chain('This is a five star hotel!')

    print(result)

if __name__ == "__main__":
    chat("", llm_model="gpt-4")
