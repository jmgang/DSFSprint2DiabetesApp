import os
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv, find_dotenv

from ExtendedConversationBufferMemory import ExtendedConversationMemory

_ = load_dotenv(find_dotenv()) # read local .env file


def chat_with_specialist(query: str, patient_information: List[str], collected_patient_information: List[str]):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-4'
    )

    template = """The following is a friendly conversation between a human patient and an AI diabetes predictor specialist. The AI is caring towards a patient. If the AI does not know the answer to a question, it truthfully says it does not know.
    The specialist would continue to ask about the patient's information: {patient_information}. The AI specialist would also store the information 
    given by the patient as collected_patient_information. Some of the information asked are questions only answerable by yes or no. The 
    collected information so far are {collected_information}. Do not ask anymore if it has been answered in the collected information. 
    Don't stop asking until all the information have been collected. 
    
    Current conversation:
    {history}
    Human: {input}
    AI Specialist: 
    {format_instructions}
    """

    patient_information_schema = ResponseSchema(name="remaining_patient_information",
                                                description="This is the remaining patient information that is needed to be asked.")

    collected_patient_information_schema = ResponseSchema(name="collected_patient_information", type="JSON object (you may shorten the key)",
                                                          description="The collected patient information. This includes information computed by AI that is needed. (i.e. BMI from Height/Weight)")

    doctor_response_schema = ResponseSchema(name="AI response", description="AI Specialist current response")

    can_stop_asking_schema = ResponseSchema(name='can_stop_asking', type="Boolean", description="Whether the AI can stop asking or not.")

    response_schemas = [patient_information_schema, collected_patient_information_schema, doctor_response_schema, can_stop_asking_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["history", "input", "patient_information", "collected_information"],
        template=template
        , partial_variables={"format_instructions": format_instructions})

    conversation = ConversationChain(llm=llm, prompt=prompt, verbose=True,
                                     memory=ExtendedConversationMemory(llm=llm, ai_prefix="AI Specialist", k=2,
                                                                              extra_variables=["patient_information", "collected_information"])
                                     )

    convo = conversation({'input': query,
                         'patient_information': patient_information,
                          'collected_information': collected_patient_information}
                        )
    response_as_dict = output_parser.parse(convo['response'])

    return convo, response_as_dict

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

# if __name__ == "__main__":
#     chat("", llm_model="gpt-4")
