"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a simple generation or Q&A use case without comprehensive prompt tuning
"""

# Install the wml and streamlit api your Python env prior to running this example:
# pip install ibm-watson-machine-learning
# pip install streamlit

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import streamlit as st

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Load environment variables from .env file
load_dotenv()

# Mendapatkan nilai API_KEY dari environment variables
api_key = os.getenv("API_KEY")
# Mendapatkan nilai IBM_CLOUD_URL dari environment variables
url = os.getenv("IBM_CLOUD_URL")
# Mendapatkan nilai PROJECT_ID dari environment variables
watsonx_project_id = os.getenv("PROJECT_ID")

if not all([api_key, url, watsonx_project_id]):
    raise ValueError("Please make sure all environment variables are set: API_KEY, IBM_CLOUD_URL, PROJECT_ID")

def get_credentials():
    # Ensure the environment variables are loaded
    load_dotenv()
    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("API_KEY")
    globals()["watsonx_project_id"] = os.getenv("PROJECT_ID")

def get_model(model_type, max_tokens, min_tokens, decoding, stop_sequences):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model

def get_prompt(question):
    instruction = "Answer this question briefly."
    examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    your_prompt = question
    end_prompt = "Answer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt

def answer_questions():
    get_credentials()

    st.title('🌠 Test watsonx.ai LLM')
    user_question = st.text_input('Ask a question, for example: What is IBM?')

    if len(user_question.strip()) == 0:
        user_question = "What is IBM?"

    final_prompt = get_prompt(user_question)
    st.text_area("Generated Prompt", final_prompt)

    model_type = ModelTypes.FLAN_UL2
    max_tokens = 100
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.']

    model = get_model(model_type, max_tokens, min_tokens, decoding, stop_sequences)

    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']

    st.markdown(f"**Answer to your question:** {user_question}")
    st.markdown(f"*{model_output}*")

if __name__ == "__main__":
    answer_questions()
