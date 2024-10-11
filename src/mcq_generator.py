import openai
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI


# Load the .env file
load_dotenv()


# Manually set the environment variable from .env, prioritizing it over system-wide variables
openai.api_key = os.getenv('OPENAI_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize your LLM client
llm = OpenAI()  # Replace with your actual API key
prompt = PromptTemplate.from_template("What is a good name for a company that manufactures {product}")


# Chain to generate a restaurant name
name_chain = LLMChain(llm=llm, prompt=prompt)
print(name_chain.run("advanced military helicopter").strip())


template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "grade", "tone", "response_json"],
    template=template)



quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

template="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)