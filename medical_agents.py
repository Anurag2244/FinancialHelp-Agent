from crewai import Agent
from dotenv import load_dotenv
from medical_tool import scrape_tool, search_tool
from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate, FewShotPromptTemplate
# from langchain.chains import LLMChain
# import openai

load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'

# def LLM(prompt):
#     return openai.Completion.create(
#         model="gpt-4o",
#         temperature=0.7,
#         max_tokens=1000,
#         api_key="OPENAI_API_KEY",
#         prompt=prompt
#     )

llm = ChatOpenAI(open_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, model = 'gpt-4o')

medical_research_agent = Agent(
    role = "Medical Assistant",
    goal = "fetches data from documents, web searches about symptoms"
           "suggest remedies",
    backstory = "Specializing in medical knowledge retrieval, this agent"
                "It have the capacity to identify the symptoms"
                "seamlessly integrates document analysis, web searches, and symptom evaluation to deliver precise health insights"
                "It provides appropiate data and crucial insights. with a knack for data"
                "uses statiscal model and machine learning"
                "Gives suggestion according to the health conditions",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

medical_suggestion_agent = Agent(
    role = "Nutrition and Medical suggestion",
    goal = "Develop and test various medical suggestion based "
           "on insights from the Medical Assistant Agent.",
    backstory = "Equipped with a deep understanding of medical field "
                "It gives the suggestion according to the symptoms by predicting the disease"
                "medical knowledge and suggested various knowledge, this agent "
                "continously checks about medical industry. It evaluates "
                "tries to get report from different symptoms"
                "the most best way to get suggestion and regularly stay update for new problems",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)
