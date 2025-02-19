from crewai import Crew, Process
from medical_agents import medical_research_agent, medical_suggestion_agent 
from medical_tasks import medical_research_task, medical_suggestion_task
from langchain_openai import ChatOpenAI
import os

# def LLM(prompt):
#     return openai.Completion.create(
#         model="gpt-4o",
#         temperature=0.7,
#         max_tokens=1000,
#         api_key="OPENAI_API_KEY",
#         prompt=prompt
#     )

manager_llm = ChatOpenAI(open_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, model = 'gpt-4o')

medical_agent_crew = Crew(
    agents = [medical_research_agent,
              medical_suggestion_agent],
    tasks = [medical_research_task,
             medical_suggestion_task],

    manager_llm=manager_llm,
    process=Process.hierarchical,
    verbose=True
)

medical_agents_input = {
    'symptom1':'cough',
    'symptom2':'high body temperature',
    'diet':'veg',
    'age':'50',
    'height':'6',
    'weight':'70',
    'diabetic':'no',
}

result = medical_agent_crew.kickoff(inputs=medical_agents_input)
print(result)
