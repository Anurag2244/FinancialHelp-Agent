from crewai import Task
from medical_tool import search_tool, scrape_tool
from medical_agents import medical_research_agent, medical_suggestion_agent

medical_research_task = Task(
    description=(
        "Continuously monitor and analyze new disease "
        "Review the symptoms ({symptom1}). "
        "Use statistical modeling and machine learning to gather information for cure of the disease "
        "identify both the ({symptom1} & {symptom2}) and predict the disease"
    ),
    expected_output=(
        "Gives the suggestion of disease and types"
        "By reviewing the {symptom1} & {symptom2} of the patient ."
    ),
    agent=medical_research_agent
)

medical_suggestion_task = Task(
    description=(
        "It re check the {symptom1} and tries to give wright output"
        "After reviewing the symptoms from input {symptom2} it tries to gives the disease"
        "After recognizing the disease suggest the daily diet"
        "Analyze user symptoms and provide possible medical conditions along with next steps."
    ),
    expected_output=(
        "Try to give more prominent output with healthy {diet} diet"
        "Gives the output how to cure this disease"
        "Gives proper diet plan according to the measurements of human body {age} years, {height} foot, {weight} kg, {diabetic}"
        "List of possible conditions and recommendations."
    ),
    agent = medical_suggestion_agent
)