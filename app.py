# Rename `os.environ` to `env` for nicer code
from os import environ as env
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv,find_dotenv
import markdown
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv('config/.env')

# Rename `os.environ` to `env` for nicer code

print('GOOGLE_API_KEY:  {}'.format(env['GOOGLE_API_KEY']))

GEMINI_API_KEY = env['GOOGLE_API_KEY']

env["GEMINI_API_KEY"] = env['GOOGLE_API_KEY']

llm = LLM(
    model="gemini/gemini-1.5-pro-latest",
    temperature=0.7
)

##
# Other Popular Models as LLM for your Agents
#
#
##

## Hugging Face (HuggingFaceHub endpoint)

#from langchain_community.llms import HuggingFaceHub

# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     huggingfacehub_api_token="<HF_TOKEN_HERE>",
#     task="text-generation",
# )

### you will pass "llm" to your agent function

#Mistral API

# OPENAI_API_KEY = your-mistral-api-key
# OPENAI_API_BASE = https://api.mistral.ai/v1
# OPENAI_MODEL_NAME = "mistral-small"


### Cohere

##rom langchain_community.chat_models import ChatCohere

# Initialize language model
# os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
# llm = ChatCohere()

### you will pass "llm" to your agent function


# Component	Description	                Key Features
# Crew	    The top-level organization	• Manages AI agent teams
#                                         • Oversees workflows
#                                         • Ensures collaboration
#                                         • Delivers outcomes
#
# AI Agents	Specialized team members	• Have specific roles (researcher, writer)
#                                         • Use designated tools
#                                         • Can delegate tasks
#                                         • Make autonomous decisions
#
# Process	    Workflow management system	• Defines collaboration patterns
#                                         • Controls task assignments
#                                         • Manages interactions
#                                         • Ensures efficient execution
#
# Tasks	    Individual assignments	    • Have clear objectives
#                                         • Use specific tools
#                                         • Feed into larger process
#                                         • Produce actionable results


# How It All Works Together
#
#     The Crew organizes the overall operation
#     AI Agents work on their specialized tasks
#     The Process ensures smooth collaboration
#     Tasks get completed to achieve the goal


###
#
# Creating Agents
#
#     Define your Agents, and provide them a role, goal and backstory.
#     It has been seen that LLMs perform better when they are role playing.
#
#     Description:
#        Agents are the core in CrewAi multisystem in project create
#         a Planner, Writer and Editor agent to work together in create technical articles
#
###

### Examples
## https://docs.crewai.com/examples/example

## Agent: Planner
#
# It's better  multiple strings than triple quote docstring:
#
planner = Agent(
    role = "Content Planner",
    goal = "Plan engaging and factually accurate content on {topic}",
    backstory = "You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation = False,
	verbose = True,
    llm = llm,
)

## Agent: Writer

writer = Agent(
    role = "Content Writer",
    goal = "Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory = "You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation = False,
    verbose = True,
    llm = llm,
)

## Agent: Editor

editor = Agent(
    role = "Editor",
    goal = "Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory = "You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation = False,
    verbose = True,
    llm = llm,
)


###
#  Creating Task for each Agent like a target
#  it's possible assigned multiples task to an agent
#
#  Creating Tasks
#
#     Define your Tasks, and provide them a description, expected_output and agent.
###

### Task: Plan

plan = Task(
    description = (
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output = "A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent = planner,
)

## Task: Write

write = Task(
    description = (
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output = "A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent = writer,
)


## Task: Edit

edit = Task(
    description = ("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output = "A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent = editor
)

###
#  Union in all resources (agent and task to work together with Crew resources)
#
#  Creating the Crew
#
#     Create your crew of Agents
#     Pass the tasks to be performed by those agents.
#         Note: In these project, the tasks will be performed
#         sequentially (i.e. they are dependent on each other), so the order
#         of the task in the list matters.
#
#     verbose=2 allows you to see all the logs of the execution.

###

crew = Crew(
    agents = [planner, writer, editor],
    tasks = [plan, write, edit],
    verbose = True,
    process=Process.sequential,
)

## Running the Crew

# Note: LLMs can provide different outputs for they same input,
#so what you get might be different in the project output
#
#  The topic was the variable defined in agent backstory a tasks descriptions
#

agents_topic = "Donald Trump and LATAM"

result = crew.kickoff(inputs = {"topic": agents_topic })

# Accessing the crew output
print(f"Raw Output: {result.raw}")
print(f"Tasks Output: {result.tasks_output}")
print(f"Token Usage: {result.token_usage}")
print(crew.usage_metrics)
html_string = markdown.markdown(result.raw)

with open('output.md', 'w') as f:
    f.write(html_string)


