from langgraph.graph import START, END, StateGraph
import operator 
import sys,os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated, Sequence, Dict, Any
import logging

LOG_FILE = 'FusionPromptExecutor.log'

load_dotenv("./agent/.env", override=True)
from  agent.environments import load_environment
load_environment()

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FusionPromptExecutor")

LANGSMITH_API_KEY=os.environ.get("LANGSMITH_API_KEY")
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# class StateType(TypedDict):
#     messages: Annotated[Sequence[str], operator.add]

class StateType(Dict[str, Any]):
    context: Annotated[Dict[str, Any], operator.or_]

def pullprompt(promptname: str) -> ChatPromptTemplate:
    try:
        prompt = langsmith_client.pull_prompt(promptname,include_model=True)
    except Exception as e:
        logging.error(f"Failed to get prompt: {e}")
        raise
    return prompt

def execute_prompt(prompt: ChatPromptTemplate, state: StateType) -> StateType:
    try:
        print(state)
        result = prompt.invoke(state["context"])
    except Exception as e:  
        logging.error(f"Failed to execute prompt: {e}")
        raise
    return result

def execute_prompt_node(state: StateType, config: RunnableConfig) -> StateType:
    """Each node does work."""
    #configuration = config["configurable"].get("configuration")
    logging.info(f"Executing execute_prompt_node with configuration: {config}")
    print("prompt_executor_state",state)
    promptnm = state["context"]["prompt_name"]
    prompt = pullprompt(promptnm)
    result = execute_prompt(prompt, state)
    logging.info(f"Finished execute_prompt_node: {result}")
    
    return {"context": {promptnm: result}}

def create_graph() -> StateGraph:
    """Create the langgraph DAG for parallel industry analysis."""
    
    logger.info("Creating prompt executor graph...")

    # Create the workflow graph with state type annotation
    workflow = StateGraph(StateType)

    workflow.add_node("execute_prompt_node", execute_prompt_node)
    
    workflow.add_edge(START, "execute_prompt_node")
    workflow.add_edge("execute_prompt_node", END)
    
    logger.info("Prompt executor graph created successfully.")
    return workflow.compile()

graph = create_graph()
graph.name = "FusionPromptExecutorGraph" 
