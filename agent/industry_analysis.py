from typing import Any, Dict, Annotated
import operator

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

import logging
import sys,os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
### TO DO
# Only load weave if it is Dev
# import weave

LOG_FILE = 'Fusion-industry_analysis.log'

load_dotenv("./agent/.env", override=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FusionIndustryAnalysis")

class StateType(Dict[str, Any]):
    context: Annotated[Dict[str, Any], operator.or_]

LANGSMITH_API_KEY=os.environ.get("LANGSMITH_API_KEY")
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

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

#@weave.op()
def industry_analysis_economic(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"].get("configuration")
    logging.info(f"Executing industry_analysis_economic with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_economic")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_economic: {result}")
    
    return {"context": {"industry_analysis_economic": result}}

#@weave.op()
def industry_analysis_market(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing industry_analysis_market with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_market")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_market: {result}")
    
    return {"context": {"industry_analysis_market": result}}

#@weave.op()
def industry_analysis_consumer(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing industry_analysis_consumer with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_consumer")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_consumer: {result}")
    
    return {"context": {"industry_analysis_consumer": result}}

#@weave.op()
def industry_analysis_external(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing industry_analysis_external with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_external")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_external: {result}")
    
    return {"context": {"industry_analysis_external": result}}   

#@weave.op()
def industry_analysis_brand(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing industry_analysis_brand with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_brand")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_brand: {result}")
    
    return {"context": {"industry_analysis_brand": result}}

#@weave.op()
def industry_analysis_supplychain(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing industry_analysis_supplychain with configuration: {config}")
    prompt = pullprompt("fusion_industry_analysis_supplychain")
    result = execute_prompt(prompt, state)
    logging.info(f"Finished industry_analysis_supplychain: {result}")
    
    return {"context": {"industry_analysis_supplychain": result}}

def coalesce(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing coalesce with configuration: {config}")
    return state

def create_graph() -> StateGraph:
    """Create the langgraph DAG for parallel industry analysis."""
    
    logger.info("Creating industry analysis graph...")

    # Create the workflow graph with state type annotation
    workflow = StateGraph(StateType)

    workflow.add_node("economic", industry_analysis_economic)
    workflow.add_node("market", industry_analysis_market)
    workflow.add_node("consumer", industry_analysis_consumer)
    workflow.add_node("external", industry_analysis_external)
    workflow.add_node("brand", industry_analysis_brand)
    workflow.add_node("supplychain", industry_analysis_supplychain)
    workflow.add_node("coalesce", coalesce)

    for node in ["economic", "market", "consumer", "external", "brand", "supplychain"]:
        workflow.add_edge(START, node)
    
    for node in ["economic", "market", "consumer", "external", "brand", "supplychain"]:
        workflow.add_edge(node, "coalesce")
    
    workflow.add_edge("coalesce", END)
    
    logger.info("Industry analysis graph created successfully.")
    return workflow.compile()

graph = create_graph()
graph.name = "FusionIndustryAnalysisGraph" 
#weave.init('c4-industry-analysis')