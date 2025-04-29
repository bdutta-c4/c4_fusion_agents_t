from typing import Any, Dict, List, Iterable,Annotated,TypedDict,Optional
from datetime import date
from dataclasses import dataclass

import pandas as pd
from langgraph.graph import START, END, StateGraph
import operator 
import sys,os
from dotenv import load_dotenv

sys.path.append("agent")

from sql_handler import SnowflakeSQLHandler
from queries import *
from cortex import CortexConfig, SnowflakeConfig
import fetchdata_knownqueries as fetchdata


import logging

LOG_FILE = 'FusionDashboard.log'

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
logger = logging.getLogger("FusionSnowflakeDataHandler")

@dataclass
class DashboardDataJson:
    df_basic: str
    df_inv_agg: str
    df_monthly_inv: str
    df_ads_agg: str
    df_monthly_ads: str
    df_ga_agg: str
    df_monthly_ga: str
    df_mystery: str
    df_budget: str
    df_monthly_budget: str
    df_rank: str
    df_oem: str
    df_monthly_all: str
    df_budget_avg: str
    df_detailed_budget: str
    df_traffic_comparison: str
    df_missing_channels: str
    df_avg_monthly_sales: str
    df_mystery_monthly: str
    df_inventory_lb: str
    df_inventory_lb_state_brand_all: str
    df_inventory_lb_state_limit: str
    df_inventory_lb_brand_limit: str
    df_budget_lb: str
    df_budget_lb_state_brand_all: str
    df_budget_lb_state_limit: str
    df_budget_lb_brand_limit: str
    df_website_lb: str
    all_states_df: str
    all_brands_df: str

@dataclass
class StateType(Dict[str, Any]):
    context: Annotated[Dict[str, Any], operator.or_]

from langchain_core.runnables.config import RunnableConfig

class ConfigSchema(TypedDict):
    DF_TABLES: Optional[List[str]]
    DF_TABLE_GROUPS: Optional[List[str]]
    DF_NL_QUERY: Optional[str]


def get_sql_handler():
    """Dependency to get SQL handler instance."""
    try:
        HOST = os.environ['HOST']
        USER = os.environ['USER']
        SNOWFLAKE_ACCOUNT = os.environ['SNOWFLAKE_ACCOUNT']
        DATABASE = "CORTEX_ANALYST_DEMO"
        SCHEMA = "FUSION_DATA"
        STAGE = "RAW_DATA"
        FILE = "semantic_model_latest.yaml"
        WAREHOUSE = "COMPUTE_WH"
        ROLE = "CORTEX_USER_ROLE"

        if os.getenv("USE_PRIVATE_KEY_FILE"):
            PRIVATE_KEY_FILE = os.environ['PRIVATE_KEY_FILE']
            PRIVATE_KEY = None
            PASSPHRASE = None
        else:
            PRIVATE_KEY_FILE = None
            PRIVATE_KEY = os.environ['PRIVATE_KEY'].encode("utf-8")
            PASSPHRASE = os.environ['PRIVATE_KEY_PASSPHRASE'].encode("utf-8")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")

    snowflake_config = SnowflakeConfig(
        SNOWFLAKE_ACCOUNT,
        USER,
        WAREHOUSE,
        ROLE,
        HOST,
        DATABASE,
        SCHEMA,
        PRIVATE_KEY_FILE,
        PRIVATE_KEY,
        PASSPHRASE)

    cortex_config = CortexConfig(
        DATABASE,
        SCHEMA,
        STAGE,
        FILE)
    try:
        handler = SnowflakeSQLHandler(snowflake_config, cortex_config)
        return handler
    except Exception as e:
        logging.error(f"Failed to create SnowflakeSQLHandler: {e}")
        raise
    finally:
        # Add any cleanup if needed
        pass


def basic_info_node(state: StateType) -> StateType:
    """Fetch basic information."""
    logger.info("Fetching basic information...")
    
    TABLE_GROUP_NAME = "basic_info"

    config_dict = state["context"].get("config")
    if config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state    

    try:
        fullctx = state["context"]
        result = fetchdata.fetch_basic_info(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_basic_info: {e}")
        raise
    logging.info(f"Fetching basic information completed.")
    return {"context": result}

def inventory_data_node(state: StateType) -> StateType:
    """Fetch inventory data."""
    logger.info("Fetching inventory data...")

    TABLE_GROUP_NAME = "inventory_data"
    config_dict = state["context"].get("config")

    if config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    
    try:
        fullctx = state["context"]
        result = fetchdata.fetch_inventory_data(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_inventory_data: {e}")
        raise
    return {"context": result}

def marketing_data_node(state: StateType) -> StateType:
    """Fetch marketing data."""
    logger.info("Fetching marketing data...")

    TABLE_GROUP_NAME = "marketing_data"
    config_dict = state["context"].get("config")

    if config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        result = fetchdata.fetch_marketing_data(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_marketing_data: {e}")
        raise
    return {"context": result}

def budget_data_node(state: StateType) -> StateType:
    """Fetch budget data."""
    logger.info("Fetching budget data...")

    TABLE_GROUP_NAME = "budget_data"

    config_dict = state["context"].get("config")

    if config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        result = fetchdata.fetch_budget_data(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_budget_data: {e}")
        raise
    return {"context": result}    

def performance_data_node(state: StateType) -> StateType:
    """Fetch performance data."""
    logger.info("Fetching performance data...")

    TABLE_GROUP_NAME = "performance_data"

    config_dict = state["context"].get("config")
    if config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        result = fetchdata.fetch_performance_data(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_performance_data: {e}")
        raise
    return {"context": result}


def combine_results(state: StateType) -> DashboardDataJson:
    """Combine all fetched data into DescriptiveDashboardData."""
    logger.info("Combining results...")
    print("state keys:", state["context"].keys())
    #print(state)

    basic = state["context"].get("basic_info")
    inv = state["context"].get("inv_data")
    mkt = state["context"].get("marketing_data")
    budget = state["context"].get("budget_data")
    perf = state["context"].get("performance_data")
    
    result = DashboardDataJson(
        basic["df_basic"] if basic else None,
        inv["df_inv_agg"] if inv else None,
        inv["df_monthly_inv"] if inv else None,
        mkt["df_ads_agg"] if mkt else None,
        mkt["df_monthly_ads"] if mkt else None,
        mkt["df_ga_agg"] if mkt else None,
        mkt["df_monthly_ga"] if mkt else None,
        perf["df_mystery"] if perf else None,
        budget["df_budget"] if budget else None,
        budget["df_monthly_budget"] if budget else None,
        perf["df_rank"] if perf else None,
        perf["df_oem"] if perf else None,
        perf["df_monthly_all"] if perf else None,
        budget["df_budget_avg"] if budget else None,
        budget["df_detailed_budget"] if budget else None,
        mkt["df_traffic_comparison"] if mkt else None,
        perf["df_missing_channels"] if perf else None,
        perf["df_avg_monthly_sales"] if perf else None,
        perf["df_mystery_monthly"] if perf else None,
        inv["df_inventory_lb"] if inv else None,
        inv["df_inventory_lb_state_brand_all"] if inv else None,
        inv["df_inventory_lb_state_limit"] if inv else None,
        inv["df_inventory_lb_brand_limit"] if inv else None,
        budget["df_budget_lb"] if budget else None,
        budget["df_budget_lb_state_brand_all"] if budget else None,
        budget["df_budget_lb_state_limit"]   if budget else None,
        budget["df_budget_lb_brand_limit"]   if budget else None,
        mkt["df_website_lb"] if mkt else None,
        basic["all_states_df"] if basic else None,
        basic["all_brands_df"] if basic else None
    )

    #state["result"] = result
    return  result

def initialize_state(state: StateType,config: RunnableConfig) -> StateType:
    """Initialize state before parallel processing."""
    #print("In init state", state)
    if not state.get("context"):
        logger.error("State must contain context fields")
        raise ValueError("State must contain context fields")
    print("config",config)
    config_dict = {} #config['configurable']
    print("config_dict",config_dict)

    logger.info("State initialized successfully.")
    return {"context": {"config" :config_dict}}

def single_table_data_node(state: StateType) -> StateType:
    """Fetch single table data."""
    logger.info("Fetching single table data...")
    return state

def nl_to_sql_node(state: StateType) -> StateType:
    # sql_query = state["context"]["nl_query"]
    # sql_handler.execute_query(sql_query)
    return state


def create_dashboard_graph() -> StateGraph:
    """Create the langgraph DAG for parallel dashboard data fetching."""
    
    logger.info("Creating dashboard graph...")

    # Create the workflow graph with state type annotation
    #workflow = StateGraph(Annotated[Dict[str, Any],"DashboardState"])
    workflow = StateGraph(StateType)
    
    # Add nodes
    workflow.add_node("init", initialize_state)
    
    # Nodes for batched data fetching
    workflow.add_node("basic_info", basic_info_node)
    workflow.add_node("inventory", inventory_data_node)
    workflow.add_node("marketing", marketing_data_node)
    workflow.add_node("budget", budget_data_node)
    workflow.add_node("performance", performance_data_node)
    workflow.add_node("combine", combine_results)
    
    # Node for single table data fetching
    workflow.add_node("single_table", single_table_data_node)

    # Node for NL to SQL conversion
    workflow.add_node("nl_to_sql", nl_to_sql_node)

    workflow.add_edge(START, "init")
    
    # Fan-out from init
    for node in ["basic_info", "marketing","inventory", "budget", "performance"]:
        workflow.add_edge("init", node)

    # Add edges from parallel nodes to combine with condition
    for node in ["basic_info", "marketing","inventory", "budget", "performance"]:
        workflow.add_edge(node,"combine")

    workflow.add_edge("init", "single_table")
    workflow.add_edge("init", "nl_to_sql")
    
    workflow.add_edge("single_table", END)
    workflow.add_edge("nl_to_sql", END)

    workflow.add_edge("combine", END)
    
    logger.info("Dashboard graph created successfully.")
    return workflow.compile()

config={}
sql_handler = get_sql_handler()
logger.info("SQL handler initialized.")

graph = create_dashboard_graph()
graph.name = "FusionDashboardGraph" 
