from typing import Any, Dict, List, Iterable,Annotated,TypedDict,Optional
from datetime import date
from dataclasses import dataclass

import pandas as pd
from langgraph.graph import START, END, StateGraph
import operator 
import sys,os

from dotenv import load_dotenv


sys.path.append("agent")
from  agent.environments import load_aws_variables, load_encrypted_variables


from sql_handler import SnowflakeSQLHandler
from queries import *
from cortex import CortexConfig, SnowflakeConfig
import fetchdata_knownqueries as fetchdata

import json

import logging

LOG_FILE = 'FusionDashboard.log'

current_path=os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(current_path,".env"), override=True)

load_encrypted_variables()

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


# Common settings that don't change between environments
SCHEMA = "APP"
STAGE = "SEMANTIC_MODEL_STAGE"
WAREHOUSE = "FUSION_WH"


def _get_environment_specific_configs():
    """
    Get environment-specific configuration values based on APP_ENVIRONMENT

    Returns:
        tuple: (database, role, file, client_name_search_service)
    """
    # Determine environment - default to 'dev' if not specified
    env = os.getenv("APP_ENVIRONMENT", "dev").lower()

    if env == "prod" or env == "production":
        logger.info("Using PRODUCTION Snowflake configuration")
        return (
            "FUSION_PROD",  # database
            "FUSION_PROD_APP_ROLE",  # role
            "semantic_model_prod.yaml",  # file
            "FUSION_PROD_CLIENT_NAME_SEARCH_SERVICE"  # client name search service
        )
    else:  # default to development
        logger.info("Using DEVELOPMENT Snowflake configuration")
        return (
            "FUSION_DEV",  # database
            "FUSION_DEV_APP_ROLE",  # role
            "semantic_model_dev.yaml",  # file
            "FUSION_DEV_CLIENT_NAME_SEARCH_SERVICE"  # client name search service
        )

def get_sql_handler():
    """Dependency to get SQL handler instance."""
    # Get environment-specific settings
    database, role, file, client_name_search_service  = _get_environment_specific_configs()
    try:
        snowflake_host = os.environ['SNOWFLAKE_HOST']
        snowflake_user = os.environ['SNOWFLAKE_USER']
        snowflake_account = os.environ['SNOWFLAKE_ACCOUNT']
        
        missing_params = []
        if not snowflake_host:
            missing_params.append("SNOWFLAKE_HOST")
        if not snowflake_user:
            missing_params.append("SNOWFLAKE_USER")
        if not snowflake_account:
            missing_params.append("SNOWFLAKE_ACCOUNT")

        if missing_params:
            raise KeyError(f"Missing required Snowflake connection parameters: {', '.join(missing_params)}")

        # Handle private key authentication
        if os.getenv("USE_PRIVATE_KEY_FILE"):
            private_key_file = os.getenv('PRIVATE_KEY_FILE')
            private_key = None
            passphrase = None
            if not private_key_file:
                raise KeyError("Missing PRIVATE_KEY_FILE environment variable")
        else:
            private_key_file = None
            private_key = os.getenv('SNOWFLAKE_PRIVATE_KEY')
            passphrase = os.getenv(
                'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')

            if not all([private_key, passphrase]):
                raise KeyError("Missing Snowflake private key credentials")

            private_key = private_key.encode("utf-8")
            passphrase = passphrase.encode("utf-8")

    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")

    snowflake_config = SnowflakeConfig(
        snowflake_account,
        snowflake_user,
        WAREHOUSE,
        role,
        snowflake_host,
        database,
        SCHEMA,
        private_key_file,
        private_key,
        passphrase)

    cortex_config = CortexConfig(
        database,
        SCHEMA,
        STAGE,
        file,
        client_name_search_service)
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
    if config_dict and config_dict.get("DF_TABLE_GROUPS"):
        df_tables_group = config_dict.get("DF_TABLE_GROUPS", [])
        if TABLE_GROUP_NAME in df_tables_group:
            proceed = True
        else:
            proceed = False
    else:
        proceed = True
    
    #print(TABLE_GROUP_NAME,proceed)

    if not proceed:
        return state    

    try:
        fullctx = state["context"]
        sql_handler.OpenifClosed()
        result = fetchdata.fetch_basic_info(fullctx,sql_handler)
        #sql_handler
        
        # check if state,brand and OEM name were added by user
        # if not, add them
        df_basic_dict = result["basic_info"]["df_basic"]
        df_basic = json.loads(df_basic_dict)

        statenm = df_basic["STATE"]["0"]
        brandnm = df_basic["BRAND"]["0"]
        oemnm = df_basic["OEM_NAME"]["0"]

        # statenm = result["basic_info"]["df_basic"]["state"]["0"]
        # brand = result["basic_info"]["df_basic"]["brand"]["0"]
        # oem_name = result["basic_info"]["df_basic"]["OEM_Name"]["0"]

        if not fullctx.get("state"):
            fullctx["state"] = statenm
        if not fullctx.get("brand"):
            fullctx["brand"] = brandnm
        if not fullctx.get("oem_name"):
            fullctx["oem"] = oemnm


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
    
    if not proceed:
        return state
    
    try:
        fullctx = state["context"]
        sql_handler.OpenifClosed()
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
    
    #print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        sql_handler.OpenifClosed()
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
    
    #print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        sql_handler.OpenifClosed()
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
    
    #print(TABLE_GROUP_NAME,proceed)
    if not proceed:
        return state
    try:
        fullctx = state["context"]
        sql_handler.OpenifClosed()
        result = fetchdata.fetch_performance_data(fullctx,sql_handler)
    except Exception as e:
        logging.error(f"Error in fetch_performance_data: {e}")
        raise
    return {"context": result}


def combine_results(state: StateType) -> DashboardDataJson:
    """Combine all fetched data into DescriptiveDashboardData."""
    logger.info("Combining results...")
 #   print("state keys:", state["context"].keys())
    
    import json

    def safe_serialize(context):
        for key, value in context.items():
            try:
                json.dumps({key: value})
            except Exception as e:
                print(f"Cannot serialize key '{key}' because: {e}")

    safe_serialize( state["context"])

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
    #config_dict = config['configurable']
    config_dict = {} #config['configurable'].get("DF_TABLES") | config['configurable'].get("DF_TABLE_GROUPS") | config['configurable'].get("DF_NL_QUERY")
    
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
    workflow.add_edge("init", "basic_info")
    
    # Fan-out from init
    for node in ["marketing","inventory", "budget", "performance"]:
        workflow.add_edge("basic_info", node)

    # Add edges from parallel nodes to combine with condition
    for node in ["marketing","inventory", "budget", "performance"]:
        workflow.add_edge(node,"combine")

    workflow.add_edge("init", "single_table")
    workflow.add_edge("init", "nl_to_sql")
    
    workflow.add_edge("single_table", END)
    workflow.add_edge("nl_to_sql", END)

    workflow.add_edge("combine", END)
    
    logger.info("Dashboard graph created successfully.")
    return workflow.compile()

#config={}

sql_handler = get_sql_handler()
logger.info("SQL handler initialized.")

graph = create_dashboard_graph()
graph.name = "FusionDashboardGraph" 
