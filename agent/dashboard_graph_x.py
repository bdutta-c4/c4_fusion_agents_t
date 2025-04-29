"""
Langgraph DAG implementation for parallel dashboard data fetching.
"""
from typing import Any, Dict, List, Iterable,Annotated,TypedDict
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

import logging

LOG_FILE = 'FusionDashboard.log'

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger
logger = logging.getLogger("FusionDashboard")

load_dotenv("./agent/.env", override=True)


#State =  Annotated[Dict[str, Any], operator.add]
# class State(TypedDict):
#     df_key : Annotated[list[Dict[str]], operator.add]
@dataclass
class StateType(Dict[str, Any]):
    context: Annotated[Dict[str, Any], operator.or_]

#State = Annotated[list[Dict[str, Any]], operator.add]

#State =  Annotated[Dict[str, Any], operator.or_]
# def merge_dicts(a: dict, b: dict) -> dict:
#     # b takes precedence on key collisions
#     return {**a, **b}

# State = Annotated[Dict[str, Any], merge_dicts]
# Define node types for different data categories

@dataclass
class DashboardDataJson:
    df_basic: str
    #df_inv_agg: str
    #df_monthly_inv: str
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
    # df_inventory_lb: str
    # df_inventory_lb_state_brand_all: str
    # df_inventory_lb_state_limit: str
    # df_inventory_lb_brand_limit: str
    df_budget_lb: str
    df_budget_lb_state_brand_all: str
    df_budget_lb_state_limit: str
    df_budget_lb_brand_limit: str
    df_website_lb: str
    all_states_df: str
    all_brands_df: str

@dataclass
class DashboardData:
    df_basic: pd.DataFrame
    df_inv_agg: pd.DataFrame
    df_monthly_inv: pd.DataFrame
    df_ads_agg: pd.DataFrame
    df_monthly_ads: pd.DataFrame
    df_ga_agg: pd.DataFrame
    df_monthly_ga: pd.DataFrame
    df_mystery: pd.DataFrame
    df_budget: pd.DataFrame
    df_monthly_budget: pd.DataFrame
    df_rank: pd.DataFrame
    df_oem: pd.DataFrame
    df_monthly_all: pd.DataFrame
    df_budget_avg: pd.DataFrame
    df_detailed_budget: pd.DataFrame
    df_traffic_comparison: pd.DataFrame
    df_missing_channels: pd.DataFrame
    df_avg_monthly_sales: pd.DataFrame
    df_mystery_monthly: pd.DataFrame
    df_inventory_lb: pd.DataFrame
    df_inventory_lb_state_brand_all: pd.DataFrame
    df_inventory_lb_state_limit: pd.DataFrame
    df_inventory_lb_brand_limit: pd.DataFrame
    df_budget_lb: pd.DataFrame
    df_budget_lb_state_brand_all: pd.DataFrame
    df_budget_lb_state_limit: pd.DataFrame
    df_budget_lb_brand_limit: pd.DataFrame
    df_website_lb: pd.DataFrame
    all_states_df: pd.DataFrame
    all_brands_df: pd.DataFrame


@dataclass
class DescriptiveDashboardData:
    """Convenience class for all the query data and their descriptions"""

    df_basic: QueryResult
    df_inv_agg: QueryResult
    df_monthly_inv: QueryResult
    df_ads_agg: QueryResult
    df_monthly_ads: QueryResult
    df_ga_agg: QueryResult
    df_monthly_ga: QueryResult
    df_mystery: QueryResult
    df_budget: QueryResult
    df_monthly_budget: QueryResult
    df_rank: QueryResult
    df_oem: QueryResult
    df_monthly_all: QueryResult
    df_budget_avg: QueryResult
    df_detailed_budget: QueryResult
    df_traffic_comparison: QueryResult
    df_missing_channels: QueryResult
    df_avg_monthly_sales: QueryResult
    df_mystery_monthly: QueryResult
    df_inventory_lb: QueryResult
    df_inventory_lb_state_brand_all: QueryResult
    df_inventory_lb_state_limit: QueryResult
    df_inventory_lb_brand_limit: QueryResult
    df_budget_lb: QueryResult
    df_budget_lb_state_brand_all: QueryResult
    df_budget_lb_state_limit: QueryResult
    df_budget_lb_brand_limit: QueryResult
    df_website_lb: QueryResult
    all_states_df: QueryResult
    all_brands_df: QueryResult


    def iter_fields(self, exclude: list[str] = None) -> Iterable[QueryResult]:
        exclude = exclude or []
        for field in fields(self):
            if field.name not in exclude:
                yield getattr(self, field.name)

    @property
    def raw_data(self) -> DashboardData:
        return DashboardData(
            self.df_basic.data,
            self.df_inv_agg.data,
            self.df_monthly_inv.data,
            self.df_ads_agg.data,
            self.df_monthly_ads.data,
            self.df_ga_agg.data,
            self.df_monthly_ga.data,
            self.df_mystery.data,
            self.df_budget.data,
            self.df_monthly_budget.data,
            self.df_rank.data,
            self.df_oem.data,
            self.df_monthly_all.data,
            self.df_budget_avg.data,
            self.df_detailed_budget.data,
            self.df_traffic_comparison.data,
            self.df_missing_channels.data,
            self.df_avg_monthly_sales.data,
            self.df_mystery_monthly.data,
            self.df_inventory_lb.data,
            self.df_inventory_lb_state_brand_all.data,
            self.df_inventory_lb_state_limit.data,
            self.df_inventory_lb_brand_limit.data,
            self.df_budget_lb.data,
            self.df_budget_lb_state_brand_all.data,
            self.df_budget_lb_state_limit.data,
            self.df_budget_lb_brand_limit.data,
            self.df_website_lb.data,
            self.all_states_df.data,
            self.all_brands_df.data)



@dataclass
class BasicInfo:
    df_basic: pd.DataFrame
    all_states_df: pd.DataFrame
    all_brands_df: pd.DataFrame

@dataclass
class InventoryData:
    df_inv_agg: pd.DataFrame
    df_monthly_inv: pd.DataFrame
    df_inventory_lb: pd.DataFrame
    df_inventory_lb_state_brand_all: pd.DataFrame
    df_inventory_lb_state_limit: pd.DataFrame
    df_inventory_lb_brand_limit: pd.DataFrame

@dataclass
class MarketingData:
    df_ads_agg: pd.DataFrame
    df_monthly_ads: pd.DataFrame
    df_ga_agg: pd.DataFrame
    df_monthly_ga: pd.DataFrame
    df_traffic_comparison: pd.DataFrame
    df_website_lb: pd.DataFrame

@dataclass
class BudgetData:
    df_budget: pd.DataFrame
    df_monthly_budget: pd.DataFrame
    df_budget_avg: pd.DataFrame
    df_detailed_budget: pd.DataFrame
    df_budget_lb: pd.DataFrame
    df_budget_lb_state_brand_all: pd.DataFrame
    df_budget_lb_state_limit: pd.DataFrame
    df_budget_lb_brand_limit: pd.DataFrame

@dataclass
class PerformanceData:
    df_mystery: pd.DataFrame
    df_mystery_monthly: pd.DataFrame
    df_rank: pd.DataFrame
    df_oem: pd.DataFrame
    df_monthly_all: pd.DataFrame
    df_missing_channels: pd.DataFrame
    df_avg_monthly_sales: pd.DataFrame


def get_sql_handler():
    """Dependency to get SQL handler instance."""
    try:
        HOST = os.environ['HOST']
        USER = os.environ['USER']
        SNOWFLAKE_ACCOUNT = os.environ['SNOWFLAKE_ACCOUNT']
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


    DATABASE = "CORTEX_ANALYST_DEMO"
    SCHEMA = "FUSION_DATA"
    STAGE = "RAW_DATA"
    FILE = "semantic_model_latest.yaml"
    WAREHOUSE = "COMPUTE_WH"
    ROLE = "CORTEX_USER_ROLE"
    
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

def df_serializer(df: pd.DataFrame) -> str:
    return df.to_json()

def df_deserializer(df: str) -> pd.DataFrame:
    return pd.read_json(df)

# Node functions for parallel processing
def fetch_basic_info(state: StateType) -> StateType:
    """Fetch basic dealer information."""
    logger.info("Fetching basic dealer information...")
    try:
        fullctx = state["context"]

        ctx = {
            "jira_id": fullctx["jira_id"],
            "start_date": fullctx["start_date"],
            "end_date": fullctx["end_date"]
        }
        executor = sql_handler.execute_query_raw

        df_basic = get_basic_dealership_info_query(ctx["jira_id"]).execute(executor)
        all_states_df = get_all_states_query().execute(executor)
        all_brands_df = get_all_brands_query().execute(executor)

        df_basic_serialized = df_serializer(df_basic.data)
        all_states_df_serialized = df_serializer(all_states_df.data)
        all_brands_df_serialized = df_serializer(all_brands_df.data)

        result = {"basic_info": {
            "df_basic": df_basic_serialized,
            "all_states_df": all_states_df_serialized,
            "all_brands_df": all_brands_df_serialized
        }}
        logger.info("Basic info fetched successfully.")
    except Exception as e:
        logging.error(f"Error in fetch_basic_info: {e}")
        raise
    return {"context": result}


def fetch_inventory_data(state: StateType) -> StateType:
    """Fetch inventory related data."""
    logger.info("Fetching inventory data...")
    fullctx = state["context"]

    ctx = {
    "jira_id":  fullctx["jira_id"],
    "start_date": fullctx["start_date"],
    "end_date": fullctx["end_date"]
    }

    executor = sql_handler.execute_query_raw
    
    df_inv_agg = get_aggregated_inventory_sales_query(**ctx).execute(executor)
    df_monthly_inv = get_monthly_inventory_sales_query(**ctx).execute(executor)
    
    df_inventory_lb = get_inventory_leaderboard_query()(**ctx).execute(executor)
    df_inventory_lb_state_brand_all = get_inventory_leaderboard_query(
        state_filter=fullctx["state"], brand_filter=fullctx["brand"], limit=None)(**ctx).execute(executor)
    df_inventory_lb_state_limit = get_inventory_leaderboard_query(
        state_filter=fullctx["state"], limit=fullctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    df_inventory_lb_brand_limit = get_inventory_leaderboard_query(
        brand_filter=fullctx["brand"], limit=fullctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    
    df_inv_agg_serialized = df_serializer(df_inv_agg.data)
    df_monthly_inv_serialized = df_serializer(df_monthly_inv.data)
    df_inventory_lb_serialized = df_serializer(df_inventory_lb.data)
    df_inventory_lb_state_brand_all_serialized = df_serializer(df_inventory_lb_state_brand_all.data)
    df_inventory_lb_state_limit_serialized = df_serializer(df_inventory_lb_state_limit.data)
    df_inventory_lb_brand_limit_serialized = df_serializer(df_inventory_lb_brand_limit.data)

    result = {"inv_data": {
        "df_inv_agg": df_inv_agg_serialized,
        "df_monthly_inv": df_monthly_inv_serialized,
        "df_inventory_lb": df_inventory_lb_serialized,
        "df_inventory_lb_state_brand_all": df_inventory_lb_state_brand_all_serialized,
        "df_inventory_lb_state_limit": df_inventory_lb_state_limit_serialized,
        "df_inventory_lb_brand_limit": df_inventory_lb_brand_limit_serialized
    }}
    logger.info("Inventory data fetched successfully.")
    return {"context": result}


def fetch_marketing_data(state: StateType) -> StateType:
    """Fetch marketing and website traffic data."""
    try:
        logger.info("Fetching marketing data...")
        fullctx = state["context"]

        ctx = {
        "jira_id": fullctx["jira_id"],
        "start_date": fullctx["start_date"],
        "end_date": fullctx["end_date"]
        }
        executor = sql_handler.execute_query_raw
        
        df_ads_agg = get_aggregated_google_ads_query(**ctx).execute(executor)
        df_monthly_ads = get_monthly_google_ads_query(**ctx).execute(executor)
        df_ga_agg = get_ga4_website_traffic_query(**ctx).execute(executor)
        df_monthly_ga = get_monthly_ga4_website_traffic_query(**ctx).execute(executor)
        df_traffic_comparison = get_website_traffic_breakdown_query(**ctx).execute(executor)
        df_website_lb = get_website_traffic_leaderboard_query()(**ctx).execute(executor)
        
        df_ads_agg_serialized = df_serializer(df_ads_agg.data)
        df_monthly_ads_serialized = df_serializer(df_monthly_ads.data)
        df_ga_agg_serialized = df_serializer(df_ga_agg.data)
        df_monthly_ga_serialized = df_serializer(df_monthly_ga.data)
        df_traffic_comparison_serialized = df_serializer(df_traffic_comparison.data)
        df_website_lb_serialized = df_serializer(df_website_lb.data)
   
        result = {"marketing_data":{           
            "df_ads_agg": df_ads_agg_serialized,
            "df_monthly_ads": df_monthly_ads_serialized,
            "df_ga_agg": df_ga_agg_serialized,
            "df_monthly_ga": df_monthly_ga_serialized,
            "df_traffic_comparison": df_traffic_comparison_serialized,
            "df_website_lb": df_website_lb_serialized
        }}
        logger.info("Marketing data fetched successfully.")
    except Exception as e:
        logging.error(f"Error in fetch_marketing_data: {e}")
        raise
    return {"context": result}

def fetch_budget_data(state: StateType) -> StateType:
    """Fetch budget related data."""
    logger.info("Fetching budget data...")
    fullctx = state["context"]

    ctx = {
    "jira_id": fullctx["jira_id"],
    "start_date": fullctx["start_date"],
    "end_date": fullctx["end_date"]
    }
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw
    
    df_budget = get_budget_allocation_query(**ctx).execute(executor)
    df_monthly_budget = get_monthly_budget_allocation_query(**ctx).execute(executor)
    df_budget_avg = get_avg_monthly_budget_query(**ctx).execute(executor)
    df_detailed_budget = get_detailed_budget_breakdown_query(**ctx).execute(executor)
    
    df_budget_lb = get_budget_leaderboard_query()(**ctx).execute(executor)
    df_budget_lb_state_brand_all = get_budget_leaderboard_query()(**ctx).execute(executor)
    df_budget_lb_state_limit = get_budget_leaderboard_query(
        state_filter=fullctx["state"], limit=fullctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    df_budget_lb_brand_limit = get_budget_leaderboard_query(
        brand_filter=fullctx["brand"], limit=fullctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    
    df_budget_serialized = df_serializer(df_budget.data)
    df_monthly_budget_serialized = df_serializer(df_monthly_budget.data)
    df_budget_avg_serialized = df_serializer(df_budget_avg.data)
    df_detailed_budget_serialized = df_serializer(df_detailed_budget.data)
    df_budget_lb_serialized = df_serializer(df_budget_lb.data)
    df_budget_lb_state_brand_all_serialized = df_serializer(df_budget_lb_state_brand_all.data)
    df_budget_lb_state_limit_serialized = df_serializer(df_budget_lb_state_limit.data)
    df_budget_lb_brand_limit_serialized = df_serializer(df_budget_lb_brand_limit.data)
    
    result = {"budget_data": {
        "df_budget": df_budget_serialized,
        "df_monthly_budget": df_monthly_budget_serialized,
        "df_budget_avg": df_budget_avg_serialized,
        "df_detailed_budget": df_detailed_budget_serialized,
        "df_budget_lb": df_budget_lb_serialized,
        "df_budget_lb_state_brand_all": df_budget_lb_state_brand_all_serialized,
        "df_budget_lb_state_limit": df_budget_lb_state_limit_serialized,
        "df_budget_lb_brand_limit": df_budget_lb_brand_limit_serialized
    }}
    logger.info("Budget data fetched successfully.")
    return {"context": result}

def fetch_performance_data(state: StateType) -> StateType:
    """Fetch performance and analytics data."""
    logger.info("Fetching performance data...")
    fullctx = state["context"]
    ctx = {
    "jira_id": fullctx["jira_id"],
    "start_date": fullctx["start_date"],
    "end_date": fullctx["end_date"]
    }
    executor = sql_handler.execute_query_raw
    
    df_mystery = get_mystery_shop_stats_query(**ctx).execute(executor)
    df_mystery_monthly = get_monthly_mystery_shop_breakdown_query(**ctx).execute(executor)
    df_rank = get_overall_sales_rank_query(**ctx).execute(executor)
    df_oem = get_oem_sales_rank_query(**ctx).execute(executor)
    df_monthly_all = get_all_dealers_monthly_data_query(**ctx).execute(executor)
    df_missing_channels = get_missing_channels_query(**ctx).execute(executor)
    df_avg_monthly_sales = get_avg_monthly_sales_query(**ctx).execute(executor)
    
    df_mystery_serialized = df_serializer(df_mystery.data)
    df_mystery_monthly_serialized = df_serializer(df_mystery_monthly.data)
    df_rank_serialized = df_serializer(df_rank.data)
    df_oem_serialized = df_serializer(df_oem.data)
    df_monthly_all_serialized = df_serializer(df_monthly_all.data)
    df_missing_channels_serialized = df_serializer(df_missing_channels.data)
    df_avg_monthly_sales_serialized = df_serializer(df_avg_monthly_sales.data)
    
    result = {"performance_data": {
        "df_mystery": df_mystery_serialized,
        "df_mystery_monthly": df_mystery_monthly_serialized,
        "df_rank": df_rank_serialized,
        "df_oem": df_oem_serialized,
        "df_monthly_all": df_monthly_all_serialized,
        "df_missing_channels": df_missing_channels_serialized,
        "df_avg_monthly_sales": df_avg_monthly_sales_serialized
    }}
    logger.info("Performance data fetched successfully.")
    return {"context": result}

def combine_results(state: StateType) -> DashboardDataJson:
    """Combine all fetched data into DescriptiveDashboardData."""
    logger.info("Combining results...")
    print("state keys:", state.keys())
    #print(state)

    basic = state["context"]["basic_info"]
    inv = state["context"]["inv_data"]
    mkt = state["context"]["marketing_data"]
    budget = state["context"]["budget_data"]
    perf = state["context"]["performance_data"]
    
    result = DashboardDataJson(
        basic["df_basic"],
        #inv["df_inv_agg"],
        #inv["df_monthly_inv"],
        mkt["df_ads_agg"],
        mkt["df_monthly_ads"],
        mkt["df_ga_agg"],
        mkt["df_monthly_ga"],
        perf["df_mystery"],
        budget["df_budget"],
        budget["df_monthly_budget"],
        perf["df_rank"],
        perf["df_oem"],
        perf["df_monthly_all"],
        budget["df_budget_avg"],
        budget["df_detailed_budget"],
        mkt["df_traffic_comparison"],
        perf["df_missing_channels"],
        perf["df_avg_monthly_sales"],
        perf["df_mystery_monthly"],
        #inv["df_inventory_lb"],
        #inv["df_inventory_lb_state_brand_all"],
        #inv["df_inventory_lb_state_limit"],
        #inv["df_inventory_lb_brand_limit"],
        budget["df_budget_lb"],
        budget["df_budget_lb_state_brand_all"],
        budget["df_budget_lb_state_limit"],
        budget["df_budget_lb_brand_limit"],
        mkt["df_website_lb"],
        basic["all_states_df"],
        basic["all_brands_df"] 
    )

    #state["result"] = result
    return  result

def initialize_state(state: StateType) -> StateType:
    """Initialize state before parallel processing."""
    #print("In init state", state)
    if not state.get("context"):
        logger.error("State must contain context fields")
        raise ValueError("State must contain context fields")

    logger.info("State initialized successfully.")
    return state

def create_dashboard_graph() -> StateGraph:
    """Create the langgraph DAG for parallel dashboard data fetching."""
    
    logger.info("Creating dashboard graph...")

    # Create the workflow graph with state type annotation
    #workflow = StateGraph(Annotated[Dict[str, Any],"DashboardState"])
    workflow = StateGraph(StateType)
    
    # Add nodes
    workflow.add_node("init", initialize_state)
    workflow.add_node("basic_info", fetch_basic_info)
    workflow.add_node("inventory", fetch_inventory_data)
    workflow.add_node("marketing", fetch_marketing_data)
    workflow.add_node("budget", fetch_budget_data)
    workflow.add_node("performance", fetch_performance_data)
    workflow.add_node("combine", combine_results)
    
    workflow.add_edge(START, "init")
    
    # Fan-out from init
    for node in ["basic_info", "marketing","inventory", "budget", "performance"]:
        workflow.add_edge("init", node)

    # Add edges from parallel nodes to combine with condition
    for node in ["basic_info", "marketing","inventory", "budget", "performance"]:
        workflow.add_edge(node,"combine")



    workflow.add_edge("combine", END)
    
    logger.info("Dashboard graph created successfully.")
    return workflow.compile()

#state = State = {}
sql_handler = get_sql_handler()
logger.info("SQL handler initialized.")

# Create and run the graph
graph = create_dashboard_graph()
graph.name = "FusionDashboardGraph" 

