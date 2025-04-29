"""
Langgraph DAG implementation for parallel dashboard data fetching.
"""
from typing import Any, Dict, List, Iterable,Annotated
from datetime import date
from dataclasses import dataclass

import pandas as pd
from langgraph.graph import START, END, StateGraph
from typing import Annotated
import operator 
import sys,os
from dotenv import load_dotenv

sys.path.append("src/agent")

from sql_handler import SnowflakeSQLHandler
from queries import *
from cortex import CortexConfig, SnowflakeConfig


# Define a State class that extends Dict[str, Any] with operator.add reducer
# @dataclass
# class State(Dict[str, Any]):
#     some_key: Annotated[list,  add]

# class State(Dict):
#     Annotated[
#     Dict[str, List[Any]],
#     add]

load_dotenv(override=True)
#State =  Annotated["somekey" : Dict[str, Any], operator.add]
State =  Annotated[Dict[str, Any], operator.add]

# Define node types for different data categories

@dataclass
class DashboardData:
    df_basic: pd.DataFrame
    #df_inv_agg: pd.DataFrame
    #df_monthly_inv: pd.DataFrame
    df_ads_agg: pd.DataFrame
    df_monthly_ads: pd.DataFrame
    df_ga_agg: pd.DataFrame
    df_monthly_ga: pd.DataFrame
    df_mystery: pd.DataFrame
    #df_budget: pd.DataFrame
    #df_monthly_budget: pd.DataFrame
    df_rank: pd.DataFrame
    df_oem: pd.DataFrame
    df_monthly_all: pd.DataFrame
    #df_budget_avg: pd.DataFrame
    #df_detailed_budget: pd.DataFrame
    df_traffic_comparison: pd.DataFrame
    df_missing_channels: pd.DataFrame
    df_avg_monthly_sales: pd.DataFrame
    df_mystery_monthly: pd.DataFrame
    #df_inventory_lb: pd.DataFrame
    #df_inventory_lb_state_brand_all: pd.DataFrame
    #df_inventory_lb_state_limit: pd.DataFrame
    #df_inventory_lb_brand_limit: pd.DataFrame
    #df_budget_lb: pd.DataFrame
    #df_budget_lb_state_brand_all: pd.DataFrame
    #df_budget_lb_state_limit: pd.DataFrame
    #df_budget_lb_brand_limit: pd.DataFrame
    #df_website_lb: pd.DataFrame
    all_states_df: pd.DataFrame
    all_brands_df: pd.DataFrame


@dataclass
class DescriptiveDashboardData:
    """Convenience class for all the query data and their descriptions"""
    # df_basic: QueryResult
    # df_inv_agg: QueryResult
    # df_monthly_inv: QueryResult
    # df_ads_agg: QueryResult
    # df_monthly_ads: QueryResult
    # df_ga_agg: QueryResult
    # df_monthly_ga: QueryResult
    # df_mystery: QueryResult
    # df_budget: QueryResult
    # df_monthly_budget: QueryResult
    # df_rank: QueryResult
    # df_oem: QueryResult
    # df_monthly_all: QueryResult
    # df_budget_avg: QueryResult
    # df_detailed_budget: QueryResult
    # df_traffic_comparison: QueryResult
    # df_missing_channels: QueryResult
    # df_avg_monthly_sales: QueryResult
    # df_mystery_monthly: QueryResult
    # df_inventory_lb: QueryResult
    # df_inventory_lb_state_brand_all: QueryResult
    # df_inventory_lb_state_limit: QueryResult
    # df_inventory_lb_brand_limit: QueryResult
    # df_budget_lb: QueryResult
    # df_budget_lb_state_brand_all: QueryResult
    # df_budget_lb_state_limit: QueryResult
    # df_budget_lb_brand_limit: QueryResult
    # df_website_lb: QueryResult
    # all_states_df: QueryResult
    # all_brands_df: QueryResult

    df_basic: QueryResult
    # df_inv_agg: QueryResult
    # df_monthly_inv: QueryResult
    df_ads_agg: QueryResult
    df_monthly_ads: QueryResult
    df_ga_agg: QueryResult
    df_monthly_ga: QueryResult
    df_mystery: QueryResult
    # df_budget: QueryResult
    # df_monthly_budget: QueryResult
    df_rank: QueryResult
    df_oem: QueryResult
    df_monthly_all: QueryResult
    # df_budget_avg: QueryResult
    # df_detailed_budget: QueryResult
    df_traffic_comparison: QueryResult
    df_missing_channels: QueryResult
    df_avg_monthly_sales: QueryResult
    df_mystery_monthly: QueryResult
    # df_inventory_lb: QueryResult
    # df_inventory_lb_state_brand_all: QueryResult
    # df_inventory_lb_state_limit: QueryResult
    # df_inventory_lb_brand_limit: QueryResult
    # df_budget_lb: QueryResult
    # df_budget_lb_state_brand_all: QueryResult
    # df_budget_lb_state_limit: QueryResult
    # df_budget_lb_brand_limit: QueryResult
    # df_website_lb: QueryResult
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
            #self.df_inv_agg.data,
            #self.df_monthly_inv.data,
            self.df_ads_agg.data,
            self.df_monthly_ads.data,
            self.df_ga_agg.data,
            self.df_monthly_ga.data,
            self.df_mystery.data,
            #self.df_budget.data,
            #self.df_monthly_budget.data,
            self.df_rank.data,
            self.df_oem.data,
            self.df_monthly_all.data,
            #self.df_budget_avg.data,
            #self.df_detailed_budget.data,
            self.df_traffic_comparison.data,
            self.df_missing_channels.data,
            self.df_avg_monthly_sales.data,
            self.df_mystery_monthly.data,
            #self.df_inventory_lb.data,
            #self.df_inventory_lb_state_brand_all.data,
            #self.df_inventory_lb_state_limit.data,
            #self.df_inventory_lb_brand_limit.data,
            #self.df_budget_lb.data,
            #self.df_budget_lb_state_brand_all.data,
            #self.df_budget_lb_state_limit.data,
            #self.df_budget_lb_brand_limit.data,
            #self.df_website_lb.data,
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
    finally:
        # Add any cleanup if needed
        pass


# Node functions for parallel processing
def fetch_basic_info(state: State) -> Dict[str, Any]:
    """Fetch basic dealer information."""
    ctx = state["context"]
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw

    df_basic = get_basic_dealership_info_query(ctx["jira_id"]).execute(executor)
    all_states_df = get_all_states_query().execute(executor)
    all_brands_df = get_all_brands_query().execute(executor)
    
    state["basic_info"] = BasicInfo(df_basic, all_states_df, all_brands_df)
    #return state

def fetch_inventory_data(state: State) -> Dict[str, Any]:
    """Fetch inventory related data."""
    ctx = state["context"]
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw
    
    df_inv_agg = get_aggregated_inventory_sales_query(**ctx).execute(executor)
    df_monthly_inv = get_monthly_inventory_sales_query(**ctx).execute(executor)
    
    df_inventory_lb = get_inventory_leaderboard_query()(**ctx).execute(executor)
    df_inventory_lb_state_brand_all = get_inventory_leaderboard_query(
        state_filter=ctx["state"], brand_filter=ctx["brand"], limit=None)(**ctx).execute(executor)
    df_inventory_lb_state_limit = get_inventory_leaderboard_query(
        state_filter=ctx["state"], limit=ctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    df_inventory_lb_brand_limit = get_inventory_leaderboard_query(
        brand_filter=ctx["brand"], limit=ctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    
    state["inventory_data"] = InventoryData(
        df_inv_agg, df_monthly_inv, df_inventory_lb,
        df_inventory_lb_state_brand_all, df_inventory_lb_state_limit,
        df_inventory_lb_brand_limit
    )
    #return state


def fetch_marketing_data(state: State) -> Dict[str, Any]:
    """Fetch marketing and website traffic data."""
    ctx = state["context"]
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw
    
    df_ads_agg = get_aggregated_google_ads_query(**ctx).execute(executor)
    df_monthly_ads = get_monthly_google_ads_query(**ctx).execute(executor)
    df_ga_agg = get_ga4_website_traffic_query(**ctx).execute(executor)
    df_monthly_ga = get_monthly_ga4_website_traffic_query(**ctx).execute(executor)
    df_traffic_comparison = get_website_traffic_breakdown_query(**ctx).execute(executor)
    df_website_lb = get_website_traffic_leaderboard_query()(**ctx).execute(executor)
    
    state["marketing_data"] = MarketingData(
        df_ads_agg, df_monthly_ads, df_ga_agg, df_monthly_ga,
        df_traffic_comparison, df_website_lb
    )
    #return state

def fetch_budget_data(state: State) -> Dict[str, Any]:
    """Fetch budget related data."""
    ctx = state["context"]
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw
    
    df_budget = get_budget_allocation_query(**ctx).execute(executor)
    df_monthly_budget = get_monthly_budget_allocation_query(**ctx).execute(executor)
    df_budget_avg = get_avg_monthly_budget_query(**ctx).execute(executor)
    df_detailed_budget = get_detailed_budget_breakdown_query(**ctx).execute(executor)
    
    df_budget_lb = get_budget_leaderboard_query()(**ctx).execute(executor)
    df_budget_lb_state_brand_all = get_budget_leaderboard_query()(**ctx).execute(executor)
    df_budget_lb_state_limit = get_budget_leaderboard_query(
        state_filter=ctx["state"], limit=ctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    df_budget_lb_brand_limit = get_budget_leaderboard_query(
        brand_filter=ctx["brand"], limit=ctx["ai_insight_lb_limit"])(**ctx).execute(executor)
    
    state["budget_data"] = BudgetData(
        df_budget, df_monthly_budget, df_budget_avg, df_detailed_budget,
        df_budget_lb, df_budget_lb_state_brand_all, df_budget_lb_state_limit,
        df_budget_lb_brand_limit
    )
    #return state

def fetch_performance_data(state: State) -> Dict[str, Any]:
    """Fetch performance and analytics data."""
    ctx = state["context"]
    #executor = state["executor"]
    executor = sql_handler.execute_query_raw
    
    df_mystery = get_mystery_shop_stats_query(**ctx).execute(executor)
    df_mystery_monthly = get_monthly_mystery_shop_breakdown_query(**ctx).execute(executor)
    df_rank = get_overall_sales_rank_query(**ctx).execute(executor)
    df_oem = get_oem_sales_rank_query(**ctx).execute(executor)
    df_monthly_all = get_all_dealers_monthly_data_query(**ctx).execute(executor)
    df_missing_channels = get_missing_channels_query(**ctx).execute(executor)
    df_avg_monthly_sales = get_avg_monthly_sales_query(**ctx).execute(executor)
    
    state["performance_data"] = PerformanceData(
        df_mystery, df_mystery_monthly, df_rank, df_oem,
        df_monthly_all, df_missing_channels, df_avg_monthly_sales
    )
    #return state

def combine_results(state: State) -> Dict[str, Any]:
    """Combine all fetched data into DescriptiveDashboardData."""
    basic = state["basic_info"]
    #inv = state["inventory_data"]
    mkt = state["marketing_data"]
    #budget = state["budget_data"]
    perf = state["performance_data"]
    
    result = DescriptiveDashboardData(
        basic.df_basic,
        #inv.df_inv_agg,
        #inv.df_monthly_inv,
        mkt.df_ads_agg,
        mkt.df_monthly_ads,
        mkt.df_ga_agg,
        mkt.df_monthly_ga,
        perf.df_mystery,
        #budget.df_budget,
        #budget.df_monthly_budget,
        perf.df_rank,
        perf.df_oem,
        perf.df_monthly_all,
        #budget.df_budget_avg,
        #budget.df_detailed_budget,
        mkt.df_traffic_comparison,
        perf.df_missing_channels,
        perf.df_avg_monthly_sales,
        perf.df_mystery_monthly,
        #inv.df_inventory_lb,
        #inv.df_inventory_lb_state_brand_all,
        #inv.df_inventory_lb_state_limit,
        #inv.df_inventory_lb_brand_limit,
        #budget.df_budget_lb,
        # budget.df_budget_lb_state_brand_all,
        # budget.df_budget_lb_state_limit,
        # budget.df_budget_lb_brand_limit,
        # mkt.df_website_lb,
        basic.all_states_df,
        basic.all_brands_df
    )

    state["result"] = result.raw_data
    return  state["result"] 

def initialize_state(state: State) -> State:
    """Initialize state before parallel processing."""

    print("State", state)
    if not state.get("context"):
        raise ValueError("State must contain 'context' fields")

    #state["executor"] = sql_handler.execute_query_raw 
    #if sql_handler else None
    # Validate required fields
    return state

def create_dashboard_graph() -> StateGraph:
    """Create the langgraph DAG for parallel dashboard data fetching."""
    
    
    # Create the workflow graph with state type annotation
    workflow = StateGraph(Annotated[Dict, "DashboardState"])
    #workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("init", initialize_state)
    workflow.add_node("basic_info", fetch_basic_info)
    #workflow.add_node("inventory", fetch_inventory_data)
    workflow.add_node("marketing", fetch_marketing_data)
    #workflow.add_node("budget", fetch_budget_data)
    workflow.add_node("performance", fetch_performance_data)
    workflow.add_node("combine", combine_results)
    
    # Define conditional routing from START
    def route_from_start(state: State) -> List[str]:
        # Route to all parallel nodes
        return ["basic_info", "inventory", "marketing", "budget", "performance"]
    
    # Define conditional routing to combine
    def route_to_combine(state: State) -> str:
        # Check if all parallel nodes have completed
        required_data = ["basic_info", "inventory_data", "marketing_data", 
                        "budget_data", "performance_data"]
        if all(key in state for key in required_data):
            return "combine"
        return None
    
    # Fan-out from init
    for node in ["basic_info", "marketing", "performance"]:
        workflow.add_edge("init", node)

    # Add edges from parallel nodes to combine with condition
    for node in ["basic_info", "marketing", "performance"]:
        workflow.add_edge(node,"combine")


    workflow.add_edge(START, "init")
    
    workflow.add_edge("combine", END)
    
    return workflow.compile()

# def get_dashboard_data_parallel(
#     jira_id: str,
#     #state: str,
#     #brand: str,
#     start_date: date,
#     end_date: date,
#     ai_insight_lb_limit: int = 50
# ) -> DescriptiveDashboardData:

state = State = {}
sql_handler = get_sql_handler()

# state["context"] = {
#         "jira_id": jira_id,
#   #      "state": state,
#   #      "brand": brand,
#         "start_date": start_date,
#         "end_date": end_date,
#         "ai_insight_lb_limit": ai_insight_lb_limit
#     }

# state["context"] = {
#         "jira_id": jira_id,
#         "start_date": start_date,
#         "end_date": end_date,
#         "ai_insight_lb_limit": ai_insight_lb_limit
#     }
state["executor"] = sql_handler.execute_query_raw if sql_handler else None

print(state)

# # Create initial state
# state = State(
#     {
#     "context": {
#         "jira_id": jira_id,
#   #      "state": state,
#   #      "brand": brand,
#         "start_date": start_date,
#         "end_date": end_date,
#         "ai_insight_lb_limit": ai_insight_lb_limit
#     },
#     "executor": sql_handler.execute_query_raw if sql_handler else None
# })


#sql_handler = get_sql_handler()

# Create and run the graph
graph = create_dashboard_graph()
graph.name = "FusionDashboardGraph" 

