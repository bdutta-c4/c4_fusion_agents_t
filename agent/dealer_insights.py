from typing import Any, Dict, Annotated
import operator
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig

import logging
import sys,os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd

import sys
import numpy as np
import statshelper as sh 
from snowflake_fetcher_agent import create_dashboard_graph
from prompt_executor import create_graph as prompt_executor_graph

### TO DO
# Only load weave if it is Dev
#import weave

LOG_FILE = 'Fusion-dealer_insights.log'

#load_dotenv("./agent/.env", override=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Fusion-dealer_insights")

LANGSMITH_API_KEY=os.environ.get("LANGSMITH_API_KEY")
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

## helper modules
def deserialize_output(output):
    TABLE_GROUPS = ["basic_info","inv_data","marketing_data","budget_data","performance_data"]
    table_dfs = {}
    for tabgrp in TABLE_GROUPS:
        tabgrp_dict = output["context"][tabgrp]
        for table_ in tabgrp_dict.keys():
            try:
                df_ = pd.read_json(tabgrp_dict[table_])
                table_dfs[table_]=df_
            except Exception as e:
                print(table_,f"Failed to read JSON: {e}")
                pass
    return table_dfs

def get_inventory_cols(df: pd.DataFrame):
    """
    Searches for column names that include 'INVENTORY' and either 'NEW' or 'USED'.
    Returns a tuple: (new_inventory_col, used_inventory_col)
    """
    new_col = None
    used_col = None
    for col in df.columns:
        col_upper = col.upper()
        if "INVENTORY" in col_upper:
            if "NEW" in col_upper:
                new_col = col
            elif "USED" in col_upper:
                used_col = col
    return new_col, used_col

def dynamic_metric_analysis(metric_dict, df_monthly_all, sales_monthly, grouping_columns, selected_dealer_id, replace_zero=False):
    """
    For each metric in metric_dict (using monthly data), compute:
      - Overall monthly correlation and regression (using all month-dealer records)
      - Grouped monthly correlations (using df_monthly_all)
      - Lagged correlations (overall and for the selected dealer)
    """
    results = {}
    # Convert monthly sales column to numeric and, if desired, replace 0 with NaN
    df_monthly_all[sales_monthly] = pd.to_numeric(
        df_monthly_all[sales_monthly], errors='coerce')
    if replace_zero:
        df_monthly_all[sales_monthly] = df_monthly_all[sales_monthly].replace(
            0, np.nan)

    for metric_name, monthly_col in metric_dict.items():
        metric_results = {}
        # Convert the metric column to numeric and, if desired, replace 0 with NaN
        df_monthly_all[monthly_col] = pd.to_numeric(
            df_monthly_all[monthly_col], errors='coerce')
        if replace_zero:
            df_monthly_all[monthly_col] = df_monthly_all[monthly_col].replace(
                0, np.nan)

        # Overall monthly correlation and regression using all month-dealer records
        overall_corr = sh.basic_correlation(
            df_monthly_all, monthly_col, sales_monthly)
        overall_reg = sh.simple_linear_regression(
            df_monthly_all, monthly_col, sales_monthly)
        metric_results['Overall Monthly Correlation'] = overall_corr
        metric_results['Overall Monthly Regression'] = overall_reg

        # Grouped monthly correlations and regressions (e.g., by STATE, BRAND, OEM_Name)
        grouped_correlations = {}
        grouped_regressions = {}
        for group in grouping_columns:
            # Compute grouped correlations
            grouped_correlations[group] = sh.grouped_correlation(
                df_monthly_all, monthly_col, sales_monthly, group)
            # Compute grouped regression parameters for each group using our helper
            grouped_regressions[group] = grouped_regression(
                df_monthly_all, monthly_col, sales_monthly, group)
        metric_results['Grouped Monthly'] = {
            'correlations': grouped_correlations,
            'regressions': grouped_regressions
        }

        # Lagged analysis (overall)
        df_monthly_all["combined"] = "all"  # constant grouping variable
        lagged_overall = sh.lagged_correlation(
            df_monthly_all, "combined", monthly_col, sales_monthly)
        metric_results['Lagged Overall'] = lagged_overall

        # Lagged analysis for the selected dealer
        df_dealer_monthly = df_monthly_all[df_monthly_all["JIRA_ID"]
                                           == selected_dealer_id]
        if not df_dealer_monthly.empty:
            lagged_selected = sh.lagged_correlation(
                df_dealer_monthly, "JIRA_ID", monthly_col, sales_monthly)
            metric_results['Lagged Selected Dealer'] = lagged_selected
        else:
            metric_results['Lagged Selected Dealer'] = None

        results[metric_name] = metric_results
    return results

def grouped_regression(df, x_col, y_col, group):
    results = {}
    # Group by the grouping variable and run a regression on each group
    for grp_val, sub_df in df.groupby(group):
        # Normalize the key (if it’s a string)
        normalized_key = grp_val.strip().upper() if isinstance(grp_val, str) else grp_val
        # Only perform regression if there are enough data points (e.g. > 1 row)
        if sub_df.shape[0] > 1:
            reg_result = sh.simple_linear_regression(sub_df, x_col, y_col)
            results[normalized_key] = reg_result
        else:
            results[normalized_key] = None
    return results

## Define state,nodes and graphs

@dataclass
class StateType(Dict[str, Any]):
    context: Annotated[Dict[str, Any], operator.or_]

#@weave.op()
def execprompt(payload, promptnm):
    graph = prompt_executor_graph()
    print("Prompt executor graph created successfully.")
    output = graph.invoke(payload)
    aimsg = output["context"][promptnm] 
    return aimsg.dict()["content"]

#@weave.op()
def get_snowflake_data_node(state: StateType) -> StateType:
    sfgraph = create_dashboard_graph()
    output = sfgraph.invoke(state)
    return output

#@weave.op()
def missing_channels_node(state: StateType) -> StateType:
    table_dfs = deserialize_output(state)
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    jira_id = table_dfs["df_basic"]["JIRA_ID"][0]
    df_channels_js = table_dfs["df_missing_channels"].to_json()
    promptnm="fusion_insights_missing_channels"
    missing_channels_payload = {"context": {"dealer_name": dealer_nm, "jira_id": jira_id,"df_missing_channels": df_channels_js,"prompt_name":promptnm}}
    output=execprompt(missing_channels_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:missing_channels_payload}
    print("Missing channels content:",result)
    return {"context": result}

#@weave.op()
def mystery_shops(state: StateType) -> StateType:
    table_dfs = deserialize_output(state)
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    df_mystery_monthly_js = table_dfs["df_mystery_monthly"].to_json()
    promptnm="fusion_insights_mysteryshop" 
    mystery_shops_payload = {"context": {"dealer_name": dealer_nm, "df_mystery_monthly": df_mystery_monthly_js,"prompt_name":promptnm}}
    output=execprompt(mystery_shops_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:mystery_shops_payload}
    logger.info("generated mystery shops content")
    return {"context": result}

#@weave.op()
def budget_efficiency(state: StateType) -> StateType:
    table_dfs = deserialize_output(state)   
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    state_nm = table_dfs["df_basic"]["STATE"][0]
    brand_nm = table_dfs["df_basic"]["BRAND"][0]
    start_date = state["context"]["start_date"]
    end_date = state["context"]["end_date"]
    ai_insight_lb_limit = state["context"]["ai_insight_lb_limit"]
    df_budget_lb_state_brand_all_js = table_dfs["df_budget_lb_state_brand_all"].to_json()
    df_budget_lb_state_limit_js = table_dfs["df_budget_lb_state_limit"].to_json()   
    df_budget_lb_brand_limit_js = table_dfs["df_budget_lb_brand_limit"].to_json()   

    promptnm="fusion_insights_budget_efficiency" 
    budget_efficiency_payload = {"context": {"dealer_name": dealer_nm, "state": state_nm, "brand": brand_nm,
    "start_date": start_date,
    "end_date": end_date,
    "ai_insight_lb_limit": ai_insight_lb_limit,
    "df_budget_lb_state_brand_all": df_budget_lb_state_brand_all_js,
    "df_budget_lb_state_limit": df_budget_lb_state_limit_js,
    "df_budget_lb_brand_limit": df_budget_lb_brand_limit_js,
    "prompt_name":promptnm}}
    output=execprompt(budget_efficiency_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:budget_efficiency_payload}
    logger.info("generated budget efficiency content")
    return {"context": result}

#@weave.op()
def inventory_turnover(state: StateType) -> StateType:
    table_dfs = deserialize_output(state)
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    state_nm = table_dfs["df_basic"]["STATE"][0]
    brand_nm = table_dfs["df_basic"]["BRAND"][0]
    start_date = state["context"]["start_date"]
    end_date = state["context"]["end_date"]
    ai_insight_lb_limit = state["context"]["ai_insight_lb_limit"]
    df_inventory_lb_state_brand_all_js = table_dfs["df_inventory_lb_state_brand_all"].to_json()
    df_inventory_lb_state_limit_js = table_dfs["df_inventory_lb_state_limit"].to_json()   
    df_inventory_lb_brand_limit_js = table_dfs["df_inventory_lb_brand_limit"].to_json()   

    promptnm="fusion_insights_inventory_turnover" 
    inventory_turnover_payload = {"context": {"dealer_name": dealer_nm, "state": state_nm, "brand": brand_nm,
    "start_date": start_date,
    "end_date": end_date,
    "ai_insight_lb_limit": ai_insight_lb_limit,
    "df_inventory_lb_state_brand_all": df_inventory_lb_state_brand_all_js,
    "df_inventory_lb_state_limit": df_inventory_lb_state_limit_js,
    "df_inventory_lb_brand_limit": df_inventory_lb_brand_limit_js,
    "prompt_name":promptnm}}
    output=execprompt(inventory_turnover_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:inventory_turnover_payload }
    logger.info("generated inventory turnover content")
    return {"context": result}

#@weave.op()
def inventory_trends(state: StateType)-> StateType:
    table_dfs = deserialize_output(state)
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    df_monthly_inv = table_dfs["df_monthly_inv"]
    if not df_monthly_inv.empty:
        new_inv_col, used_inv_col = get_inventory_cols(df_monthly_inv)
        if new_inv_col and used_inv_col:
            df_monthly_inv["TOTAL_INVENTORY"] = df_monthly_inv[new_inv_col] + \
                df_monthly_inv[used_inv_col]
        else:
            logging.error("Inventory columns not found in monthly inventory data.")

    df_monthly_inv_fmt = df_monthly_inv[['MONTH', 'YEAR', 'TOTAL_INVENTORY', 'NEW_SALES_TOTAL', 'USED_SALES_TOTAL']].to_string(
        index=False) if not df_monthly_inv.empty else "N/A"
    
    promptnm="fusion_insights_inventory_trends"
    inventory_trend_payload = {"context": {"dealer_name": dealer_nm, "df_monthly_inv_fmt": df_monthly_inv_fmt,  "prompt_name":promptnm}}
    output=execprompt(inventory_trend_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:inventory_trend_payload }
    logger.info("generated inventory turnover content")
    return {"context": result}

#@weave.op()
def marketing_spend_v_sales_trend(state: StateType):
    table_dfs = deserialize_output(state)
    dealer_nm = table_dfs["df_basic"]["DEALERSHIP_NAME"][0]
    df_monthly_ads = table_dfs["df_monthly_ads"]
    df_monthly_inv = table_dfs["df_monthly_inv"]

    spend_column = None
    if not df_monthly_ads.empty:
        df_monthly_ads.columns = df_monthly_ads.columns.str.upper()
        df_monthly_ads["DATE"] = pd.to_datetime(df_monthly_ads["YEAR"].astype(
            str) + "-" + df_monthly_ads["MONTH"].astype(str), format="%Y-%m")

    if not df_monthly_inv.empty:
        df_monthly_inv.columns = df_monthly_inv.columns.str.upper()
        df_monthly_inv["DATE"] = pd.to_datetime(df_monthly_inv["YEAR"].astype(
            str) + "-" + df_monthly_inv["MONTH"].astype(str), format="%Y-%m")

    for col in df_monthly_ads.columns:
        if "SPEND" in col.upper():
            spend_column = col
            break
    
    merged_data = pd.merge(
        df_monthly_ads[['DATE', spend_column]],
        df_monthly_inv[[
            'DATE', 'NEW_SALES_TOTAL', 'USED_SALES_TOTAL']],
        on='DATE',
        how='inner'
    )
    merged_data["TOTAL_SALES"] = merged_data["NEW_SALES_TOTAL"] + \
        merged_data["USED_SALES_TOTAL"]

    promptnm="fusion_insights_spend_v_sales"
    spend_v_sales_payload = {"context": {"dealer_name": dealer_nm, "merged_data": merged_data.to_json(), "prompt_name":promptnm}}
    output=execprompt(spend_v_sales_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:spend_v_sales_payload}
    logger.info("generated marketing_spend_v_sales_trend content")
    return {"context": result}

#@weave.op()
def inventory_trend_analysis(state: StateType) -> StateType:
    table_dfs = deserialize_output(state)
    df_monthly_all = table_dfs["df_monthly_all"]
    df_basic = table_dfs["df_basic"]
    df_inv_agg = table_dfs["df_inv_agg"]
    df_monthly_inv = table_dfs["df_monthly_inv"]
    
    dealer_name= df_basic["DEALERSHIP_NAME"][0]
    brand_name = df_basic["BRAND"][0]
    jira_id = df_basic["JIRA_ID"][0]
    start_date = state["context"]["start_date"]
    end_date = state["context"]["end_date"]

    if not df_monthly_inv.empty:
        df_monthly_inv.columns = df_monthly_inv.columns.str.upper()
        df_monthly_inv["DATE"] = pd.to_datetime(df_monthly_inv["YEAR"].astype(
            str) + "-" + df_monthly_inv["MONTH"].astype(str), format="%Y-%m")
        new_inv_col, used_inv_col = get_inventory_cols(df_monthly_inv)
        if new_inv_col and used_inv_col:
            df_monthly_inv["TOTAL_INVENTORY"] = df_monthly_inv[new_inv_col] + \
                df_monthly_inv[used_inv_col]
    else:
        print("Inventory columns not found in monthly inventory data.")



    if not df_inv_agg.empty:
        df_inv_agg.columns = df_inv_agg.columns.str.upper()
        avg_sales = df_inv_agg["AVG_SALES"].iloc[0] if "AVG_SALES" in df_inv_agg.columns else 0
        avg_inventory = df_inv_agg["AVG_INVENTORY"].iloc[0] if "AVG_INVENTORY" in df_inv_agg.columns else 0
    else:
        avg_sales = 0
        avg_inventory = 0

    if "MONTHLY_INVENTORY" not in df_monthly_all.columns:
        new_col_all, used_col_all = get_inventory_cols(df_monthly_all)
        if new_col_all and used_col_all:
            df_monthly_all["MONTHLY_INVENTORY"] = df_monthly_all[new_col_all] + \
            df_monthly_all[used_col_all]
    else:
        print("Inventory columns not found in all dealers monthly data.")
        #return
    # Also ensure that TOTAL_SALES exists (as returned by your query)
    if "TOTAL_SALES" not in df_monthly_all.columns:
        print("Sales columns not found in all dealers monthly data.")
        #return

    df_monthly_inv["TOTAL_INVENTORY"] = df_monthly_inv["NEW_INVENTORY_AVERAGE"] + \
                df_monthly_inv["USED_INVENTORY_AVERAGE"]
            
    if "STATE" in df_basic.columns:
        state_val = df_basic["STATE"].iloc[0]
        df_state = df_monthly_all[df_monthly_all["STATE"] == state_val]
        state_inv = df_state["MONTHLY_INVENTORY"].mean() if not df_state.empty else None
        state_sales = df_state["TOTAL_SALES"].mean() if not df_state.empty else None
        state_ratio = (state_sales / state_inv) if state_inv and state_inv > 0 else None
    else:
        state_inv = state_sales = state_ratio = None

    if "BRAND" in df_basic.columns:
        brand_val = df_basic["BRAND"].iloc[0]
        df_brand = df_monthly_all[df_monthly_all["BRAND"] == brand_val]
        brand_inv = df_brand["MONTHLY_INVENTORY"].mean() if not df_brand.empty else None
        brand_sales = df_brand["TOTAL_SALES"].mean() if not df_brand.empty else None
        brand_ratio = (brand_sales / brand_inv) if brand_inv and brand_inv > 0 else None
    else:
        brand_inv = brand_sales = brand_ratio = None

    curr_inv = avg_inventory
    curr_sales = avg_sales
    #curr_ratio = (curr_inv / curr_sales) if curr_sales and curr_sales > 0 else None
    curr_ratio = (curr_sales / curr_inv) if curr_inv and curr_inv > 0 else None


    comparison_data = {
        "Metric": ["Avg Monthly Inventory", "Avg Monthly Sales", "Inventory-to-Sales Ratio"],
        "Current Dealer": [
            f"{curr_inv:.0f} units" if curr_inv is not None else "N/A",
            f"{curr_sales:.0f} vehicles" if curr_sales is not None else "N/A",
            f"{curr_ratio:.1f}:1" if curr_ratio is not None else "N/A"
        ],
        "State Avg": [
            f"{state_inv:.0f} units" if state_inv is not None else "N/A",
            f"{state_sales:.0f} vehicles" if state_sales is not None else "N/A",
            f"{state_ratio:.1f}:1" if state_ratio is not None else "N/A"
        ],
        "Brand Avg": [
            f"{brand_inv:.0f} units" if brand_inv is not None else "N/A",
            f"{brand_sales:.0f} vehicles" if brand_sales is not None else "N/A",
            f"{brand_ratio:.1f}:1" if brand_ratio is not None else "N/A"
        ]
    }
    df_comparison = pd.DataFrame(comparison_data)


    metrics_inventory = {"Inventory": "MONTHLY_INVENTORY"}
    sales_col = "TOTAL_SALES"  # Reusing the sales column from before
    grouping_columns = ["STATE", "BRAND"]

    # Perform dynamic analysis for Inventory vs Sales
    inventory_vs_sales_analysis = dynamic_metric_analysis(
    metrics_inventory,
    df_monthly_all.copy(),
    sales_col,
    grouping_columns,
    jira_id,
    replace_zero=True)

    overall_reg = inventory_vs_sales_analysis["Inventory"]["Overall Monthly Regression"]
    extra_inventory_for_one_sale = -1
    if overall_reg and "params" in overall_reg and "MONTHLY_INVENTORY" in overall_reg["params"]:
        inv_slope = overall_reg["params"]["MONTHLY_INVENTORY"]
        if inv_slope and inv_slope != 0:
            extra_inventory_for_one_sale = 1 / inv_slope
            overall_insight = f"Overall Insight: An additional {extra_inventory_for_one_sale:,.0f} units in inventory could yield ~1 extra sale."
        else:
            overall_insight = "Overall Insight: Regression slope is zero, so additional inventory is not expected to increase sales."
    else:
        inv_slope = None
        overall_insight = "Overall Insight: No overall regression parameters available."

    state_val_norm = state_val.strip().upper() if state_val and isinstance(state_val, str) else None
    brand_val_norm = brand_val.strip().upper() if brand_val and isinstance(brand_val, str) else None

    state_regressions_inv = inventory_vs_sales_analysis["Inventory"]["Grouped Monthly"].get("regressions", {}).get("STATE", {})
    state_regressions_inv_norm = {k.strip().upper(): v for k, v in state_regressions_inv.items() if isinstance(k, str)}
    state_inv_result = state_regressions_inv_norm.get(state_val_norm, None) if state_val_norm else None

    if state_val_norm:
        if state_inv_result and "params" in state_inv_result and "MONTHLY_INVENTORY" in state_inv_result["params"]:
            state_inv_slope = state_inv_result["params"]["MONTHLY_INVENTORY"]
            if state_inv_slope and state_inv_slope != 0:
                extra_inventory_state = 1 / state_inv_slope
                state_inventory_insight = f"State Inventory Insight: An additional {extra_inventory_state:,.0f} units in inventory could yield ~1 extra sale in {state_val_norm.title()}."
            else:
                state_inventory_insight = f"State Inventory Insight: Regression slope for inventory in state {state_val_norm.title()} is zero or not available."
        else:
            state_inventory_insight = f"State Inventory Insight: No regression parameters available for inventory in state {state_val_norm.title()}."
    else:
        state_inventory_insight = "State Inventory Insight: State information is not available."

    # --- Brand-level Insight ---
    brand_regressions_inv = inventory_vs_sales_analysis["Inventory"]["Grouped Monthly"].get("regressions", {}).get("BRAND", {})
    brand_regressions_inv_norm = {k.strip().upper(): v for k, v in brand_regressions_inv.items() if isinstance(k, str)}
    brand_inv_result = brand_regressions_inv_norm.get(brand_val_norm, None) if brand_val_norm else None

    if brand_val_norm:
        if brand_inv_result and "params" in brand_inv_result and "MONTHLY_INVENTORY" in brand_inv_result["params"]:
            brand_inv_slope = brand_inv_result["params"]["MONTHLY_INVENTORY"]
            if brand_inv_slope and brand_inv_slope != 0:
                extra_inventory_brand = 1 / brand_inv_slope
                brand_inventory_insight = f"Brand Inventory Insight: An additional {extra_inventory_brand:,.0f} units in inventory could yield ~1 extra sale for brand {brand_val_norm.title()}."
            else:
                brand_inventory_insight = f"Brand Inventory Insight: Regression slope for inventory for brand {brand_val_norm.title()} is zero or not available."
        else:
            brand_inventory_insight = f"Brand Inventory Insight: No regression parameters available for inventory for brand {brand_val_norm.title()}."
    else:
        brand_inventory_insight = "Brand Inventory Insight: Brand information is not available."

    curr_ratio_fmt = f"{curr_ratio:.1f}:1" if curr_ratio is not None else "N/A"
    inventory_vs_sales_correlation = inventory_vs_sales_analysis["Inventory"]["Overall Monthly Correlation"]
    overall_reg = overall_reg["params"]["MONTHLY_INVENTORY"] if overall_reg and "params" in overall_reg and "MONTHLY_INVENTORY" in overall_reg["params"] else "N/A"
    extra_inventory_for_one_sale =extra_inventory_for_one_sale if inv_slope and inv_slope != 0 else "N/A"

    monthly_total_inventory = df_monthly_inv[['DATE', 'TOTAL_INVENTORY']].to_string(index=False) if not df_monthly_inv.empty else "N/A"

    promptnm="fusion_insights_inventory_analysis"

    curr_ratio_fmt = f"{curr_ratio:.1f}:1" if curr_ratio is not None else "N/A"

    inventory_analysis_payload = {"context": {"start_date": start_date,"end_date": end_date,"dealer_name": dealer_name, "brand": brand_name,
    "avg_sales": avg_sales,"avg_inventory": avg_inventory,"curr_inv": curr_inv,"curr_sales": curr_sales,"curr_ratio_fmt":curr_ratio_fmt,
    "state_inv":state_inv,"state_sales":state_sales,"state_ratio":state_ratio,"brand_inv":brand_inv,"brand_sales":brand_sales,
    "brand_ratio":brand_ratio,"monthly_total_inventory":monthly_total_inventory,"inventory_vs_sales_correlation":inventory_vs_sales_correlation,
    "overall_reg":overall_reg,"extra_inventory_for_one_sale":extra_inventory_for_one_sale,"overall_insight":overall_insight,
    "state_inventory_insight":state_inventory_insight,"brand_inventory_insight":brand_inventory_insight,"prompt_name":promptnm}}
    
    output=execprompt(inventory_analysis_payload,promptnm)  
    result = {promptnm:output, 'var'+promptnm:inventory_analysis_payload}
    logger.info("generated inventory analysis content")
    return {"context": result}

#@weave.op()
def budget_spend_analysis(state: StateType):
    table_dfs = deserialize_output(state)   
    df_monthly_all = table_dfs["df_monthly_all"]
    df_basic = table_dfs["df_basic"]
    dealer_name = df_basic["DEALERSHIP_NAME"][0]
    brand_name = df_basic["BRAND"][0]
    df_detailed_budget = table_dfs["df_detailed_budget"]
    df_inv_agg = table_dfs["df_inv_agg"]
    df_ads_agg = table_dfs["df_ads_agg"]
    df_budget_avg = table_dfs["df_budget_avg"]

    jira_id = state["context"]["jira_id"]
    start_date = state["context"]["start_date"]
    end_date = state["context"]["end_date"]
    metrics_budget = {"Budget": "MONTHLY_BUDGET"}
    sales_col = "TOTAL_SALES"
    grouping_columns = ["STATE", "BRAND"]

    if not df_inv_agg.empty:
        df_inv_agg.columns = df_inv_agg.columns.str.upper()
        avg_sales = df_inv_agg["AVG_SALES"].iloc[0] if "AVG_SALES" in df_inv_agg.columns else 0
        avg_inventory = df_inv_agg["AVG_INVENTORY"].iloc[0] if "AVG_INVENTORY" in df_inv_agg.columns else 0
    else:
        avg_sales = 0
        avg_inventory = 0

    if not df_ads_agg.empty:
        df_ads_agg.columns = df_ads_agg.columns.str.upper()
        avg_spend = df_ads_agg["AVG_AD_SPEND"].iloc[0] if "AVG_AD_SPEND" in df_ads_agg.columns else 0
    else:
        avg_spend = 0

    avg_budget = df_budget_avg["AVG_MONTHLY_BUDGET"].iloc[0] if not df_budget_avg.empty else 0

    budget_vs_sales_analysis = dynamic_metric_analysis(
        metrics_budget,
        df_monthly_all.copy(),
        sales_col,
        grouping_columns,
        jira_id,
        replace_zero=True
    )


    overall_monthly_budget_regression = budget_vs_sales_analysis["Budget"]["Overall Monthly Regression"]
    extra_spend_for_one_sale = None
    if "params" in overall_monthly_budget_regression:
        slope = overall_monthly_budget_regression["params"]["MONTHLY_BUDGET"]
        if slope and slope != 0:
            extra_spend_for_one_sale = 1 / slope
            extra_spend_fmt = f"**Insight:** A ${extra_spend_for_one_sale:,.0f} increase in spend could yield ~1 extra sale."
        else:
            extra_spend_fmt = "**Insight:** Regression slope is zero, so additional spend is not expected to increase sales."
    else:
        extra_spend_fmt = "No overall monthly budget regression parameters available"

    # Normalize state and brand values by stripping extra whitespace and converting to uppercase.
    state_raw = df_basic["STATE"].iloc[0] if "STATE" in df_basic.columns else None
    brand_raw = df_basic["BRAND"].iloc[0] if "BRAND" in df_basic.columns else None

    state_val = state_raw.strip().upper() if state_raw and isinstance(state_raw, str) else None
    brand_val = brand_raw.strip().upper() if brand_raw and isinstance(brand_raw, str) else None

    # State-level insight using the new grouped regressions
    state_regressions = budget_vs_sales_analysis["Budget"]["Grouped Monthly"].get("regressions", {}).get("STATE", {})
    # Normalize keys in state_regressions:
    state_regressions_norm = {k.strip().upper(): v for k, v in state_regressions.items() if isinstance(k, str)}
    state_reg_result = state_regressions_norm.get(state_val, None) if state_val else None

    if state_val:
        if state_reg_result and "params" in state_reg_result and "MONTHLY_BUDGET" in state_reg_result["params"]:
            state_slope = state_reg_result["params"]["MONTHLY_BUDGET"]
            if state_slope and state_slope != 0:
                extra_spend_state = 1 / state_slope
                state_insight = f"A ${extra_spend_state:,.0f} increase in spend could yield ~1 extra sale in {state_val.title()}."
            else:
                state_insight = f"Regression slope for state {state_val.title()} is zero or not available."
        else:
            state_insight = f"No regression parameters available for state {state_val.title()}"
    else:
        state_insight = "State information is not available."

    # Brand-level insight using the new grouped regressions
    brand_regressions = budget_vs_sales_analysis["Budget"]["Grouped Monthly"].get("regressions", {}).get("BRAND", {})
    # Normalize keys for brand:
    brand_regressions_norm = {k.strip().upper(): v for k, v in brand_regressions.items() if isinstance(k, str)}
    brand_reg_result = brand_regressions_norm.get(brand_val, None) if brand_val else None

    if brand_val:
        if brand_reg_result and "params" in brand_reg_result and "MONTHLY_BUDGET" in brand_reg_result["params"]:
            brand_slope = brand_reg_result["params"]["MONTHLY_BUDGET"]
            if brand_slope and brand_slope != 0:
                extra_spend_brand = 1 / brand_slope
                brand_insight = f"A ${extra_spend_brand:,.0f} increase in spend could yield ~1 extra sale for brand {brand_val.title()}."
            else:
                brand_insight = f"Regression slope for brand {brand_val.title()} is zero or not available."
        else:
            brand_insight = f"No regression parameters available for brand {brand_val.title()}"
    else:
        brand_insight = "Brand information is not available."

    if df_detailed_budget.empty:
        mean_budget_fmt = "N/A"
        median_budget_fmt = "N/A"
        range_budget_fmt = "N/A"
    else:
        mean_budget_fmt = f"${df_detailed_budget['TOTAL_CLIENT_BUDGET'].mean():,.0f}"
        median_budget_fmt = f"${df_detailed_budget['TOTAL_CLIENT_BUDGET'].median():,.0f}"
        range_budget_fmt = f"${df_detailed_budget['TOTAL_CLIENT_BUDGET'].min():,.0f} - ${df_detailed_budget['TOTAL_CLIENT_BUDGET'].max():,.0f}"

    df_detailed_budget_js = df_detailed_budget.to_json()
    overall_reg_slope_budget = budget_vs_sales_analysis["Budget"]["Overall Monthly Regression"]["params"]["MONTHLY_BUDGET"] if budget_vs_sales_analysis["Budget"]["Overall Monthly Regression"].get("params") else "N/A"
    overall_corr_budget = budget_vs_sales_analysis["Budget"]["Overall Monthly Correlation"]
    
    promptnm="fusion_insights_spend_analysis"
    budget_spend_analysis_payload = {"context": {"start_date": start_date,"end_date": end_date,"dealer_name": dealer_name, "brand": brand_name,
    "avg_spend": avg_spend,"avg_budget": avg_budget,"avg_sales": avg_sales,"df_detailed_budget_js": df_detailed_budget_js,
    "mean_budget_fmt": mean_budget_fmt,"median_budget_fmt": median_budget_fmt,"range_budget_fmt": range_budget_fmt,
    "overall_reg_slope_budget": overall_reg_slope_budget,"overall_corr_budget": overall_corr_budget,"extra_spend_fmt": extra_spend_fmt,
    "state_insight": state_insight,"brand_insight": brand_insight,"prompt_name":promptnm}}
    
    output=execprompt(budget_spend_analysis_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:budget_spend_analysis_payload }
    logger.info("generated budget spend analysis content")
    return {"context": result}

#@weave.op()
def Website_traffic_analysis(state: StateType)->StateType:
    table_dfs = deserialize_output(state)   
    df_basic = table_dfs["df_basic"]
    df_inv_agg = table_dfs["df_inv_agg"]
    df_monthly_ads = table_dfs["df_monthly_ads"]
    df_ga_agg = table_dfs["df_ga_agg"]
    df_monthly_ga = table_dfs["df_monthly_ga"]

    dealer_name = df_basic["DEALERSHIP_NAME"][0]
    brand = df_basic["BRAND"][0]
    jira_id = state["context"]["jira_id"]
    start_date = state["context"]["start_date"]
    end_date = state["context"]["end_date"]

    df_monthly_all = table_dfs["df_monthly_all"]

    metrics_traffic = {"Website Traffic": "TOTAL_SESSIONS"}
    sales_col = "TOTAL_SALES"  # Reusing the sales column from before
    grouping_columns = ["STATE", "BRAND"]
    if not df_inv_agg.empty:
        df_inv_agg.columns = df_inv_agg.columns.str.upper()
        avg_sales = df_inv_agg["AVG_SALES"].iloc[0] if "AVG_SALES" in df_inv_agg.columns else 0
        avg_inventory = df_inv_agg["AVG_INVENTORY"].iloc[0] if "AVG_INVENTORY" in df_inv_agg.columns else 0
    else:
        avg_sales = 0
        avg_inventory = 0

    if not df_monthly_ads.empty:
        df_monthly_ads.columns = df_monthly_ads.columns.str.upper()
        df_monthly_ads["DATE"] = pd.to_datetime(df_monthly_ads["YEAR"].astype(str) + "-" + df_monthly_ads["MONTH"].astype(str), format="%Y-%m")
    if not df_ga_agg.empty:
        df_ga_agg.columns = df_ga_agg.columns.str.upper()
        avg_sessions = df_ga_agg["AVG_SESSIONS"].iloc[0] if "AVG_SESSIONS" in df_ga_agg.columns else 0
    else:
        avg_sessions = 0
    if not df_monthly_ga.empty:
        df_monthly_ga.columns = df_monthly_ga.columns.str.upper()
        df_monthly_ga["DATE"] = pd.to_datetime(df_monthly_ga["YEAR"].astype(str) + "-" + df_monthly_ga["MONTH"].astype(str), format="%Y-%m")

    # Perform dynamic analysis for Website Traffic vs Sales
    traffic_vs_sales_analysis = dynamic_metric_analysis(
        metrics_traffic,
        df_monthly_all.copy(),
        sales_col,
        grouping_columns,
        jira_id,
        replace_zero=True
    )

    # --- Overall Traffic Insight Calculation ---
    overall_reg = traffic_vs_sales_analysis["Website Traffic"]["Overall Monthly Regression"]
    traffic_slope = None
    extra_traffic_for_one_sale = -1
    if overall_reg and "params" in overall_reg and "TOTAL_SESSIONS" in overall_reg["params"]:
        traffic_slope = overall_reg["params"]["TOTAL_SESSIONS"]
        if traffic_slope and traffic_slope != 0:
            extra_traffic_for_one_sale = 1 / traffic_slope
            overall_traffic_insight = f"Overall Traffic Insight: An additional {extra_traffic_for_one_sale:,.0f} website sessions could yield ~1 extra sale."
        else:
            overall_traffic_insight = "Overall Traffic Insight: Regression slope is zero, so additional website traffic is not expected to increase sales."
    else:
        overall_traffic_insight = "Overall Traffic Insight: No overall regression parameters available."

    # Normalize state and brand values by stripping extra whitespace and converting to uppercase.
    state_raw = df_basic["STATE"].iloc[0] if "STATE" in df_basic.columns else None
    brand_raw = df_basic["BRAND"].iloc[0] if "BRAND" in df_basic.columns else None

    state_val = state_raw.strip().upper() if state_raw and isinstance(state_raw, str) else None
    brand_val = brand_raw.strip().upper() if brand_raw and isinstance(brand_raw, str) else None

    # --- Normalize state and brand values for grouped lookup ---
    state_val_norm = state_val.strip().upper() if state_val and isinstance(state_val, str) else None
    brand_val_norm = brand_val.strip().upper() if brand_val and isinstance(brand_val, str) else None

    # --- State-level Traffic Insight ---
    state_regressions_traffic = traffic_vs_sales_analysis["Website Traffic"]["Grouped Monthly"].get("regressions", {}).get("STATE", {})
    state_regressions_traffic_norm = {k.strip().upper(): v for k, v in state_regressions_traffic.items() if isinstance(k, str)}
    state_traffic_result = state_regressions_traffic_norm.get(state_val_norm, None) if state_val_norm else None

    if state_val_norm:
        if state_traffic_result and "params" in state_traffic_result and "TOTAL_SESSIONS" in state_traffic_result["params"]:
            state_traffic_slope = state_traffic_result["params"]["TOTAL_SESSIONS"]
            if state_traffic_slope and state_traffic_slope != 0:
                extra_traffic_state = 1 / state_traffic_slope
                state_traffic_insight = f"State Traffic Insight: An additional {extra_traffic_state:,.0f} website sessions could yield ~1 extra sale in {state_val_norm.title()}."
            else:
                state_traffic_insight = f"State Traffic Insight: Regression slope for website traffic in state {state_val_norm.title()} is zero or not available."
        else:
            state_traffic_insight = f"State Traffic Insight: No regression parameters available for website traffic in state {state_val_norm.title()}."
    else:
        state_traffic_insight = "State Traffic Insight: State information is not available."

    # --- Brand-level Traffic Insight ---
    brand_regressions_traffic = traffic_vs_sales_analysis["Website Traffic"]["Grouped Monthly"].get("regressions", {}).get("BRAND", {})
    brand_regressions_traffic_norm = {k.strip().upper(): v for k, v in brand_regressions_traffic.items() if isinstance(k, str)}
    brand_traffic_result = brand_regressions_traffic_norm.get(brand_val_norm, None) if brand_val_norm else None

    if brand_val_norm:
        if brand_traffic_result and "params" in brand_traffic_result and "TOTAL_SESSIONS" in brand_traffic_result["params"]:
            brand_traffic_slope = brand_traffic_result["params"]["TOTAL_SESSIONS"]
            if brand_traffic_slope and brand_traffic_slope != 0:
                extra_traffic_brand = 1 / brand_traffic_slope
                brand_traffic_insight = f"Brand Traffic Insight: An additional {extra_traffic_brand:,.0f} website sessions could yield ~1 extra sale for brand {brand_val_norm.title()}."
            else:
                brand_traffic_insight = f"Brand Traffic Insight: Regression slope for website traffic for brand {brand_val_norm.title()} is zero or not available."
        else:
            brand_traffic_insight = f"Brand Traffic Insight: No regression parameters available for website traffic for brand {brand_val_norm.title()}."
    else:
        brand_traffic_insight = "Brand Traffic Insight: Brand information is not available."

    df_dealer_stats = df_monthly_all[df_monthly_all["JIRA_ID"] == jira_id].copy()
    df_dealer_stats["TRAFFIC_TO_SALES_RATIO"] = df_dealer_stats["TOTAL_SESSIONS"] / \
        df_dealer_stats["TOTAL_SALES"].clip(lower=1)

    # Calculate the average traffic needed per sale for the current dealer
    dealer_avg_traffic_per_sale = df_dealer_stats["TRAFFIC_TO_SALES_RATIO"].mean()

    # Calculate state and brand averages for comparison
    state_data = df_monthly_all[df_monthly_all["STATE"] == state_val].copy()
    state_data["TRAFFIC_TO_SALES_RATIO"] = state_data["TOTAL_SESSIONS"] / \
    state_data["TOTAL_SALES"].clip(lower=1)
    state_avg_traffic_per_sale = state_data["TRAFFIC_TO_SALES_RATIO"].mean()

    brand_data = df_monthly_all[df_monthly_all["BRAND"] == brand_val].copy()
    brand_data["TRAFFIC_TO_SALES_RATIO"] = brand_data["TOTAL_SESSIONS"] / \
    brand_data["TOTAL_SALES"].clip(lower=1)
    brand_avg_traffic_per_sale = brand_data["TRAFFIC_TO_SALES_RATIO"].mean()

    # Create comparison table
    traffic_comparison_data = {
    "Metric": ["Sessions Per Sale"],
    "Current Dealer": [
        f"{dealer_avg_traffic_per_sale:.1f} sessions" if not pd.isna(
            dealer_avg_traffic_per_sale) else "N/A"
        ],
    "State Avg": [
        f"{state_avg_traffic_per_sale:.1f} sessions" if not pd.isna(
            state_avg_traffic_per_sale) else "N/A"
        ],
    "Brand Avg": [
        f"{brand_avg_traffic_per_sale:.1f} sessions" if not pd.isna(
            brand_avg_traffic_per_sale) else "N/A"
        ]
    }
    df_traffic_efficiency = pd.DataFrame(traffic_comparison_data)  # Different variable name!

    traffic_monthly_corr = traffic_vs_sales_analysis["Website Traffic"]["Overall Monthly Correlation"]
    over_reg_total_sessions = overall_reg["params"]["TOTAL_SESSIONS"] if overall_reg and "params" in overall_reg and "TOTAL_SESSIONS" in overall_reg["params"] else "N/A"
    extra_traffic_for_one_sale = extra_traffic_for_one_sale if traffic_slope and traffic_slope != 0 else "N/A"

    promptnm="fusion_insights_traffic_analysis"
    website_traffic_analysis_payload = {"context": {"start_date": start_date,"end_date": end_date,"dealer_name": dealer_name, "brand": brand,
    "avg_sales": avg_sales,"avg_sessions": avg_sessions,"dealer_avg_traffic_per_sale": dealer_avg_traffic_per_sale,
    "state_avg_traffic_per_sale": state_avg_traffic_per_sale,"brand_avg_traffic_per_sale": brand_avg_traffic_per_sale,
    "traffic_monthly_corr": traffic_monthly_corr,"over_reg_total_sessions": over_reg_total_sessions,"extra_traffic_for_one_sale": extra_traffic_for_one_sale,
    "overall_traffic_insight": overall_traffic_insight,"state_traffic_insight": state_traffic_insight,
    "brand_traffic_insight": brand_traffic_insight,"prompt_name":promptnm}}
    output=execprompt(website_traffic_analysis_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:website_traffic_analysis_payload}
    logger.info("Website traffic content generated")
    return {"context": result}

#@weave.op()
def get_channel_analysis(state: StateType)->StateType:
    table_dfs = deserialize_output(state)   
    df_monthly_all = table_dfs["df_monthly_all"]
    df_ga_agg = table_dfs["df_ga_agg"]
    df_missing_channels = table_dfs["df_missing_channels"]
    df_traffic_comparison = table_dfs["df_traffic_comparison"]
    df_basic = table_dfs["df_basic"]
    dealer_name = df_basic["DEALERSHIP_NAME"][0]
    brand = df_basic["BRAND"][0]
    jira_id = df_basic["JIRA_ID"][0]
    if not df_ga_agg.empty:
        df_ga_agg.columns = df_ga_agg.columns.str.upper()
        avg_sessions = df_ga_agg["AVG_SESSIONS"].iloc[0] if "AVG_SESSIONS" in df_ga_agg.columns else 0
    else:
        avg_sessions = 0

    df_dealer_stats = df_monthly_all[df_monthly_all["JIRA_ID"] == jira_id].copy()
    df_dealer_stats["TRAFFIC_TO_SALES_RATIO"] = df_dealer_stats["TOTAL_SESSIONS"] / \
        df_dealer_stats["TOTAL_SALES"].clip(lower=1)

    # Calculate the average traffic needed per sale for the current dealer
    dealer_avg_traffic_per_sale = df_dealer_stats["TRAFFIC_TO_SALES_RATIO"].mean()

    avg_sessions_fmt = f"{avg_sessions:,.0f}" if avg_sessions is not None else "N/A"
    dealer_avg_traffic_per_sale_fmt = f"{dealer_avg_traffic_per_sale:.1f}" if dealer_avg_traffic_per_sale is not None else "N/A"
    df_traffic_comparison_js= df_traffic_comparison.to_json(index=False)
    df_missing_channels_js= df_missing_channels.to_json(index=False)
    promptnm = "fusion_insights_channel_performance"

    channel_analysis_payload = {"context": {"dealer_name": dealer_name, "brand": brand,
        "avg_sessions_fmt": avg_sessions_fmt,
        "dealer_avg_traffic_per_sale_fmt": dealer_avg_traffic_per_sale_fmt,
        "df_traffic_comparison_js": df_traffic_comparison_js,
        "df_missing_channels_js": df_missing_channels_js,
        "prompt_name":promptnm}}
        
    output=execprompt(channel_analysis_payload,promptnm)
    result = {promptnm:output, 'var'+promptnm:channel_analysis_payload  }
    logger.info("generated channel analysis content")
    return {"context": result}


def coalesce(state: StateType, config: RunnableConfig) -> Dict[str, Any]:
    """Each node does work."""
    #configuration = config["configurable"]
    logging.info(f"Executing coalesce with configuration: {config}")
    return state

def create_graph()-> StateGraph:
    logger.info("Creating dashboard graph...")

    # Create the workflow graph with state type annotation
    workflow = StateGraph(StateType)
    
    # Add nodes
    workflow.add_node("get_snowflake_data", get_snowflake_data_node)
    workflow.add_node("missing_channels", missing_channels_node)
    workflow.add_node("mystery_shops", mystery_shops)
    workflow.add_node("budget_efficiency", budget_efficiency)
    workflow.add_node("inventory_turnover", inventory_turnover)
    workflow.add_node("inventory_trends", inventory_trends)
    workflow.add_node("marketing_spend_v_sales_trend", marketing_spend_v_sales_trend)
    workflow.add_node("inventory_trend_analysis", inventory_trend_analysis)
    workflow.add_node("budget_spend_analysis", budget_spend_analysis)
    workflow.add_node("website_traffic_analysis", Website_traffic_analysis)
    workflow.add_node("channel_analysis", get_channel_analysis)
    workflow.add_node("coalesce", coalesce)
    
    # Add edges
    workflow.add_edge(START, "get_snowflake_data")
    workflow.add_edge("get_snowflake_data", "missing_channels")
    workflow.add_edge("get_snowflake_data", "mystery_shops")
    workflow.add_edge("get_snowflake_data", "budget_efficiency")
    workflow.add_edge("get_snowflake_data", "inventory_turnover")
    workflow.add_edge("get_snowflake_data", "inventory_trends")
    workflow.add_edge("get_snowflake_data", "marketing_spend_v_sales_trend")
    workflow.add_edge("get_snowflake_data", "inventory_trend_analysis")
    workflow.add_edge("get_snowflake_data", "budget_spend_analysis")
    workflow.add_edge("get_snowflake_data", "website_traffic_analysis")
    workflow.add_edge("get_snowflake_data", "channel_analysis")

    workflow.add_edge("missing_channels", "coalesce")
    workflow.add_edge("mystery_shops", "coalesce")
    workflow.add_edge("budget_efficiency", "coalesce")
    workflow.add_edge("inventory_turnover", "coalesce")
    workflow.add_edge("inventory_trends", "coalesce")
    workflow.add_edge("marketing_spend_v_sales_trend", "coalesce")
    workflow.add_edge("inventory_trend_analysis", "coalesce")
    workflow.add_edge("budget_spend_analysis", "coalesce")
    workflow.add_edge("website_traffic_analysis", "coalesce")
    workflow.add_edge("channel_analysis", "coalesce")
    workflow.add_edge("coalesce", END)
    
    # Compile the workflow into an executable graph
    graph = workflow.compile()
    graph.name = "FusionDealerInsights"  # This defines the custom name in LangSmith
    return graph

#weave.init('c4-dealer-insights')
graph = create_graph()

