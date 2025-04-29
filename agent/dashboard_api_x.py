"""
FastAPI server for the dashboard graph.
"""
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn

from sql_handler import SnowflakeSQLHandler
from dashboard_graph import get_dashboard_data_parallel, DescriptiveDashboardData

app = FastAPI(title="Dashboard Graph API")

# Pydantic models for request/response
class DashboardRequest(BaseModel):
    jira_id: str
    state: str
    brand: str
    start_date: date
    end_date: date
    ai_insight_lb_limit: Optional[int] = 50

class DashboardResponse(BaseModel):
    """Response model that mirrors DescriptiveDashboardData but with serializable types."""
    basic_info: dict
    inventory_data: dict
    marketing_data: dict
    budget_data: dict
    performance_data: dict

def get_sql_handler():
    """Dependency to get SQL handler instance."""
    try:
        handler = SnowflakeSQLHandler()
        yield handler
    finally:
        # Add any cleanup if needed
        pass

@app.post("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    request: DashboardRequest,
    sql_handler: SnowflakeSQLHandler = Depends(get_sql_handler)
) -> DashboardResponse:
    """Get dashboard data using parallel processing."""
    try:
        result: DescriptiveDashboardData = get_dashboard_data_parallel(
            jira_id=request.jira_id,
            state=request.state,
            brand=request.brand,
            start_date=request.start_date,
            end_date=request.end_date,
            ai_insight_lb_limit=request.ai_insight_lb_limit,
            sql_handler=sql_handler
        )
        
        # Convert DataFrames to dictionaries for JSON serialization
        return DashboardResponse(
            basic_info={
                "df_basic": result.df_basic.to_dict(),
                "all_states_df": result.all_states_df.to_dict(),
                "all_brands_df": result.all_brands_df.to_dict()
            },
            inventory_data={
                "df_inv_agg": result.df_inv_agg.to_dict(),
                "df_monthly_inv": result.df_monthly_inv.to_dict(),
                "df_inventory_lb": result.df_inventory_lb.to_dict(),
                "df_inventory_lb_state_brand_all": result.df_inventory_lb_state_brand_all.to_dict(),
                "df_inventory_lb_state_limit": result.df_inventory_lb_state_limit.to_dict(),
                "df_inventory_lb_brand_limit": result.df_inventory_lb_brand_limit.to_dict()
            },
            marketing_data={
                "df_ads_agg": result.df_ads_agg.to_dict(),
                "df_monthly_ads": result.df_monthly_ads.to_dict(),
                "df_ga_agg": result.df_ga_agg.to_dict(),
                "df_monthly_ga": result.df_monthly_ga.to_dict(),
                "df_traffic_comparison": result.df_traffic_comparison.to_dict(),
                "df_website_lb": result.df_website_lb.to_dict()
            },
            budget_data={
                "df_budget": result.df_budget.to_dict(),
                "df_monthly_budget": result.df_monthly_budget.to_dict(),
                "df_budget_avg": result.df_budget_avg.to_dict(),
                "df_detailed_budget": result.df_detailed_budget.to_dict(),
                "df_budget_lb": result.df_budget_lb.to_dict(),
                "df_budget_lb_state_brand_all": result.df_budget_lb_state_brand_all.to_dict(),
                "df_budget_lb_state_limit": result.df_budget_lb_state_limit.to_dict(),
                "df_budget_lb_brand_limit": result.df_budget_lb_brand_limit.to_dict()
            },
            performance_data={
                "df_mystery": result.df_mystery.to_dict(),
                "df_mystery_monthly": result.df_mystery_monthly.to_dict(),
                "df_rank": result.df_rank.to_dict(),
                "df_oem": result.df_oem.to_dict(),
                "df_monthly_all": result.df_monthly_all.to_dict(),
                "df_missing_channels": result.df_missing_channels.to_dict(),
                "df_avg_monthly_sales": result.df_avg_monthly_sales.to_dict()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)
