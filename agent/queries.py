'''
File: queries.py

Predefined SQL queries for dealership data.

Copyright 2025 C-4 Analytics, LLC
'''

import inspect
import re
from inspect import cleandoc
from dataclasses import dataclass
from datetime import date
from functools import wraps
from typing import Any, Callable, Type, TypeVar

import pandas as pd

# Define type variables for better type annotations
T = TypeVar('T')
R = TypeVar('R')
SqlFunc = TypeVar('SqlFunc', bound=Callable[..., str])
QueryFactory = Callable[..., 'DescriptiveQuery']


@dataclass
class QueryResult:
    description: str
    data: pd.DataFrame


class DescriptiveQuery:
    def __init__(self,
                 sql_template: str,
                 description_template: str | Callable[..., str],
                 sql_params: dict[str, Any],
                 desc_params: dict[str, Any],
                 is_positional: bool = False):
        """
        Initialize a DescriptiveQuery with SQL template and parameters.

        Args:
            sql_template: SQL template string with placeholders
            description_template: Description template string
            sql_params: Parameters to be used in SQL execution
            desc_params: Parameters to be used in description formatting
            is_positional: Whether to use positional parameters for SQL execution
        """
        self.sql_template = sql_template
        self.description_template = description_template
        self.sql_params = sql_params
        self.desc_params = desc_params
        self.is_positional = is_positional

    def execute(self, executor: Callable[[str, dict[str, Any] | list[Any]], pd.DataFrame]) -> QueryResult:
        """
        Execute the query with the stored parameters.

        Args:
            executor: Function that executes SQL and returns a DataFrame
        """
        # Format the description with the stored parameters
        if callable(self.description_template):
            # Get the signature of the function
            sig = inspect.signature(self.description_template)
            # Filter the context dictionary to only include keys that the function expects
            ctx = {name: self.desc_params[name]
                   for name in sig.parameters if name in self.desc_params}
            description = self.description_template(**ctx)
        else:
            try:
                description = self.description_template.format(
                    **self.desc_params)
            except KeyError:
                # If parameters aren't found in the template, use the template as-is
                description = self.description_template

        # Execute the query using the provided executor function
        if self.is_positional:
            # For positional parameters, convert dict to list in order
            param_values = list(self.sql_params.values())
            data = executor(self.sql_template, param_values)
        else:
            # For named parameters, use the dict
            data = executor(self.sql_template, self.sql_params)

        # Return both description and data together
        return QueryResult(description=description, data=data)


def get_python_type(format_specifier: str) -> Type:
    """
    Map format specifiers to actual Python types.

    Args:
        format_specifier: A single character format specifier like 'd', 's', 'f'

    Returns:
        The corresponding Python type (int, str, float, etc.)
    """
    # Map format specifiers to actual Python types
    type_mapping = {
        # Integers
        'd': int,    # Decimal integer
        'i': int,    # Same as 'd'
        'o': int,    # Octal format
        'u': int,    # Obsolete, same as 'd'
        'x': int,    # Hex format (lowercase)
        'X': int,    # Hex format (uppercase)

        # Floating point
        'e': float,  # Exponent notation (lowercase)
        'E': float,  # Exponent notation (uppercase)
        'f': float,  # Fixed point
        'F': float,  # Same as 'f'
        'g': float,  # General format (lower)
        'G': float,  # General format (upper)

        # String/character
        'c': str,    # Single character
        'r': str,    # String (uses repr())
        's': str,    # String (uses str())

        # Other
        'a': str,    # String (uses ascii())
        'b': bytes,  # Binary format
        '%': str,    # Literal % character
    }

    return type_mapping.get(format_specifier, Any)


def extract_pyformat_variables(sql_string: str, *,
                               __named_pattern=re.compile(
                                   r'%\(([^)]+)\)([diouxXeEfFgGcrsab%])'),
                               __unnamed_pattern=re.compile(
                                   r'%(?!\()(?:[0-9.+-]*)([diouxXeEfFgGcrsab%])')
                               ) -> tuple[dict[str, Type], list[Type]]:
    """
    Extract pyformat variables from a SQL string.
    Handles both named variables like %(name)s and unnamed ones like %.5f
    Ensures each variable name is associated with only one type.

    Args:
        sql_string: A SQL string containing pyformat variables

    Returns:
        A list of dictionaries, each containing 'name' (str) and 'type' (actual Type object)

    Raises:
        InconsistentVariableTypeError: If the same variable name is used with different types
    """
    # Dictionary to track variables and their types
    named_var_types: dict[str, Type] = {}

    # 1. Process named parameters: %(name)s format
    named_matches = __named_pattern.finditer(sql_string)

    for match in named_matches:
        var_name = match.group(1)
        var_type = match.group(2)

        # Get Python type for this format specifier
        python_type = get_python_type(var_type)

        # Check for type consistency
        if var_name in named_var_types and named_var_types[var_name] != python_type:
            raise ValueError(
                f"Variable '{var_name}' is used with multiple types: "
                f"'{named_var_types[var_name].__name__}' and '{python_type.__name__}'"
            )

        named_var_types[var_name] = python_type

    # 2. Process unnamed parameters: %.5f format
    # This pattern matches format specifiers like %.5f, %d, %s, etc.
    # It captures any optional flags, width, precision before the type character
    unnamed_matches = __unnamed_pattern.finditer(sql_string)

    unnamed_var_types: list[Type] = []

    for match in unnamed_matches:
        var_type = match.group(1)

        # Get Python type for this format specifier
        python_type = get_python_type(var_type)
        unnamed_var_types.append(python_type)

    return named_var_types, unnamed_var_types


def extract_format_params(template: str, *, __pat=re.compile(r'\{([^}]+)\}')) -> set[str]:
    """Extract parameter names from a format string using {name} pattern."""
    # Filter out parameters that are part of SQL formatting
    all_matches = __pat.findall(template)
    sql_formatting_params = {'where_clause', 'limit_clause',
                             'having_clause', 'order_by_clause', 'group_by_clause'}
    return {param for param in all_matches if param not in sql_formatting_params}


def get_common_type_annotation(var_name: str, default_type: Type = Any):
    """Return specific type hint annotations for common variable names"""

    annotation_map = {
        "jira_id": str,
        "start_date": date,
        "end_date": date
    }

    return annotation_map.get(var_name, default_type)


def filter_kwargs(kwargs: dict, sig: inspect.Signature) -> dict:
    """Filter keyword arguments to conform with the given signature by excluding any extras"""
    expected_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in expected_params}


def with_description(description_template: str | Callable[..., str]) -> Callable[[SqlFunc], Callable[..., QueryFactory] | Callable[..., DescriptiveQuery]]:
    """
    Decorator that transforms a function to handle both static and dynamic SQL queries.
    If the function has any arguments, it's assumed to be dynamic, so a query factory will be returned.
    Otherwise a DescriptiveQuery instance is returned.

    Args:
        description_template: Template string for describing the query results

    Returns:
        A decorator function that handles the function transformation
    """

    def handle_args_error(e: TypeError, func, sig: inspect.Signature):
        params = list(sig.parameters.values())
        pos_only_count = sum(1 for p in params if p.kind ==
                             inspect.Parameter.POSITIONAL_ONLY)
        pos_or_kw_count = sum(1 for p in params if p.kind ==
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
        kw_only_params = [p.name for p in params if p.kind ==
                          inspect.Parameter.KEYWORD_ONLY]

        # Build a helpful message
        msg = f"Error calling '{func.__name__}': Received too many positional arguments.\n"
        msg += f"This function accepts {pos_only_count + pos_or_kw_count} positional arguments"

        if kw_only_params:
            msg += f" and has the following keyword-only parameters: {', '.join(kw_only_params)}.\n"
            msg += "Please use keyword syntax (param=value) for these parameters."

        raise TypeError(msg) from e

    def decorator(func: SqlFunc) -> Callable[..., QueryFactory] | Callable[..., DescriptiveQuery]:
        # Get the original function signature
        original_sig = inspect.signature(func)

        # Check if the function takes parameters (dynamic SQL)
        is_dynamic_sql = len(original_sig.parameters) > 0

        # For dynamic SQL functions
        if is_dynamic_sql:
            @wraps(func)
            def dynamic_wrapper(*args: Any, **kwargs: Any) -> QueryFactory:
                # Bind the arguments to parameters
                bound_args = original_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Get the SQL-building parameters
                sql_building_params = dict(bound_args.arguments)

                # Generate the SQL template
                sql_template = cleandoc(func(*args, **kwargs))

                # Extract SQL execution parameters
                named_params_type_map, _ = extract_pyformat_variables(
                    sql_template)

                # Create a factory function that returns a DescriptiveQuery when executed
                # with SQL parameters

                # Extract parameters from the description template if it's a string
                if callable(description_template):
                    # For function descriptions, extract parameters from its signature
                    desc_sig = inspect.signature(description_template)
                    desc_params = {name for name, param in desc_sig.parameters.items()
                                   if param.kind != inspect.Parameter.VAR_KEYWORD}
                else:
                    # For string templates, extract from the format string
                    desc_params = extract_format_params(description_template)

                # Create a new signature for the returned factory function
                param_list = []

                # For each param, check if we know of a more specific
                # annotation type, otherwise use the format specifier
                for param, ftype in named_params_type_map.items():
                    param_list.append(inspect.Parameter(
                        name=param,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        annotation=get_common_type_annotation(param, ftype)
                    ))

                # Add description parameters that aren't in SQL params
                for param in desc_params:
                    if param not in named_params_type_map and param not in sql_building_params:
                        param_list.append(inspect.Parameter(
                            name=param,
                            kind=inspect.Parameter.KEYWORD_ONLY,
                            annotation=get_common_type_annotation(param, Any)
                        ))

                factory_sig = inspect.Signature(parameters=param_list)

                # Create the factory function that will return a DescriptiveQuery
                def query_factory(**exec_params: Any) -> DescriptiveQuery:

                    # bind arguments but ignore extras
                    try:
                        bound_args = factory_sig.bind(
                            **filter_kwargs(exec_params, factory_sig))
                    except TypeError as e:
                        handle_args_error(e, query_factory, factory_sig)
                    bound_args.apply_defaults()

                    # Create a complete description params dict including SQL-building params
                    # This ensures all parameters are available for the description template
                    all_desc_params = {**sql_building_params, **exec_params}

                    return DescriptiveQuery(
                        sql_template=sql_template,
                        description_template=description_template,
                        sql_params=exec_params,
                        desc_params=all_desc_params,
                        is_positional=False
                    )

                # Set the signature on the factory
                query_factory.__signature__ = factory_sig
                query_factory.__name__ = f"{func.__name__}_factory"
                query_factory.__qualname__ = f"{func.__name__}_factory"

                return query_factory

            return dynamic_wrapper

        # For static SQL functions
        else:
            # Get the SQL directly
            sql = cleandoc(func())

            # Determine parameter style
            named_params_type_map, unnamed_param_types = extract_pyformat_variables(
                sql)

            positional_count = len(unnamed_param_types)
            is_positional = positional_count > 0 and not named_params_type_map

            # Extract description parameters
            desc_params = extract_format_params(description_template)

            # Create the function with the appropriate signature
            def create_static_function() -> Callable[..., DescriptiveQuery]:
                param_list = []

                if named_params_type_map:
                    # For named parameters
                    all_params = named_params_type_map.keys() | desc_params
                    for param in all_params:
                        param_list.append(inspect.Parameter(
                            name=param,
                            kind=inspect.Parameter.KEYWORD_ONLY,
                            annotation=get_common_type_annotation(
                                param, named_params_type_map.get(param, Any))
                        ))
                elif positional_count > 0:
                    # For positional parameters
                    for i in range(max(positional_count, len(desc_params))):
                        param_name = f"arg{i}" if i >= len(
                            desc_params) else list(desc_params)[i]
                        param_list.append(inspect.Parameter(
                            name=param_name,
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=get_common_type_annotation(
                                param_name, unnamed_param_types.pop(0))))

                # Create the new signature
                sig = inspect.Signature(parameters=param_list)

                # Define the function
                @wraps(func)
                def static_wrapper(*args, **kwargs) -> DescriptiveQuery:

                    # bind arguments but ignore extras
                    try:
                        bound_args = sig.bind(
                            *args, **filter_kwargs(kwargs, sig))
                    except TypeError as e:
                        handle_args_error(e, static_wrapper, sig)
                    bound_args.apply_defaults()
                    params = dict(bound_args.arguments)

                    return DescriptiveQuery(
                        sql_template=sql,
                        description_template=description_template,
                        sql_params=params,
                        desc_params=params,
                        is_positional=is_positional
                    )

                static_wrapper.__signature__ = sig
                return static_wrapper

            return create_static_function()

    return decorator


# --------------------------------------------------------------------------------------
# 1) Basic Dealer Queries (Single-Dealer-Level)
# --------------------------------------------------------------------------------------
@with_description("Basic information about the given dealership.")
def get_basic_dealership_info_query():
    return """
    SELECT
        JIRA_ID,
        DEALERSHIP_NAME,
        DEALERSHIP_GROUP,
        BRAND,
        ACCOUNT_LEAD,
        TEAM_LEAD,
        ACCOUNT_DIRECTOR,
        STREET_ADDRESS,
        CITY,
        STATE,
        ZIP,
        WEBSITE_PROVIDER,
        OEM_PROGRAM AS OEM_Name
    FROM CLIENT_DETAILS
    WHERE JIRA_ID = %s
    """


@with_description("The average monthly total sales (both new and used) for the given dealership between {start_date} and {end_date}.")
def get_aggregated_inventory_sales_query():
    return """
    WITH MonthlySales AS (
        SELECT
            JIRA_ID,
            MONTH,
            YEAR,
            SUM(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS monthly_sales,
            SUM(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS monthly_inventory
        FROM INVENTORY_AND_SALES
        WHERE
            JIRA_ID = %(jira_id)s
            AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY JIRA_ID, MONTH, YEAR
    )
    SELECT
        JIRA_ID,
        AVG(monthly_sales) AS AVG_SALES,
        AVG(monthly_inventory) AS AVG_INVENTORY
    FROM MonthlySales
    GROUP BY JIRA_ID;
    """


@with_description("Monthly inventory and sales data for the given dealership between {start_date} and {end_date}.")
def get_monthly_inventory_sales_query():
    return """
    SELECT
        JIRA_ID,
        MONTH,
        YEAR,
        NEW_INVENTORY_AVERAGE,
        USED_INVENTORY_AVERAGE,
        NEW_SALES_TOTAL,
        USED_SALES_TOTAL
    FROM INVENTORY_AND_SALES
    WHERE JIRA_ID = %(jira_id)s
    AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY YEAR ASC, MONTH ASC
    """


@with_description("Aggregated Google Ads data for the given dealership between {start_date} and {end_date}.")
def get_aggregated_google_ads_query():
    return """
    WITH MonthlyAds AS (
        SELECT
            JIRA_ID,
            MONTH,
            YEAR,
            SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS monthly_spend,
            SUM(clicks) AS monthly_clicks,
            SUM(impressions) AS monthly_impressions,
            SUM(conversions) AS monthly_conversions
        FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
        WHERE JIRA_ID = %(jira_id)s
        AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY JIRA_ID, MONTH, YEAR
    )
    SELECT
        JIRA_ID,
        AVG(monthly_spend) AS AVG_AD_SPEND,
        SUM(monthly_clicks) AS total_clicks,
        SUM(monthly_impressions) AS total_impressions,
        CASE
            WHEN SUM(monthly_impressions) = 0 THEN 0
            ELSE ROUND((SUM(monthly_clicks) * 100.0) / SUM(monthly_impressions), 2)
        END AS ctr_percent,
        SUM(monthly_conversions) AS total_conversions
    FROM MonthlyAds
    GROUP BY JIRA_ID;
    """


@with_description("GA4 website traffic data for the given dealership between {start_date} and {end_date}.")
def get_ga4_website_traffic_query():
    return """
    WITH MonthlySessions AS (
        SELECT
            JIRA_ID,
            MONTH,
            YEAR,
            SUM(SESSIONS) AS monthly_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
        WHERE JIRA_ID = %(jira_id)s
        AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY JIRA_ID, MONTH, YEAR
    )
    SELECT AVG(monthly_sessions) AS AVG_SESSIONS
    FROM MonthlySessions
    """


@with_description("Statistics about mystery shops response times for the given dealership between {start_date} and {end_date}.")
def get_mystery_shop_stats_query():
    return """
    WITH ValidMystery AS (
        SELECT
            JIRA_ID,
            CASE WHEN CAST(auto_email_response_time AS DECIMAL) > 0
                AND CAST(auto_email_response_time AS DECIMAL) < 60
                THEN CAST(auto_email_response_time AS DECIMAL) END AS auto_email_valid,
            CASE WHEN CAST(personal_email_response_time AS DECIMAL) > 0
                AND CAST(personal_email_response_time AS DECIMAL) < 60
                THEN CAST(personal_email_response_time AS DECIMAL) END AS personal_email_valid,
            CASE WHEN CAST(call_response_time AS DECIMAL) > 0
                AND CAST(call_response_time AS DECIMAL) < 60
                THEN CAST(call_response_time AS DECIMAL) END AS call_valid,
            CASE WHEN CAST(text_response_time AS DECIMAL) > 0
                AND CAST(text_response_time AS DECIMAL) < 60
                THEN CAST(text_response_time AS DECIMAL) END AS text_valid
        FROM MYSTERY_SHOPS_CLIENT_RESPONSE_TIMES
        WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    ),
    MysteryAgg AS (
        SELECT
            JIRA_ID,
            AVG(auto_email_valid) AS avg_auto_email,
            AVG(personal_email_valid) AS avg_personal_email,
            AVG(call_valid) AS avg_call_time,
            AVG(text_valid) AS avg_text_time
        FROM ValidMystery
        GROUP BY JIRA_ID
    )
    SELECT *
    FROM MysteryAgg
    WHERE JIRA_ID = %(jira_id)s
    """


@with_description("Total budget allocation data for different channels for the given dealership between {start_date} and {end_date}.")
def get_budget_allocation_query():
    return """
    WITH BudgetAgg AS (
        SELECT
            JIRA_ID,
            SUM(Total_Client_Budget) AS sum_budget,
            SUM(Total_Remarketing) AS sum_remarketing,
            SUM(Total_Display) AS sum_display,
            SUM(Total_Search) AS sum_search,
            SUM(Total_Discovery_Performance_Max) AS sum_discovery,
            SUM(Total_Shopping) AS sum_shopping,
            SUM(Total_Video) AS sum_video,
            SUM(Other_Total) AS sum_other,
            SUM(Total_Social) AS sum_social
        FROM CLIENT_BUDGETS
        WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY JIRA_ID
    )
    SELECT *
    FROM BudgetAgg
    WHERE JIRA_ID = %(jira_id)s
    """


@with_description("Overall sales rank for the given dealership between {start_date} and {end_date}.")
def get_overall_sales_rank_query():
    return """
    WITH SalesAgg AS (
      SELECT
          cd.JIRA_ID,
          COALESCE(SUM(inv.NEW_SALES_TOTAL + inv.USED_SALES_TOTAL), 0) AS total_sales
      FROM CLIENT_DETAILS cd
      LEFT JOIN INVENTORY_AND_SALES inv
         ON cd.JIRA_ID = inv.JIRA_ID
         AND DATE_FROM_PARTS(inv.YEAR, inv.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY cd.JIRA_ID
    ),
    RankedSales AS (
      SELECT
          JIRA_ID,
          total_sales,
          RANK() OVER (ORDER BY total_sales DESC) AS sales_rank,
          COUNT(*) OVER () AS total_dealers
      FROM SalesAgg
    )
    SELECT sales_rank, total_dealers
    FROM RankedSales
    WHERE JIRA_ID = %(jira_id)s
    LIMIT 1
    """


@with_description("OEM Program sales rank for the given dealership between {start_date} and {end_date}.")
def get_oem_sales_rank_query():
    return """
    WITH DealerOEM AS (
      SELECT
          JIRA_ID,
          OEM_PROGRAM AS OEM_Name
      FROM CLIENT_DETAILS
      WHERE JIRA_ID = %(jira_id)s
      LIMIT 1
    ),
    OEMSalesAgg AS (
      SELECT
          c.JIRA_ID,
          c.OEM_PROGRAM AS OEM_Name,
          COALESCE(SUM(i.NEW_SALES_TOTAL + i.USED_SALES_TOTAL), 0) AS total_sales
      FROM CLIENT_DETAILS c
      LEFT JOIN INVENTORY_AND_SALES i
         ON c.JIRA_ID = i.JIRA_ID
         AND DATE_FROM_PARTS(i.YEAR, i.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY c.JIRA_ID, c.OEM_PROGRAM
    ),
    OEMFiltered AS (
      SELECT *
      FROM OEMSalesAgg
      WHERE OEM_Name = (SELECT OEM_Name FROM DealerOEM)
    ),
    RankedOEM AS (
      SELECT
          JIRA_ID,
          total_sales,
          RANK() OVER (ORDER BY total_sales DESC) AS oem_sales_rank,
          COUNT(*) OVER () AS total_dealers_in_oem
      FROM OEMFiltered
    )
    SELECT oem_sales_rank, total_dealers_in_oem
    FROM RankedOEM
    WHERE JIRA_ID = %(jira_id)s
    LIMIT 1
    """


# --------------------------------------------------------------------------------------
# 2) Additional Queries for State/Brand Group Comparisons or Aggregates
# --------------------------------------------------------------------------------------
@with_description("State-level aggregated metrics for {state} between {start_date} and {end_date}.")
def get_state_aggregated_metrics_query():
    return """
    WITH InvAgg AS (
      SELECT
          JIRA_ID,
          SUM(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS total_sales,
          AVG(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS avg_inventory
      FROM INVENTORY_AND_SALES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    AdsAgg AS (
      SELECT
          JIRA_ID,
          SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend
      FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    GA_Agg AS (
      SELECT
          JIRA_ID,
          SUM(SESSIONS) AS total_sessions
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT
      cd.STATE,
      SUM(COALESCE(ia.total_sales,0)) AS total_sales,
      AVG(COALESCE(ia.avg_inventory,0)) AS avg_inventory,
      SUM(COALESCE(aa.total_ad_spend,0)) AS total_ad_spend,
      SUM(COALESCE(ga.total_sessions,0)) AS total_sessions
    FROM CLIENT_DETAILS cd
    LEFT JOIN InvAgg ia ON cd.JIRA_ID = ia.JIRA_ID
    LEFT JOIN AdsAgg aa ON cd.JIRA_ID = aa.JIRA_ID
    LEFT JOIN GA_Agg ga ON cd.JIRA_ID = ga.JIRA_ID
    WHERE cd.STATE = %(state)s
    GROUP BY cd.STATE
    """


@with_description("Brand-level aggregated metrics for {brand} between {start_date} and {end_date}.")
def get_brand_aggregated_metrics_query():
    return """
    WITH InvAgg AS (
      SELECT
          JIRA_ID,
          SUM(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS total_sales,
          AVG(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS avg_inventory
      FROM INVENTORY_AND_SALES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    AdsAgg AS (
      SELECT
          JIRA_ID,
          SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend
      FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    GA_Agg AS (
      SELECT
          JIRA_ID,
          SUM(SESSIONS) AS total_sessions
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT
      cd.BRAND,
      SUM(COALESCE(ia.total_sales,0)) AS total_sales,
      AVG(COALESCE(ia.avg_inventory,0)) AS avg_inventory,
      SUM(COALESCE(aa.total_ad_spend,0)) AS total_ad_spend,
      SUM(COALESCE(ga.total_sessions,0)) AS total_sessions
    FROM CLIENT_DETAILS cd
    LEFT JOIN InvAgg ia ON cd.JIRA_ID = ia.JIRA_ID
    LEFT JOIN AdsAgg aa ON cd.JIRA_ID = aa.JIRA_ID
    LEFT JOIN GA_Agg ga ON cd.JIRA_ID = ga.JIRA_ID
    WHERE cd.BRAND = %(brand)s
    GROUP BY cd.BRAND
    """


@with_description("Aggregated metrics for OEM Program: {oem} between {start_date} and {end_date}.")
def get_oem_group_aggregated_metrics_query():
    return """
    WITH InvAgg AS (
      SELECT
          JIRA_ID,
          SUM(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS total_sales,
          AVG(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS avg_inventory
      FROM INVENTORY_AND_SALES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    AdsAgg AS (
      SELECT
          JIRA_ID,
          SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend
      FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    GA_Agg AS (
      SELECT
          JIRA_ID,
          SUM(SESSIONS) AS total_sessions
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT
      cd.OEM_PROGRAM AS OEM_Name,
      SUM(COALESCE(ia.total_sales,0)) AS total_sales,
      AVG(COALESCE(ia.avg_inventory,0)) AS avg_inventory,
      SUM(COALESCE(aa.total_ad_spend,0)) AS total_ad_spend,
      SUM(COALESCE(ga.total_sessions,0)) AS total_sessions
    FROM CLIENT_DETAILS cd
    LEFT JOIN InvAgg ia ON cd.JIRA_ID = ia.JIRA_ID
    LEFT JOIN AdsAgg aa ON cd.JIRA_ID = aa.JIRA_ID
    LEFT JOIN GA_Agg ga ON cd.JIRA_ID = ga.JIRA_ID
    WHERE cd.OEM_PROGRAM = %(oem)s
    GROUP BY cd.OEM_PROGRAM
    """


# --------------------------------------------------------------------------------------
# 3) Queries Returning Data Across ALL Dealers (Needed for Multi-Dealer Stats)
# --------------------------------------------------------------------------------------
@with_description("Aggregated total metrics across all dealerships between {start_date} and {end_date}.")
def get_all_dealers_aggregated_metrics_query():
    return """
    WITH Inv_PreAgg AS (
      SELECT
          JIRA_ID,
          AVG(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS avg_sales_per_month,
          AVG(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS avg_inventory_per_month
      FROM INVENTORY_AND_SALES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    GA_PreAgg AS (
      SELECT
          JIRA_ID,
          AVG(SESSIONS) AS avg_sessions_per_month
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    Ads_PreAgg AS (
      SELECT
          JIRA_ID,
          SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend
      FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    Budget_PreAgg AS (
      SELECT
          JIRA_ID,
          SUM(Total_Client_Budget) AS total_budget
      FROM CLIENT_BUDGETS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT
        cd.JIRA_ID,
        cd.DEALERSHIP_NAME,
        cd.STATE,
        cd.BRAND,
        cd.OEM_PROGRAM AS OEM_Name,
        COALESCE(ip.avg_sales_per_month, 0)     AS avg_sales_per_month,
        COALESCE(ip.avg_inventory_per_month, 0) AS avg_inventory_per_month,
        COALESCE(ga.avg_sessions_per_month, 0)  AS avg_sessions_per_month,
        COALESCE(ap.total_ad_spend, 0)          AS total_ad_spend,
        COALESCE(bp.total_budget, 0)            AS total_budget
    FROM CLIENT_DETAILS cd
    LEFT JOIN Inv_PreAgg ip ON cd.JIRA_ID = ip.JIRA_ID
    LEFT JOIN GA_PreAgg ga ON cd.JIRA_ID = ga.JIRA_ID
    LEFT JOIN Ads_PreAgg ap ON cd.JIRA_ID = ap.JIRA_ID
    LEFT JOIN Budget_PreAgg bp ON cd.JIRA_ID = bp.JIRA_ID
    """


@with_description("Aggregated monthly metrics across all dealerships between {start_date} and {end_date}.")
def get_all_dealers_monthly_data_query():
    return """
    WITH Inv AS (
        SELECT
            i.JIRA_ID,
            i.MONTH,
            i.YEAR,
            (i.NEW_SALES_TOTAL + i.USED_SALES_TOTAL) AS total_sales,
            (i.NEW_INVENTORY_AVERAGE + i.USED_INVENTORY_AVERAGE) AS monthly_inventory
        FROM INVENTORY_AND_SALES i
        WHERE DATE_FROM_PARTS(i.YEAR, i.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    ),
    GA AS (
        SELECT
            g.JIRA_ID,
            g.MONTH,
            g.YEAR,
            SUM(g.SESSIONS) AS total_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024 g
        WHERE DATE_FROM_PARTS(g.YEAR, g.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY g.JIRA_ID, g.MONTH, g.YEAR
    ),
    Ads AS (
        SELECT
            a.JIRA_ID,
            a.MONTH,
            a.YEAR,
            SUM(CAST(REPLACE(REPLACE(a.cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend
        FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS a
        WHERE DATE_FROM_PARTS(a.YEAR, a.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY a.JIRA_ID, a.MONTH, a.YEAR
    ),
    Budget AS (
        SELECT
            b.JIRA_ID,
            b.MONTH,
            b.YEAR,
            SUM(b.Total_Client_Budget) AS monthly_budget
        FROM CLIENT_BUDGETS b
        WHERE DATE_FROM_PARTS(b.YEAR, b.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY b.JIRA_ID, b.MONTH, b.YEAR
    )
    SELECT
        cd.JIRA_ID,
        cd.DEALERSHIP_NAME,
        cd.STATE,
        cd.BRAND,
        cd.OEM_PROGRAM AS OEM_Name,
        i.MONTH,
        i.YEAR,
        COALESCE(i.total_sales, 0)       AS total_sales,
        COALESCE(i.monthly_inventory, 0) AS monthly_inventory,
        COALESCE(g.total_sessions, 0)    AS total_sessions,
        COALESCE(ad.total_ad_spend, 0)   AS total_ad_spend,
        COALESCE(bd.monthly_budget, 0)   AS monthly_budget
    FROM CLIENT_DETAILS cd
    LEFT JOIN Inv i
        ON cd.JIRA_ID = i.JIRA_ID
    LEFT JOIN GA g
        ON cd.JIRA_ID = g.JIRA_ID
        AND i.MONTH = g.MONTH
        AND i.YEAR = g.YEAR
    LEFT JOIN Ads ad
        ON cd.JIRA_ID = ad.JIRA_ID
        AND i.MONTH = ad.MONTH
        AND i.YEAR = ad.YEAR
    LEFT JOIN Budget bd
        ON cd.JIRA_ID = bd.JIRA_ID
        AND i.MONTH = bd.MONTH
        AND i.YEAR = bd.YEAR
    ORDER BY cd.JIRA_ID, i.YEAR, i.MONTH
    """


# Original query prept for dynamic start and end dates
@with_description("Mystery shop response time metrics across all dealerships between {start_date} and {end_date}.")
def get_all_dealers_monthly_mystery_query():
    return """
    WITH MonthlyMystery AS (
        SELECT
            ms.JIRA_ID,
            ms.YEAR,
            ms.MONTH,
            AVG(
              CASE WHEN CAST(ms.auto_email_response_time AS DECIMAL) > 0
                   AND CAST(ms.auto_email_response_time AS DECIMAL) < 60
                   THEN CAST(ms.auto_email_response_time AS DECIMAL)
              END
            ) AS avg_auto_email,
            AVG(
              CASE WHEN CAST(ms.personal_email_response_time AS DECIMAL) > 0
                   AND CAST(ms.personal_email_response_time AS DECIMAL) < 60
                   THEN CAST(ms.personal_email_response_time AS DECIMAL)
              END
            ) AS avg_personal_email,
            AVG(
              CASE WHEN CAST(ms.call_response_time AS DECIMAL) > 0
                   AND CAST(ms.call_response_time AS DECIMAL) < 60
                   THEN CAST(ms.call_response_time AS DECIMAL)
              END
            ) AS avg_call_time,
            AVG(
              CASE WHEN CAST(ms.text_response_time AS DECIMAL) > 0
                   AND CAST(ms.text_response_time AS DECIMAL) < 60
                   THEN CAST(ms.text_response_time AS DECIMAL)
              END
            ) AS avg_text_time
        FROM MYSTERY_SHOPS_CLIENT_RESPONSE_TIMES ms
        WHERE DATE_FROM_PARTS(ms.YEAR, ms.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY ms.JIRA_ID, ms.YEAR, ms.MONTH
    )
    SELECT
        cd.JIRA_ID,
        cd.DEALERSHIP_NAME,
        cd.STATE,
        cd.BRAND,
        cd.OEM_PROGRAM AS OEM_Name,
        mm.MONTH,
        mm.YEAR,
        mm.avg_auto_email,
        mm.avg_personal_email,
        mm.avg_call_time,
        mm.avg_text_time
    FROM CLIENT_DETAILS cd
    LEFT JOIN MonthlyMystery mm ON cd.JIRA_ID = mm.JIRA_ID
    ORDER BY cd.JIRA_ID, mm.YEAR, mm.MONTH
    """


# --------------------------------------------------------------------------------------
# 4) Other Queries
# --------------------------------------------------------------------------------------
@with_description("Monthly budget allocation for the given dealership between {start_date} and {end_date}.")
def get_monthly_budget_allocation_query():
    return """
    SELECT
        JIRA_ID,
        MONTH,
        YEAR,
        Total_Client_Budget,
        Total_Remarketing,
        Total_Display,
        Total_Search,
        Total_Discovery_Performance_Max,
        Total_Shopping,
        Total_Video,
        Other_Total,
        Total_Social
    FROM CLIENT_BUDGETS
    WHERE JIRA_ID = %(jira_id)s
    AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    ORDER BY YEAR ASC, MONTH ASC
    """


@with_description("GA4 Organic search metrics for the given dealership between {start_date} and {end_date}.")
def get_ga4_organic_search_query():
    return """
    WITH OrganicAgg AS (
      SELECT
          JIRA_ID,
          SUM(SESSIONS) AS organic_sessions
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE SESSION_DEFAULT_CHANNEL_GROUP = 'Organic Search'
      AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT organic_sessions
    FROM OrganicAgg
    WHERE JIRA_ID = %(jira_id)s
    """


@with_description("Cohort analysis data for all dealerships")
def get_cohort_analysis_query():
    return """
    WITH CohortData AS (
      SELECT
          JIRA_ID,
          DATE_FORMAT(JOIN_DATE, '%Y-%m') AS join_month
      FROM CLIENT_DETAILS
      WHERE JOIN_DATE IS NOT NULL
    )
    SELECT *
    FROM CohortData
    """


@with_description("Mystery shop stats for all dealerships between {start_date} and {end_date}.")
def get_all_mystery_shop_stats_query():
    return """
    WITH ValidMystery AS (
      SELECT
          JIRA_ID,
          CASE WHEN CAST(auto_email_response_time AS DECIMAL) > 0
                AND CAST(auto_email_response_time AS DECIMAL) < 60
                THEN CAST(auto_email_response_time AS DECIMAL) END AS auto_email_valid,
          CASE WHEN CAST(personal_email_response_time AS DECIMAL) > 0
                AND CAST(personal_email_response_time AS DECIMAL) < 60
                THEN CAST(personal_email_response_time AS DECIMAL) END AS personal_email_valid,
          CASE WHEN CAST(call_response_time AS DECIMAL) > 0
                AND CAST(call_response_time AS DECIMAL) < 60
                THEN CAST(call_response_time AS DECIMAL) END AS call_valid,
          CASE WHEN CAST(text_response_time AS DECIMAL) > 0
                AND CAST(text_response_time AS DECIMAL) < 60
                THEN CAST(text_response_time AS DECIMAL) END AS text_valid
      FROM MYSTERY_SHOPS_CLIENT_RESPONSE_TIMES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    ),
    MysteryAgg AS (
      SELECT
          JIRA_ID,
          AVG(auto_email_valid)     AS avg_auto_email,
          AVG(personal_email_valid) AS avg_personal_email,
          AVG(call_valid)           AS avg_call_time,
          AVG(text_valid)           AS avg_text_time
      FROM ValidMystery
      GROUP BY JIRA_ID
    )
    SELECT *
    FROM MysteryAgg
    """


@with_description("Aggregated metrics for all dealerships grouped by brand between {start_date} and {end_date}.")
def get_all_brands_aggregated_metrics_query():
    return """
    WITH InvAgg AS (
      SELECT
          JIRA_ID,
          SUM(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS total_sales,
          AVG(NEW_INVENTORY_AVERAGE + USED_INVENTORY_AVERAGE) AS avg_inventory
      FROM INVENTORY_AND_SALES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    AdsAgg AS (
      SELECT
          JIRA_ID,
          SUM(CAST(REPLACE(REPLACE(cost, '$',''), ',', '') AS DECIMAL(10,2))) AS total_ad_spend
      FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    ),
    GA_Agg AS (
      SELECT
          JIRA_ID,
          SUM(SESSIONS) AS total_sessions
      FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
      GROUP BY JIRA_ID
    )
    SELECT
      cd.BRAND,
      SUM(COALESCE(ia.total_sales,0)) AS total_sales,
      AVG(COALESCE(ia.avg_inventory,0)) AS avg_inventory,
      SUM(COALESCE(aa.total_ad_spend,0)) AS total_ad_spend,
      SUM(COALESCE(ga.total_sessions,0)) AS total_sessions
    FROM CLIENT_DETAILS cd
    LEFT JOIN InvAgg ia ON cd.JIRA_ID = ia.JIRA_ID
    LEFT JOIN AdsAgg aa ON cd.JIRA_ID = aa.JIRA_ID
    LEFT JOIN GA_Agg ga ON cd.JIRA_ID = ga.JIRA_ID
    GROUP BY cd.BRAND
    """


@with_description("Monthly google ads data for the given dealership between {start_date} and {end_date}.")
def get_monthly_google_ads_query():
    return """
    SELECT
       JIRA_ID,
       MONTH,
       YEAR,
       SUM(CAST(REPLACE(REPLACE(cost, '$',''),',','') AS DECIMAL(10,2))) AS total_ad_spend,
       SUM(clicks) AS total_clicks,
       SUM(impressions) AS total_impressions,
       CASE
            WHEN SUM(impressions) = 0 THEN 0
            ELSE ROUND((SUM(clicks) * 100.0) / SUM(impressions), 2)
       END AS ctr_percent,
       SUM(conversions) AS total_conversions
    FROM GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS
    WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    AND JIRA_ID = %(jira_id)s
    GROUP BY JIRA_ID, MONTH, YEAR
    ORDER BY YEAR ASC, MONTH ASC
    """


@with_description("Monthly GA4 website traffic data for the given dealership between {start_date} and {end_date}.")
def get_monthly_ga4_website_traffic_query():
    return """
    SELECT
      JIRA_ID,
      MONTH,
      YEAR,
      SUM(SESSIONS) AS total_sessions,
      SUM(VIEWS) AS total_views,
      SUM(NEW_USERS) AS total_new_users,
      SUM(KEY_EVENTS) AS total_key_events,
      SUM(ENGAGED_SESSIONS) AS total_engaged_sessions
    FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
    WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    AND JIRA_ID = %(jira_id)s
    GROUP BY JIRA_ID, MONTH, YEAR
    ORDER BY YEAR ASC, MONTH ASC
    """


@with_description("Detailed monthly budget breakdown for the given dealership between {start_date} and {end_date}.")
def get_detailed_budget_breakdown_query():
    return """
    WITH MonthlyBudget AS (
        SELECT
            JIRA_ID,
            MONTH,
            YEAR,
            Total_Client_Budget,
            Total_Remarketing,
            Remarketing_GDN,
            Remarketing_AIM,
            Total_Display,
            Display_GDN,
            Display_AIM,
            Display_Bing_Audience_Ads,
            Total_Search,
            Search_Google,
            Search_Bing,
            Total_Discovery_Performance_Max,
            DISCOVERY_PERFORMANCE_MAX_LOCAL,
            DISCOVERY_PERFORMANCE_MAX_DISCOVERY,
            DISCOVERY_PERFORMANCE_MAX_PERFORMANCE_MAX,
            Total_Shopping,
            Shopping_Google,
            Shopping_Bing,
            Total_Video,
            Video_Youtube,
            Video_CTV,
            Other_Total,
            Other_Criteo,
            Other_Waze,
            Other_Amazon,
            Total_Social,
            Social_Facebook,
            Social_Snapchat
        FROM CLIENT_BUDGETS
        WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        AND JIRA_ID = %(jira_id)s
    )
    SELECT *
    FROM MonthlyBudget
    ORDER BY YEAR, MONTH;
    """


@with_description("Website traffic data breakdown for the given dealership between {start_date} and {end_date}.")
def get_website_traffic_breakdown_query():
    return """
    WITH MonthlyTraffic AS (
        SELECT
            JIRA_ID,
            MONTH,
            YEAR,
            SESSION_DEFAULT_CHANNEL_GROUP,
            SUM(SESSIONS) AS total_sessions_month
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
        WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        AND JIRA_ID = %(jira_id)s
        GROUP BY JIRA_ID, MONTH, YEAR, SESSION_DEFAULT_CHANNEL_GROUP
    ),
    ChannelAggregated AS (
        SELECT
            SESSION_DEFAULT_CHANNEL_GROUP,
            SUM(total_sessions_month) AS total_sessions,
            AVG(total_sessions_month) AS avg_monthly_sessions
        FROM MonthlyTraffic
        GROUP BY SESSION_DEFAULT_CHANNEL_GROUP
    ),
    StateTraffic AS (
        SELECT
            g.SESSION_DEFAULT_CHANNEL_GROUP,
            AVG(g.SESSIONS) AS state_avg_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024 g
        JOIN client_details c ON g.JIRA_ID = c.JIRA_ID
        WHERE c.STATE = (SELECT STATE FROM client_details WHERE JIRA_ID = %(jira_id)s)
        AND DATE_FROM_PARTS(g.YEAR, g.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY g.SESSION_DEFAULT_CHANNEL_GROUP
    ),
    BrandTraffic AS (
        SELECT
            g.SESSION_DEFAULT_CHANNEL_GROUP,
            AVG(g.SESSIONS) AS brand_avg_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024 g
        JOIN CLIENT_DETAILS c ON g.JIRA_ID = c.JIRA_ID
        WHERE c.BRAND = (SELECT BRAND FROM CLIENT_DETAILS WHERE JIRA_ID = %(jira_id)s)
        AND DATE_FROM_PARTS(g.YEAR, g.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY g.SESSION_DEFAULT_CHANNEL_GROUP
    )
    SELECT
        ca.SESSION_DEFAULT_CHANNEL_GROUP,
        ca.total_sessions,
        ca.avg_monthly_sessions,
        st.state_avg_sessions,
        bt.brand_avg_sessions
    FROM ChannelAggregated ca
    LEFT JOIN StateTraffic st
           ON ca.SESSION_DEFAULT_CHANNEL_GROUP = st.SESSION_DEFAULT_CHANNEL_GROUP
    LEFT JOIN BrandTraffic bt
           ON ca.SESSION_DEFAULT_CHANNEL_GROUP = bt.SESSION_DEFAULT_CHANNEL_GROUP
    ORDER BY ca.SESSION_DEFAULT_CHANNEL_GROUP;
    """


@with_description("Data on missing traffic channels compared to matching state and brand groups for the given dealership between {start_date} and {end_date}.")
def get_missing_channels_query():
    return """
    WITH StateTraffic AS (
        SELECT
            g.SESSION_DEFAULT_CHANNEL_GROUP,
            AVG(g.SESSIONS) AS state_avg_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024 g
        JOIN CLIENT_DETAILS c ON g.JIRA_ID = c.JIRA_ID
        WHERE c.STATE = (SELECT STATE FROM CLIENT_DETAILS WHERE JIRA_ID = %(jira_id)s)
        AND DATE_FROM_PARTS(g.YEAR, g.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY g.SESSION_DEFAULT_CHANNEL_GROUP
    ),
    BrandTraffic AS (
        SELECT
            g.SESSION_DEFAULT_CHANNEL_GROUP,
            AVG(g.SESSIONS) AS brand_avg_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024 g
        JOIN CLIENT_DETAILS c ON g.JIRA_ID = c.JIRA_ID
        WHERE c.BRAND = (SELECT BRAND FROM CLIENT_DETAILS WHERE JIRA_ID = %(jira_id)s)
        AND DATE_FROM_PARTS(g.YEAR, g.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY g.SESSION_DEFAULT_CHANNEL_GROUP
    ),
    DealerChannels AS (
        SELECT
            SESSION_DEFAULT_CHANNEL_GROUP
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
        WHERE JIRA_ID = %(jira_id)s
        AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY SESSION_DEFAULT_CHANNEL_GROUP
    ),
    AllChannels AS (
        SELECT SESSION_DEFAULT_CHANNEL_GROUP FROM StateTraffic
        UNION
        SELECT SESSION_DEFAULT_CHANNEL_GROUP FROM BrandTraffic
    )
    SELECT
        ac.SESSION_DEFAULT_CHANNEL_GROUP,
        st.state_avg_sessions,
        bt.brand_avg_sessions
    FROM AllChannels ac
    LEFT JOIN DealerChannels dc
        ON ac.SESSION_DEFAULT_CHANNEL_GROUP = dc.SESSION_DEFAULT_CHANNEL_GROUP
    LEFT JOIN StateTraffic st
        ON ac.SESSION_DEFAULT_CHANNEL_GROUP = st.SESSION_DEFAULT_CHANNEL_GROUP
    LEFT JOIN BrandTraffic bt
        ON ac.SESSION_DEFAULT_CHANNEL_GROUP = bt.SESSION_DEFAULT_CHANNEL_GROUP
    WHERE dc.SESSION_DEFAULT_CHANNEL_GROUP IS NULL
    ORDER BY ac.SESSION_DEFAULT_CHANNEL_GROUP;
    """


@with_description("Monthly mystery shops data breakdown for the given dealership between {start_date} and {end_date}.")
def get_monthly_mystery_shop_breakdown_query():
    return """
    WITH ValidMystery AS (
      SELECT
          JIRA_ID,
          YEAR,
          MONTH,
          CASE
            WHEN CAST(auto_email_response_time AS DECIMAL) > 0
                 AND CAST(auto_email_response_time AS DECIMAL) < 60
            THEN CAST(auto_email_response_time AS DECIMAL)
          END AS auto_email_valid,
          CASE
            WHEN CAST(personal_email_response_time AS DECIMAL) > 0
                 AND CAST(personal_email_response_time AS DECIMAL) < 60
            THEN CAST(personal_email_response_time AS DECIMAL)
          END AS personal_email_valid,
          CASE
            WHEN CAST(call_response_time AS DECIMAL) > 0
                 AND CAST(call_response_time AS DECIMAL) < 60
            THEN CAST(call_response_time AS DECIMAL)
          END AS call_valid,
          CASE
            WHEN CAST(text_response_time AS DECIMAL) > 0
                 AND CAST(text_response_time AS DECIMAL) < 60
            THEN CAST(text_response_time AS DECIMAL)
          END AS text_valid
      FROM MYSTERY_SHOPS_CLIENT_RESPONSE_TIMES
      WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        AND JIRA_ID = %(jira_id)s
    )
    SELECT
      YEAR,
      MONTH,
      AVG(auto_email_valid)     AS avg_auto_email,
      AVG(personal_email_valid) AS avg_personal_email,
      AVG(call_valid)           AS avg_call_time,
      AVG(text_valid)           AS avg_text_time,
      COUNT(auto_email_valid)   AS count_auto_email,
      COUNT(personal_email_valid) AS count_personal_email,
      COUNT(call_valid)           AS count_call,
      COUNT(text_valid)           AS count_text
    FROM ValidMystery
    GROUP BY YEAR, MONTH
    ORDER BY YEAR, MONTH;
    """


@with_description("All states where we have clients")
def get_all_states_query():
    return """
    SELECT DISTINCT STATE
    FROM CLIENT_DETAILS
    WHERE STATE IS NOT NULL AND STATE != ''
    ORDER BY STATE
    """


@with_description("All distinct brands across all clients")
def get_all_brands_query():
    return """
    SELECT DISTINCT BRAND
    FROM CLIENT_DETAILS
    WHERE BRAND IS NOT NULL AND BRAND != ''
    ORDER BY BRAND
    """


def leaderboard_description_generator(leaderboard_type: str):
    def generate_leaderboard_description(
            *, limit: int | None,
            start_date: date,
            end_date: date,
            state_filter: str | None = None,
            brand_filter: str | None = None):
        """Generate a dynamic description for the leaderboard query"""
        parts = [f"from {start_date} to {end_date}"]
        if state_filter:
            parts.append(f"in {state_filter}")
        if brand_filter:
            parts.append(f"for {brand_filter} brand")
        filters = " ".join(parts)
        if limit is not None:
            return f"Top {limit} {leaderboard_type} leaderboard data {filters}"
        return f"{leaderboard_type.capitalize()} leaderboard data {filters}"
    return generate_leaderboard_description


@with_description(leaderboard_description_generator("inventory"))
def get_inventory_leaderboard_query(*, state_filter: str = None, brand_filter: str = None, limit: int = 10):
    base_query = """
    WITH SalesMetrics AS (
        SELECT
            cd.JIRA_ID,
            cd.DEALERSHIP_NAME,
            cd.STATE,
            cd.BRAND,
            SUM(inv.NEW_SALES_TOTAL + inv.USED_SALES_TOTAL) AS total_sales,
            -- average monthly sales = total sales / count of distinct months
            SUM(inv.NEW_SALES_TOTAL + inv.USED_SALES_TOTAL) / COUNT(DISTINCT inv.MONTH) AS avg_monthly_sales,
            AVG(inv.NEW_INVENTORY_AVERAGE + inv.USED_INVENTORY_AVERAGE) AS avg_inventory
        FROM CLIENT_DETAILS cd
        LEFT JOIN INVENTORY_AND_SALES inv
            ON cd.JIRA_ID = inv.JIRA_ID
            AND DATE_FROM_PARTS(inv.YEAR, inv.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        {where_clause}
        GROUP BY cd.JIRA_ID, cd.DEALERSHIP_NAME, cd.STATE, cd.BRAND
    ),
    RankedDealers AS (
        SELECT
            sm.*,
            ROW_NUMBER() OVER (ORDER BY sm.total_sales DESC) AS dealer_rank,
            CASE
                WHEN sm.avg_inventory = 0 THEN 0
                ELSE sm.total_sales / sm.avg_inventory
            END AS inventory_turnover_ratio
        FROM SalesMetrics sm
        WHERE sm.total_sales > 0
    )
    SELECT
        JIRA_ID,
        dealer_rank AS "Rank",
        DEALERSHIP_NAME AS "Dealership Name",
        STATE,
        BRAND,
        total_sales AS "Total Sales",
        ROUND(avg_monthly_sales, 2) AS "Avg Monthly Sales",
        ROUND(avg_inventory, 2) AS "Avg Inventory",
        ROUND(inventory_turnover_ratio, 2) AS "Inventory Turnover Ratio"
    FROM RankedDealers
    {limit_clause}
    ORDER BY dealer_rank;
    """

    filters = []
    if state_filter:
        filters.append(f"cd.STATE = '{state_filter}'")
    if brand_filter:
        filters.append(f"cd.BRAND = '{brand_filter}'")

    where_clause = " WHERE " + " AND ".join(filters) if filters else ""
    limit_clause = f"WHERE dealer_rank <= {limit}" if limit else ""

    return base_query.format(where_clause=where_clause, limit_clause=limit_clause)


@with_description(leaderboard_description_generator("budget"))
def get_budget_leaderboard_query(*, state_filter: str = None, brand_filter: str = None, limit: int = 10):
    base_query = """
    WITH BudgetMetrics AS (
        SELECT
            cd.JIRA_ID,
            cd.DEALERSHIP_NAME,
            cd.STATE,
            cd.BRAND,
            SUM(bam.Total_Client_Budget) AS total_budget
        FROM CLIENT_DETAILS cd
        LEFT JOIN CLIENT_BUDGETS bam
            ON cd.JIRA_ID = bam.JIRA_ID
            AND DATE_FROM_PARTS(bam.YEAR, bam.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        {where_clause}
        GROUP BY cd.JIRA_ID, cd.DEALERSHIP_NAME, cd.STATE, cd.BRAND
    ),
    SalesMetrics AS (
        SELECT
            cd.JIRA_ID,
            SUM(inv.NEW_SALES_TOTAL + inv.USED_SALES_TOTAL) AS total_sales
        FROM CLIENT_DETAILS cd
        LEFT JOIN INVENTORY_AND_SALES inv
            ON cd.JIRA_ID = inv.JIRA_ID
            AND DATE_FROM_PARTS(inv.YEAR, inv.MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        {where_clause}
        GROUP BY cd.JIRA_ID
    ),
    Combined AS (
        SELECT
            bm.JIRA_ID,
            bm.DEALERSHIP_NAME,
            bm.STATE,
            bm.BRAND,
            bm.total_budget,
            sm.total_sales,
            CASE
                WHEN sm.total_sales = 0 THEN 0
                ELSE bm.total_budget / sm.total_sales
            END AS cost_per_sale
        FROM BudgetMetrics bm
        LEFT JOIN SalesMetrics sm
            ON bm.JIRA_ID = sm.JIRA_ID
    )
    SELECT
        JIRA_ID,
        ROW_NUMBER() OVER (ORDER BY total_budget DESC) AS "Rank",
        DEALERSHIP_NAME AS "Dealership Name",
        STATE,
        BRAND,
        total_budget AS "Total Budget",
        ROUND(cost_per_sale, 2) AS "Cost per Sale"
    FROM Combined
    ORDER BY "Rank"
    {limit_clause};
    """

    filters = []
    if state_filter:
        filters.append(f"cd.STATE = '{state_filter}'")
    if brand_filter:
        filters.append(f"cd.BRAND = '{brand_filter}'")

    where_clause = " WHERE " + " AND ".join(filters) if filters else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    return base_query.format(where_clause=where_clause, limit_clause=limit_clause)


@with_description(leaderboard_description_generator("website"))
def get_website_traffic_leaderboard_query(*, state_filter: str = None, brand_filter: str = None, limit: int = 10):
    base_query = """
    WITH GA_PreAgg AS (
        SELECT
            JIRA_ID,
            SUM(SESSIONS) AS total_sessions
        FROM GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024
        WHERE DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY JIRA_ID
    )
    SELECT
        cd.JIRA_ID,
        cd.DEALERSHIP_NAME,
        cd.STATE,
        cd.BRAND,
        COALESCE(ga.total_sessions, 0) AS total_sessions
    FROM CLIENT_DETAILS cd
    LEFT JOIN GA_PreAgg ga ON cd.JIRA_ID = ga.JIRA_ID
    {where_clause}
    ORDER BY total_sessions DESC
    {limit_clause};
    """

    filters = []
    if state_filter:
        filters.append(f"cd.STATE = '{state_filter}'")
    if brand_filter:
        filters.append(f"cd.BRAND = '{brand_filter}'")

    where_clause = " WHERE " + " AND ".join(filters) if filters else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    return base_query.format(where_clause=where_clause, limit_clause=limit_clause)


@with_description("Average monthly budget data for the given dealership")
def get_avg_monthly_budget_query():
    return """
    SELECT
        JIRA_ID,
        AVG(Total_Client_Budget) AS AVG_MONTHLY_BUDGET
    FROM CLIENT_BUDGETS
    WHERE
      JIRA_ID = %(jira_id)s
      AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY JIRA_ID
    """


@with_description("Average monthly sales data for the given dealership")
def get_avg_monthly_sales_query():
    return """
    SELECT AVG(NEW_SALES_TOTAL + USED_SALES_TOTAL) AS AVG_MONTHLY_SALES
    FROM INVENTORY_AND_SALES
    WHERE JIRA_ID = %(jira_id)s
      AND DATE_FROM_PARTS(YEAR, MONTH, 1) BETWEEN %(start_date)s AND %(end_date)s
    """


@with_description("Date ranges of all available for each data source")
def get_table_date_ranges_query():
    """Show available data data ranges"""

    def get_ymd_sql(table: str, table_name: str) -> str:
        return cleandoc(f"""
        SELECT
            '{table_name}' AS "Data Source",
            MIN(TO_DATE(YEAR || '-' || LPAD(MONTH, 2, '0') || '-' || LPAD(DAY, 2, '0'))) AS "Min Date",
            MAX(TO_DATE(YEAR || '-' || LPAD(MONTH, 2, '0') || '-' || LPAD(DAY, 2, '0'))) AS "Max Date"
        FROM
            {table}
        WHERE
            JIRA_ID = %(jira_id)s
        """)

    def get_ym_sql(table: str, table_name: str) -> str:
        return cleandoc(f"""
        SELECT
            '{table_name}' AS "Data Source",
            MIN(TO_DATE(YEAR || '-' || LPAD(MONTH, 2, '0') || '-01')) AS "Min Date",
            MAX(LAST_DAY(TO_DATE(YEAR || '-' || LPAD(MONTH, 2, '0') || '-01'))) AS "Max Date"
        FROM
            {table}
        WHERE
            JIRA_ID = %(jira_id)s
        """)

    def get_current_sql(table: str, table_name: str) -> str:
        return cleandoc(f"""
        SELECT
            '{table_name}' AS "Data Source",
            DATE_TRUNC('MONTH', CURRENT_DATE()) AS "Min Date",
            LAST_DAY(CURRENT_DATE()) AS "Max Date"
        FROM
            (SELECT 1 FROM {table} WHERE JIRA_ID = %(jira_id)s LIMIT 1) dummy
        """)

    table_configs = [
        ('ATLAS_CALLS_MEETINGS', 'month', 'Atlas Calls Data'),
        ('INVENTORY_AND_SALES', 'month', 'Monthly Inventory and Sales'),
        ('INVENTORY_DAILY_CLIENTS', 'day', 'Daily Inventory'),
        ('CLIENT_BUDGETS', 'month', 'Client Budgets'),
        ('CLIENT_DETAILS', 'current', 'Client Details'),
        ('GA4_PAGEANALYTICS_MONTHLY', 'month', 'GA4 Page Data'),
        ('GA4_WEBSITESOURCES_ALLCLIENTS_MONTHLY_2024',
            'month', 'GA4 Traffic Source Data'),
        ('GOOGLEADSDATA_BYCAMPAIGN_ALLCLIENTS',
            'month', 'Google Ads Campaign Data'),
        ('MYSTERY_SHOPS_CLIENT_RESPONSE_TIMES', 'month',
         'Mystery Shops Client Response Times'),
        ('MYSTERY_SHOPS_CLIENT_RESPONSE_TRANSCRIPTS', 'month',
         'Mystery Shops Client Response Transcripts'),
    ]

    sql_parts = []
    for table, table_type, table_name in table_configs:
        match table_type:
            case 'day':
                sql_parts.append(get_ymd_sql(table, table_name))
            case 'month':
                sql_parts.append(get_ym_sql(table, table_name))
            case 'current':
                sql_parts.append(get_current_sql(table, table_name))
    final_sql = "\n\nUNION ALL\n\n".join(
        sql_parts) + "\n\nORDER BY\n    \"Data Source\";"
    return "\n".join([line for line in final_sql.splitlines() if line.strip()])
