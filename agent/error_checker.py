'''
File: error_checker.py

Error checking logic for snowflake queries

Copyright 2025 C-4 Analytics, LLC
'''

import logging
import os
from enum import Enum
import dspy
import dspy.predict

from dspy_utils import claude_3_7_sonnet, o3_mini_high, o3_mini_low

logger = logging.getLogger("fusion")

current_path=os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_path,"claude_semantic_model.yaml"), "r") as fp:
    semantic_model_text = fp.read()

SNOWFLAKE_SQL_INSTRUCTIONS = f"""Here is an overview of the Snowflake tables you can access:

{semantic_model_text}

When asked about the latest data, ensure that the latest month and year are explicitly identified by checking the maximum values in the table for both the MONTH and YEAR columns. The latest date is the highest year with the most recent month of that year.
DO NOT use subqueries in your answer. Instead represent them as common table expressions (CTEs).
You should join tables using the client's JIRA_ID.
"""


class SQLAnalysisResult(Enum):
    NO_ISSUES = 0
    LOGIC_ISSUE = 1
    DATA_VALIDATION_REQUIRED = 2


class QuerySpotCheckValidator(dspy.Signature):
    __doc__ = f"""Your task is to validate AI-generated Snowflake SQL queries and correct common mistakes as outline below:

    For context, clients are auto dealerships.

    When querying the client_details table:
    - Ensure that any string literal filters are against valid brand names. Here is a list of all valid brands: [Acura,Alfa_Romeo,Aston_Martin,Audi,BMW,Bentley,Buick,CDJR,Cadillac,Chevrolet,Chrysler,Command,Commercial,Commercial_Vans,Dodge,FIAT,Fiat,Ford,GMC,Genesis,Group,Group_Site,Honda,Hyundai,INFINITI,Infiniti,Isuzu,Jaguar,Jeep,Jerr-Dan,Karma,Kawasaki,Kia,Lamborghini,Land_Rover,Lexus,Lincoln,Lotus,MINI,MV-1,Maserati,Mazda,McLaren,Mercedes-Benz,Mitsubishi,Nissan,Other,Parts,Porsche,Powersports,Pre-Owned,Pre-owned,Quicklane,RAM,Ram,Rolls-Royce,Service,Sprinter,Subaru,Toyota,Tractor_(Other),Vespa,Volkswagen,Volvo]
    - Ensure that multiple spellings of the same brand are handled correctly (e.g. FIAT vs Fiat, RAM vs Ram, etc.), either with an OR clause, or case-insensitive logic such as ILIKE.
    - A client might have a brand like "Bentley, Lamborghini, Rolls-Royce" or "Lotus, Aston_Martin, Karma, Pre-Owned", so the filter must be robust enough to match these cases.

    A common mistake is to mix up the JIRA_ID and dealership name. A JIRA_ID should consist of only capital letters or numbers, with no special characters, spaces, etc.
    For the given AI-generated SQL and the user intent, return a corrected version, or the original if no corrections are needed.
    """

    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    sql: str = dspy.InputField()
    validated_sql: str = dspy.OutputField()


class QueryAnalyzer(dspy.Signature):
    __doc__ = f"""Your task is to analyze potentially problematic Snowflake SQL queries and identify whether it returned zero data due to an error with the query, or because there is genuinely no data.
    {SNOWFLAKE_SQL_INSTRUCTIONS}
    For example, consider the following scenario:
    - CLIENT_DETAILS has data through Jan 2025, but for client X only until Sept 2024
    - User asks about latest data for client X in Jan 2025
    - An improper query might first check all of CLIENT_DETAILS for the latest month/year, before filtering by client X. This will return no results. The query should instead filter by client X first, and then find the latest month/year's data.
    Take extra care when string literals are involved - exact matches usually cause issues (often fixed by replacing with a LIKE operator).
    DO NOT use subqueries in your answer. Instead represent them as common table expressions (CTEs)
    A common mistake is to mix up the JIRA_ID and dealership name. A JIRA_ID should consist of only capital letters or numbers, with no special characters, spaces, etc.
    Now, for the given query and its corresponding user intent, determine if it is problematic, and return your analysis.
    """
    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    sql: str = dspy.InputField(
        desc="The Snowflake SQL which returned no data.")
    analysis: str = dspy.OutputField(
        desc="A description of the problem and step-by-step instructions on how to correct the SQL, or N/A if no problem was identified.")


class QuerySpotCheckerOutputFormatter(dspy.Signature):
    """You are given an analysis of a Snowflake SQL query which resulted in no data returned. Your task is to decide which category it falls into using the following guidelines:
    LOGIC_ISSUE: Choose this if the analysis indicates that there is a mistake or flaw in the SQL query (for example, syntax errors or flawed logic in the SQL).
    DATA_VALIDATION_REQUIRED: Choose this if the analysis suggests that while the SQL query appears correct at first glance, there is uncertainty about whether the issue is due to the filter condition being too strict (or flawed) or simply because no data matches. For instance, if the analysis hints that the literal string comparison might be problematic (e.g., not accounting for case or spacing variations) and recommends further investigation (such as selecting distinct brand values to see what is valid).
    NO_ISSUES: Choose this only if the analysis is N/A.
    """
    analysis: str = dspy.InputField()
    result: SQLAnalysisResult = dspy.OutputField()


class QuerySpotChecker(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        # do not use COT wrapper if we are using o3 reasoning
        self.analyzer = dspy.Predict(QueryAnalyzer)
        # do not use COT wrapper if we are using o3 reasoning
        self.formatter = dspy.Predict(QuerySpotCheckerOutputFormatter)

    def forward(self, sql_query: str, user_intent: str):

        with dspy.context(lm=o3_mini_high):
            prediction = self.analyzer(sql=sql_query, user_intent=user_intent)

            print("!@#$!#@!#@ Analyzer prediciton:", prediction)

        with dspy.context(lm=o3_mini_low):
            formatted_result = self.formatter(analysis=prediction.analysis)

            print("!@#$!#@!#@ formatted_result:", formatted_result)

        match formatted_result.result:
            case SQLAnalysisResult.NO_ISSUES:
                return dspy.Prediction(problem=False, need_more_info=False, fix_instructions=None)
            case SQLAnalysisResult.LOGIC_ISSUE:
                return dspy.Prediction(problem=True, need_more_info=False, fix_instructions=prediction.analysis)
            case SQLAnalysisResult.DATA_VALIDATION_REQUIRED:
                return dspy.Prediction(problem=False, need_more_info=True, fix_instructions=prediction.analysis)
            case _:
                raise ValueError(
                    f"Spot checker returned an invalid prediction: {prediction}")


class QueryDataValidator(dspy.Signature):
    __doc__ = f"""You are given an analysis of a problematic Snowflake SQL query.
    {SNOWFLAKE_SQL_INSTRUCTIONS}

    To resolve any ambiguities, we need to inspect the actual values in the relevant column(s) to determine if there are variations or anomalies that the current filter does not account for.

    Your task is to generate a natural language query that will help diagnose the underlying issue by retrieving a summary of the data from the column(s) involved in the filter condition of the original query.

    """
    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    original_sql: str = dspy.InputField(
        desc="The Snowflake SQL which returned no data.")
    analysis: str = dspy.InputField(
        desc="A description of the problem and guidance on how to correct the SQL.")
    data_validation_queries: list[str] = dspy.OutputField(
        desc="1-3 natural language queries to resolve any filter ambiguities in the original SQL")


class QueryErrorHandler(dspy.Signature):
    __doc__ = f"""Your task is to fix broken Snowflake SQL queries which failed to execute due to some error.
    {SNOWFLAKE_SQL_INSTRUCTIONS}
    Now, for the given query, its corresponding user intent, and the provided error details, create a fixed version."""
    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    sql: str = dspy.InputField(
        desc="The Snowflake SQL which returned no data.")
    error: str = dspy.InputField(
        desc="The error raised when executing the SQL.")
    fix_instructions: str = dspy.OutputField(
        desc="A description of the problem and step-by-step instructions on how to fix the SQL.")


class QueryRewriter(dspy.Signature):
    __doc__ = """Your task is to rewrite the given problematic Snowflake SQL query to return the correct data. A query is considered problematic if it returned zero data.
    {SNOWFLAKE_SQL_INSTRUCTIONS}
    Now, for the given problematic query and its corresponding user intent, rewrite the query to produce the desired results.
    """
    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    original_sql: str = dspy.InputField(
        desc="The Snowflake SQL query which returned no data.")
    fix_instructions: str = dspy.InputField(
        desc="Instructions on how to revise the original SQL query.")
    new_sql: str = dspy.OutputField(
        desc="The revised query that will produce correct results.")


class QueryWriter(dspy.Signature):
    __doc__ = f"""Your task is to create a Snowflake SQL query to return the desired data based on a given user intent.
    {SNOWFLAKE_SQL_INSTRUCTIONS}
    Now, for the given user intent, construct the SQL that will produce the desired results.
    """
    user_intent: str = dspy.InputField(
        desc="The user's actual intended result of the query.")
    sql: str = dspy.OutputField(
        desc="The SQL query that will produce the desired results.")
    details: str = dspy.OutputField(
        desc="Brief explanation of the generated SQL.")


class QuerySpotCheckHandler:
    def __init__(self):
        self.query_checker = QuerySpotChecker()
        self.query_rewriter = dspy.ChainOfThought(QueryRewriter)
        self.query_generator = dspy.ChainOfThought(QueryWriter)
        # do not use COT with reasoning models
        self.query_error_handler = dspy.Predict(QueryErrorHandler)
        self.data_validator = dspy.Predict(QueryDataValidator)
        self.query_validator = dspy.Predict(QuerySpotCheckValidator)

    def validate_query(self, sql_query: str, user_intent: str) -> str:
        """Quickly validate the query prior to execution"""

        logger.info("Validating query")

        with dspy.context(lm=claude_3_7_sonnet):
            validation_result = self.query_validator(
                sql=sql_query, user_intent=user_intent)
            return validation_result.validated_sql

    def spot_check_query(self, sql_query: str, user_intent: str) -> tuple[bool, bool, str]:
        """Determine if the query returned zero data because it was malformed, or because there is a problem with the underlying data."""

        logger.info("Spot-checking query")

        spot_check_result = self.query_checker(
            sql_query=sql_query, user_intent=user_intent)

        return spot_check_result.problem, spot_check_result.need_more_info, spot_check_result.fix_instructions

    def generate_query(self, user_intent: str) -> str:
        """Create a SQL query from scratch to produce correct results."""

        logger.info("Generating new query")

        with dspy.context(lm=claude_3_7_sonnet):
            prediction = self.query_generator(user_intent=user_intent)
            return prediction.sql

    def fix_broken_query(self, user_intent: str, sql: str, error: str) -> str:
        """Rewrite the broken query so it executes properly."""

        logger.info("Rewriting syntactically incorrect query")

        with dspy.context(lm=o3_mini_high):
            error_handler_pred = self.query_error_handler(
                user_intent=user_intent, sql=sql, error=error)

        with dspy.context(lm=claude_3_7_sonnet):
            rewriter_pred = self.query_rewriter(
                original_sql=sql, user_intent=user_intent, fix_instructions=error_handler_pred.fix_instructions)
            return rewriter_pred.new_sql

    def rewrite_query(self, sql_query: str, user_intent: str, problem_description: str) -> str:
        """Rewrite the query to produce correct results."""

        logger.info("Rewriting logically incorrect query")

        with dspy.context(lm=claude_3_7_sonnet):
            prediction = self.query_rewriter(
                original_sql=sql_query, user_intent=user_intent, fix_instructions=problem_description)
            return prediction.new_sql

    def get_data_validation_queries(self, user_intent: str, sql_query: str, problem_description: str) -> list[str]:
        """Generate natural language queries to fetch validation data to resolve issues with the original SQL query"""

        logger.info("Generating data validation queries")

        with dspy.context(lm=o3_mini_high):
            prediction = self.data_validator(
                user_intent=user_intent, original_sql=sql_query, analysis=problem_description)
            return prediction.data_validation_queries
