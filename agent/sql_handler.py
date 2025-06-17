'''
File: sql_handler.py

SQL execution state machine logic

Copyright 2025 C-4 Analytics, LLC
'''

import logging
from dataclasses import dataclass
from enum import Enum
from tempfile import TemporaryFile
from typing import IO, Any, Callable, Sequence

import pandas as pd
import requests
from snowflake.connector import DictCursor
from transitions import Machine, State

from cortex import CortexConfig, CortexHandler, SnowflakeConfig
from error_checker import QuerySpotCheckHandler

logger = logging.getLogger("fusion")


class QueryState(Enum):
    GENERATE_SQL_CORTEX = 0
    GENERATE_SQL_CLAUDE = 1
    EXECUTE_SQL = 2
    HANDLE_ERROR = 3
    HANDLE_NO_DATA = 4
    DONE = 5
    CORTEX_RETURNED_SUGGESTIONS = 6
    DATA_ANALYSIS_REQUIRED = 7
    VALIDATE_SQL = 8


@dataclass
class QueryResponse:
    content: list[dict]
    sql: str | None = None
    df: pd.DataFrame | None = None
    suggestions: list[str] | None = None


class QueryStateMachine:
    """Manage the state of a SQL query and invoke retry logic as needed"""

    def __init__(self, cortex_handler: CortexHandler, error_checker: QuerySpotCheckHandler, user_question, max_retries=10):

        self.cortex_handler: CortexHandler = cortex_handler
        self.query_error_checker: QuerySpotCheckHandler = error_checker
        self.user_question: str = user_question

        # disable all retry logic
        if max_retries == 0:
            self.max_cortex_retries = 0
            self.max_retries = 0
        else:
            self.max_cortex_retries = 3
            self.max_retries: int = max(
                max_retries - self.max_cortex_retries, 1)

        self.attempt: int = 0

        self.error_message: str = None
        self.query_response: QueryResponse = None

        states = [
            State(name=QueryState.GENERATE_SQL_CORTEX,
                  on_enter='on_enter_generate_sql_cortex'),
            State(name=QueryState.GENERATE_SQL_CLAUDE,
                  on_enter='on_enter_generate_sql_claude'),
            State(name=QueryState.CORTEX_RETURNED_SUGGESTIONS,
                  on_enter='on_cortex_returned_suggestions',
                  final=True),
            State(name=QueryState.EXECUTE_SQL,
                  on_enter='on_enter_execute_sql'),
            State(name=QueryState.VALIDATE_SQL,
                  on_enter='on_enter_validate_sql'),
            State(name=QueryState.HANDLE_ERROR,
                  on_enter='on_enter_handle_error'),
            State(name=QueryState.HANDLE_NO_DATA,
                  on_enter='on_enter_handle_no_data'),
            State(name=QueryState.DATA_ANALYSIS_REQUIRED,
                  on_enter='on_enter_data_analysis_required',
                  final=True),
            State(name=QueryState.DONE, final=True),
        ]

        transitions = [
            ['start', '*', QueryState.GENERATE_SQL_CORTEX],
            ['cortex_returned_suggestions', QueryState.GENERATE_SQL_CORTEX,
                QueryState.CORTEX_RETURNED_SUGGESTIONS],
            ['generate_sql_retry', QueryState.GENERATE_SQL_CORTEX,
                QueryState.GENERATE_SQL_CORTEX],
            ['generate_sql_failure', QueryState.GENERATE_SQL_CORTEX,
                QueryState.GENERATE_SQL_CLAUDE],
            ['generate_sql_complete', QueryState.GENERATE_SQL_CORTEX,
                QueryState.VALIDATE_SQL],
            ['generate_sql_complete', QueryState.GENERATE_SQL_CLAUDE,
                QueryState.VALIDATE_SQL],
            ['generate_sql_retry', QueryState.GENERATE_SQL_CLAUDE,
                QueryState.GENERATE_SQL_CLAUDE],
            ['validate_sql_complete', QueryState.VALIDATE_SQL,
                QueryState.EXECUTE_SQL],
            ['success_with_data', QueryState.EXECUTE_SQL, QueryState.DONE],
            ['success_no_data', QueryState.EXECUTE_SQL, QueryState.HANDLE_NO_DATA],
            ['error_encountered', QueryState.EXECUTE_SQL, QueryState.HANDLE_ERROR],
            ['retry_after_error', QueryState.HANDLE_ERROR, QueryState.EXECUTE_SQL],
            ['exhausted_retries', QueryState.HANDLE_ERROR, QueryState.DONE],
            ['revise_query', QueryState.HANDLE_NO_DATA, QueryState.EXECUTE_SQL],
            ['data_analysis_required', QueryState.HANDLE_NO_DATA,
                QueryState.DATA_ANALYSIS_REQUIRED],
            ['no_data_is_valid', QueryState.HANDLE_NO_DATA, QueryState.DONE],
            ['exhausted_retries', QueryState.HANDLE_NO_DATA, QueryState.DONE]
        ]

        self.machine = Machine(
            model=self, states=states, transitions=transitions, initial=QueryState.GENERATE_SQL_CORTEX)

    def get_cortex_sql(self) -> tuple[bool, QueryResponse]:
        """Send a query to Cortex and parse the response with SQL if available.
        Returns a bool indicating success, and a QueryResponse object
        containing the parsed response data.
        """

        try:
            response = self.cortex_handler.send_message(self.user_question)
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return False, None
        except Exception as e:
            logger.exception(f"Failed request with error: {e}")
            return False, None

        logger.info(f"Cortex Response::: {response}")
        if not isinstance(response, dict):
            return False, None

        # Handle errors
        if "error" in response:
            return False, None

        # Safely get message content
        message_data = response.get("message", {})
        if not message_data:
            return False, None
        elif not isinstance(message_data, dict):
            return False, None

        content = message_data.get("content", [])
        self.outputs = content

        cortex_response = QueryResponse(content)

        response_ok = False

        for item in content:
            match item["type"]:
                case "suggestions":
                    cortex_response.suggestions = item["suggestions"]
                    response_ok = True
                case "sql":
                    cortex_response.sql = item["statement"]
                    response_ok = True

        return response_ok, cortex_response

    def update_sql_response(self, new_sql: str, overwrite: bool = False):
        """Update the response object to reflect the latest SQL"""

        if self.query_response is None:
            self.query_response = QueryResponse(
                [{"type": "sql", "statement": new_sql}],
                new_sql
            )
            return

        self.query_response.sql = new_sql

        if overwrite:
            # overwrite the content block
            self.query_response.content = [
                {"type": "sql", "statement": new_sql}]
        else:
            # just update in-place and preserve other content
            for item in self.query_response.content:
                if item["type"] == "sql":
                    item["statement"] = new_sql
                    return

    def on_cortex_returned_suggestions(self):
        """
        Cortex Analyst returned suggestions instead of SQL
        """

    def on_enter_generate_sql_cortex(self):
        """
        Generate SQL from user question using Cortex Analyst.
        """

        logger.info(
            f"[State: GENERATE_SQL_CORTEX] Attempt #{self.attempt + 1}")

        ok, self.query_response = self.get_cortex_sql()
        if not ok:
            logger.warning("Cortex failed to generate SQL - retrying.")
            self.attempt += 1
            if self.attempt >= self.max_cortex_retries:
                logger.error(
                    "Cortex Analyst Max retries exceeded. Using fallback.")
                # Re-attempt generation with claude
                self.generate_sql_failure()
                return

            # Re-attempt generation with Cortex Analyst
            self.generate_sql_retry()
            return
        elif self.query_response.suggestions is not None:
            # cortex returned suggestions instead of SQL
            self.cortex_returned_suggestions()
            return

        self.generate_sql_complete()

    def on_enter_generate_sql_claude(self):
        """
        Generate SQL from user question using Cortex Analyst.
        """

        logger.info(
            f"[State: GENERATE_SQL_CLAUDE] Attempt #{self.attempt + 1}")

        try:
            new_sql = self.query_error_checker.generate_query(
                user_intent=self.user_question)
            self.update_sql_response(new_sql, overwrite=True)
            self.generate_sql_complete()
        except Exception as error_message:
            self.error_message = error_message
            logger.error(
                f"Failed to generate query with Claude: {error_message}")

            self.attempt += 1
            if self.attempt >= self.max_cortex_retries:
                logger.error("Claude Max retries exceeded.")
                self.exhausted_retries()
                return

            # Re-attempt generation with Claude
            self.generate_sql_retry()

    def on_enter_validate_sql(self):
        """
        Validate the previously generated SQL and decide next transition.
        """
        logger.info("[State: VALIDATE] Validating SQL...")
        try:
            sql_query = self.query_response.sql
            validated_sql_query = self.query_error_checker.validate_query(
                sql_query, self.user_question)
            if sql_query != validated_sql_query:
                logger.warning(
                    f"Validator returned a different query:\nORIGINAL:\n{sql_query}\nVALIDATED:\n{validated_sql_query}\n")
                self.update_sql_response(validated_sql_query)

            # pass validated SQL to executor
            self.validate_sql_complete()

        except Exception as error_message:
            self.error_message = error_message
            logger.error(f"SQL Error: {error_message}")
            self.error_encountered()

    def on_enter_execute_sql(self):
        """
        Execute the previously generated SQL and decide next transition.
        """
        logger.info("[State: EXECUTE_SQL] Executing SQL...")
        try:
            df = pd.read_sql(self.query_response.sql,
                             self.cortex_handler.snowflake_connection)
            if df.empty:
                logger.info("SQL GOT NO DATA!")
                self.success_no_data()
            else:
                logger.info("SQL GOT DATA!")
                self.success_with_data()
                self.query_response.df = df

        except Exception as error_message:
            self.error_message = error_message
            logger.error(f"SQL Error: {error_message}")
            self.error_encountered()

    def on_enter_handle_error(self):
        """
        Handle an error from the last execution.
        Possibly ask LLM to revise the SQL or stop if we've hit our max retries.
        """
        logger.info(f"[State: HANDLE_ERROR]")
        self.attempt += 1
        if self.attempt >= self.max_retries:
            logger.error("Max retries exceeded. Exiting without results.")
            self.exhausted_retries()
            return

        # Ask LLM for a revised SQL
        logger.info(
            f"Asking LLM to revise the query... (Attempt {self.attempt + 1} / {self.max_retries}).")
        # self.conversation_history.append({"role": "system", "content": f"SQL Error: {self.error_message}"})

        new_sql = self.query_error_checker.fix_broken_query(
            sql=self.query_response.sql,
            user_intent=self.user_question,
            error=self.error_message)
        self.update_sql_response(new_sql)
        # Retry by going back to EXECUTE_SQL
        self.retry_after_error()

    def on_enter_handle_no_data(self):
        """
        Query returned zero rows. Ask the LLM if the query is correct or incorrect.
        If incorrect, we attempt to revise. If correct or we exceed attempts, we end.
        """
        logger.info("[State: HANDLE_NO_DATA] Handling no-data scenario.")
        self.attempt += 1
        if self.attempt >= self.max_retries:
            logger.error("Max retries exceeded. Exiting without results.")
            self.exhausted_retries()
            return

        # Ask LLM if no data is correct or if a revised query is needed
        try:
            logger.warning("No data returned - running spot checker")
            problem, need_more_info, fix_instructions = self.query_error_checker.spot_check_query(
                sql_query=self.query_response.sql, user_intent=self.user_question)
            logger.info(
                f"Spot checker results: Problem: {problem} Analysis Required {need_more_info} Fix Instructions: {fix_instructions}")
        except Exception as exc:
            logger.exception("Failed to run spot checker")
            # if we can't run the spot checker, just treat is as valid and continue
            self.no_data_is_valid()
            return

        # Decide if we want to revise or accept no data as final
        if problem:
            # revise the SQL based on LLM feedback
            new_sql = self.query_error_checker.rewrite_query(
                sql_query=self.query_response.sql,
                user_intent=self.user_question,
                problem_description=fix_instructions)
            self.update_sql_response(new_sql)
            self.revise_query()
        elif need_more_info:
            # The LLM suggests we need to look at the data first

            logger.warning("Generating validation queries")

            data_validation_queries = self.query_error_checker.get_data_validation_queries(
                user_intent=self.user_question,
                sql_query=self.query_response.sql,
                problem_description=fix_instructions)

            # print("@@@ Got validation queries:", data_validation_queries)

            validation_examples = []

            # only execute up to 3 validation queries
            for validation_query in data_validation_queries[:3]:
                try:
                    machine = QueryStateMachine(
                        self.cortex_handler,
                        self.query_error_checker,
                        validation_query,
                        max_retries=0)  # do not retry for validation questions

                    response = machine.run()
                    if response.df is not None:
                        # print(
                        #     f"@@@@@@ Got sql:\n{response.sql}\ndata:{response}\nfor query {validation_query}")

                        # limit the df size to avoid polluting the context
                        result = response.df.head(
                            100).to_dict(orient='records')
                        if len(response.df) > 100:
                            # indicate that result is truncated
                            result.append("etc...")

                        validation_examples.append({"query": response.sql,
                                                    "result": result})
                except Exception:
                    logger.exception("Failed to run validation query")

            # revise the SQL based on LLM feedback, if possible
            if validation_examples:
                fix_instructions += "\nHere are some examples of real data:\n"

                for example in validation_examples:
                    fix_instructions += f"Query:\n{example['query']}Result:{example['result']}"

                new_sql = self.query_error_checker.rewrite_query(
                    sql_query=self.query_response.sql,
                    user_intent=self.user_question,
                    problem_description=fix_instructions)

                self.update_sql_response(new_sql)

                self.revise_query()
            else:
                # If we can't resolve on our own, defer to the user
                suggestions = [
                    f"The query returned no data. Further analysis is required. {fix_instructions}"]
                # need to add to content list or it won't get parsed
                self.query_response.content.append(
                    {"type": "suggestions", "suggestions": suggestions})
                self.query_response.suggestions = suggestions
                self.data_analysis_required()
        else:
            # The LLM suggests there's genuinely no data
            self.no_data_is_valid()

    def on_enter_data_analysis_required(self):
        """
        LLM determined that analysis of the data is required before querying.
        """

    def run(self) -> QueryResponse:
        """
        Run the workflow from the initial state until we reach 'DONE'.
        """
        try:
            self.start()
            while not self.machine.get_state(self.state).final:
                logger.warning("STATE:", self.state)
                pass
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user.")
            raise
        except Exception:
            logger.exception("Failed to execute query state machine")
            return QueryResponse([{"type": "error", "error": "Failed to retrieve data."}])

        if self.query_response is None:
            return QueryResponse([{"type": "error", "error": "Failed to retrieve data."}])

        return self.query_response


class SnowflakeSQLHandler:
    """Wrapper class to handle Snowflake data retrieval"""

    def __init__(self, snowflake_config: SnowflakeConfig, cortex_config: CortexConfig):
        self.snowflake_config = snowflake_config
        self.cortex_config = cortex_config
        self.cortex_handler = CortexHandler(self.snowflake_config, self.cortex_config)
        self.cortex_handler.open()
        self.df_row_limit: int = 50  # max number of rows to return as
        self.query_error_checker = QuerySpotCheckHandler()
    
    def OpenifClosed(self):
        if self.cortex_handler is None:
            self.cortex_handler = CortexHandler(self.snowflake_config, self.cortex_config)
        self.cortex_handler.open()
        self.df_row_limit: int = 50  # max number of rows to return as
        self.query_error_checker = QuerySpotCheckHandler()

    def close(self):
        if self.cortex_handler is not None:
            self.cortex_handler.close()
            self.cortex_handler = None

    def execute_query(self, user_question: str,
                      create_file_callback: Callable[[IO], None],
                      log_content_callback: Callable[[
                          list[dict]], None] = None,
                      max_retries: int = 10) -> list[dict[str, Any]]:
        """Execute the Snowflake text-to-SQL pipeline and engage retry logic as needed

        Returns a JSON-friendly dict structure for LLM tool calling purposes

        """

        machine = QueryStateMachine(
            self.cortex_handler,
            self.query_error_checker,
            user_question,
            max_retries=max_retries)

        response = machine.run()

        # need to format the content to preserve logical ordering of output
        output_array = []

        # raw content for logging purposes
        raw_content = [{"type": "text", "text": "**Cortex Agent:**"}]

        for item in response.content:
            output = {}

            raw_content.append(item)

            match item["type"]:

                case "text":
                    item_text = item["text"]
                    output["text"] = item_text
                case "suggestions":
                    output["suggestions"] = response.suggestions
                case "sql":
                    output = {"sql": response.sql}
                    if response.df is not None:

                        raw_content.append(
                            {"type": "dataframe", "dataframe": response.df})

                        # return under 50 rows directly as json, otherwise use file upload
                        if len(response.df) <= self.df_row_limit:
                            output["dataframe"] = response.df.to_json(
                                orient='records')
                        else:
                            # if dataframe is too large, create an intermediate file
                            with TemporaryFile("wb+") as fp:
                                response.df.to_csv(fp)
                                fp.seek(0)
                                file_id = create_file_callback(fp)
                                output["file_id"] = file_id
                                output[
                                    "message"] = f"The data has been downloaded to a file with ID '{file_id}'"

            output_array.append(output)

        log_content_callback(raw_content)

        return {"resultsArray": output_array}

    def execute_query_raw(self, query: str, params: Sequence[Any] | dict[Any, Any] | None = None) -> pd.DataFrame:
        """Execute a raw SQL query with the given parameters and return a dataframe"""
        max_retry =2
        attempt = 0
        if params is None:
            params = ()
        with self.cortex_handler.snowflake_connection.cursor(DictCursor) as cursor:
            while attempt < max_retry:
                try:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    df = pd.DataFrame(rows)
                    return df
                except Exception as e:
                    attempt += 1
                    self.OpenifClosed()
                    if attempt == max_retry:
                        logger.error(f"Failed to execute query: {e}")
                        raise

