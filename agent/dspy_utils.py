'''
File: dspy_utils.py

Generic utilities for dspy

Copyright 2025 C-4 Analytics, LLC
'''

import os

import dspy

# define the various LMs
claude_3_7_sonnet = dspy.LM(
    'claude-3-7-sonnet-20250219', api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=0.5)

claude_3_7_sonnet_high_temp = dspy.LM(
    'claude-3-7-sonnet-20250219', api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=1.0)

o3_mini_high = dspy.LM('openai/o3-mini', api_key=os.getenv('OPENAI_API_KEY'), temperature=1.0, max_tokens=25000, reasoning_effort="high")

o3_mini_low = dspy.LM('openai/o3-mini', api_key=os.getenv('OPENAI_API_KEY'), temperature=1.0, max_tokens=25000, reasoning_effort="low")

# o3_mini_high = dspy.LM('openai/o3-mini', api_key=os.getenv('OPENAI_API_KEY'), temperature=1.0, reasoning_effort="high")

# o3_mini_low = dspy.LM('openai/o3-mini', api_key=os.getenv('OPENAI_API_KEY'), temperature=1.0, reasoning_effort="low")

gpt_4o_mini = dspy.LM('openai/gpt-4o-mini',
                      api_key=os.getenv('OPENAI_API_KEY'), temperature=1.0)
