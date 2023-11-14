from main import get_llm_response

from credentials.api.openai import OPENAI_API_KEY
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Query PDF Source
QUERY = "Describe sktime"
get_llm_response(QUERY)