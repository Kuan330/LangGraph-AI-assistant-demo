"""
This is a simple test to ensure that the DuckDuckGo search tool from langchain-community is working correctly.
"""

from langchain_community.tools import DuckDuckGoSearchRun

try:
    print("Attempting to search DuckDuckGo...")
    search = DuckDuckGoSearchRun()
    result = search.invoke("What is the capital of France?")
    print("\n This is Working:")
    print(result)
except Exception as e:
    print("\n This is NOT Working, check the error below:")
    print(e)