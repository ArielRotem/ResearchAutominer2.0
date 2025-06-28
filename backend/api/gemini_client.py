
import google.generativeai as genai
import os

# Configure the Gemini API key
# In a real application, this would be loaded from a secure location.
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY" # Replace with your actual key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def generate_function(prompt: str, headers: list[str]) -> str:
    """Generates a Python function using the Gemini API."""
    
    model = genai.GenerativeModel('gemini-pro')
    
    full_prompt = f"""
    You are an expert Python programmer. Write a single Python function that takes a pandas DataFrame as input and performs the following task:

    Task: "{prompt}"

    The DataFrame has the following columns: {headers}

    The function should be named `custom_function`, take one argument `df`, and return a new pandas DataFrame.
    Do not include any other code, comments, or explanations outside of the function definition.
    """
    
    response = model.generate_content(full_prompt)
    return response.text
