import os
import openai
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def call_openai_api(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call OpenAI API with a simple prompt.
    
    Args:
        prompt (str): The prompt to send to OpenAI
        api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
    
    Returns:
        str: The response from OpenAI API
        
    Raises:
        ValueError: If no API key is provided
        Exception: For other API errors
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Either pass it as a parameter or set OPENAI_API_KEY environment variable.")
    
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            #model="gpt-3.5-turbo",  # You can change this to other models like "gpt-4"
            model="gpt-4o",  # You can change this to other models like "gpt-4"
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # Adjust as needed
            temperature=0.7   # Adjust creativity level (0.0 = deterministic, 1.0 = very creative)
        )
        
        # Extract and return the response
        return response.choices[0].message.content
        
    except openai.AuthenticationError:
        raise Exception("Invalid API key. Please check your OpenAI API key.")
    except openai.RateLimitError:
        raise Exception("Rate limit exceeded. Please try again later.")
    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")

# Example usage
if __name__ == "__main__":
    # Example 1: Using environment variable for API key
    try:
        response = call_openai_api("What is the capital of France?")
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using the prompt from your prompt.txt file
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        
        # You can use a specific part of the prompt or the whole thing
        sample_prompt = "Explain what this prompt is for: " + prompt_content[:200] + "..."
        response = call_openai_api(sample_prompt)
        print("\nPrompt Analysis Response:", response)
    except FileNotFoundError:
        print("prompt.txt file not found")
    except Exception as e:
        print(f"Error: {e}")
