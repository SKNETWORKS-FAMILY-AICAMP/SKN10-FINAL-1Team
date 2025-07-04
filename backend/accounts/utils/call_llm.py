
import os
import logging
import json
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
# def call_llm(prompt: str, use_cache: bool = True) -> str:
#     import google.generativeai as genai # Ensure genai is imported
#     # Log the prompt
#     logger.info(f"PROMPT: {prompt}")

#     # Check cache if enabled
#     if use_cache:
#         # Load cache from disk
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 logger.warning(f"Failed to load cache, starting with empty cache")

#         # Return from cache if exists
#         if prompt in cache:
#             logger.info(f"RESPONSE: {cache[prompt]}")
#             return cache[prompt]

#     # Call the LLM if not in cache or cache disabled
#     # Option 1: Using Vertex AI (if GEMINI_PROJECT_ID and GEMINI_LOCATION are set)
#     # client = genai.Client(
#     #     vertexai=True,
#     #     project=os.getenv("GEMINI_PROJECT_ID", "your-project-id"), # Replace with your actual project ID if not using env var
#     #     location=os.getenv("GEMINI_LOCATION", "us-central1") # Replace with your actual location if not using env var
#     # )

#     # Option 2: Using AI Studio API Key (ensure GEMINI_API_KEY is set)
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         logger.error("GEMINI_API_KEY environment variable not set.")
#         raise ValueError("GEMINI_API_KEY environment variable not set.")
    
#     genai.configure(api_key=api_key)
    
#     # For text-only input, use the gemini-pro model
#     model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest") # Using a common, generally available model
#     model = genai.GenerativeModel(model_name)
    
#     try:
#         response = model.generate_content(prompt)
#         response_text = response.text
#     except Exception as e:
#         logger.error(f"Error during Gemini API call: {e}")
#         # Try to get more details from the exception if available
#         if hasattr(e, 'message'):
#             logger.error(f"Error message: {e.message}")
#         if hasattr(response, 'prompt_feedback'):
#              logger.error(f"Prompt Feedback: {response.prompt_feedback}")
#         if hasattr(response, 'candidates') and response.candidates:
#              logger.error(f"Finish Reason: {response.candidates[0].finish_reason}")
#              logger.error(f"Safety Ratings: {response.candidates[0].safety_ratings}")
#         raise # Re-raise the exception after logging

#     # Log the response
#     logger.info(f"RESPONSE: {response_text}")

#     # Update cache if enabled
#     if use_cache:
#         # Load cache again to avoid overwrites
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 pass

#         # Add to cache and save
#         cache[prompt] = response_text
#         try:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(cache, f, ensure_ascii=False, indent=4) # Added indent for readability
#         except Exception as e:
#             logger.error(f"Failed to save cache: {e}")

#     return response_text


# # Use Azure OpenAI
# def call_llm(prompt, use_cache: bool = True):
#     from openai import AzureOpenAI
#     endpoint = "https://<azure openai name>.openai.azure.com/"
#     deployment = "<deployment name>"

#     subscription_key = "<azure openai key>"
#     api_version = "<api version>"

#     client = AzureOpenAI(
#         api_version=api_version,
#         azure_endpoint=endpoint,
#         api_key=subscription_key,
#     )

#     r = client.chat.completions.create(
#         model=deployment,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         max_completion_tokens=40000,
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

# # Use Anthropic Claude 3.7 Sonnet Extended Thinking
# def call_llm(prompt, use_cache: bool = True):
#     from anthropic import Anthropic
#     client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key"))
#     response = client.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=21000,
#         thinking={
#             "type": "enabled",
#             "budget_tokens": 20000
#         },
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.content[1].text

# Use OpenAI (e.g., gpt-3.5-turbo, gpt-4)
def call_llm(prompt: str, use_cache: bool = True) -> str:
    from openai import OpenAI
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, starting with empty cache")
        if prompt in cache:
            logger.info(f"RESPONSE (from cache): {cache[prompt]}")
            return cache[prompt]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    
    # Default to gpt-3.5-turbo, can be overridden by OPENAI_MODEL env var
    # Other options: "gpt-4", "gpt-4-turbo-preview", etc.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o") 
    logger.info(f"Using OpenAI model: {openai_model}")

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}]
            # You might need to adjust other parameters like max_tokens, temperature, etc.
            # response_format={"type": "text"} # This might not be needed for all models or could cause issues
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        # Attempt to log more detailed error information if available
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_data = e.response.json()
                logger.error(f"OpenAI API Error Details: {err_data}")
            except json.JSONDecodeError:
                logger.error(f"OpenAI API Error (non-JSON response): {e.response.text}")
        elif hasattr(e, 'message'):
             logger.error(f"Error message: {e.message}")
        raise

    logger.info(f"RESPONSE: {response_text}")

    if use_cache:
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass
        cache[prompt] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text

# Use OpenRouter API
# def call_llm(prompt: str, use_cache: bool = True) -> str:
#     import requests
#     # Log the prompt
#     logger.info(f"PROMPT: {prompt}")

#     # Check cache if enabled
#     if use_cache:
#         # Load cache from disk
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 logger.warning(f"Failed to load cache, starting with empty cache")

#         # Return from cache if exists
#         if prompt in cache:
#             logger.info(f"RESPONSE: {cache[prompt]}")
#             return cache[prompt]

#     # OpenRouter API configuration
#     api_key = os.getenv("OPENROUTER_API_KEY", "")
#     model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#     }

#     data = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}]
#     }

#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",
#         headers=headers,
#         json=data
#     )

#     if response.status_code != 200:
#         error_msg = f"OpenRouter API call failed with status {response.status_code}: {response.text}"
#         logger.error(error_msg)
#         raise Exception(error_msg)
#     try:
#         response_text = response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         error_msg = f"Failed to parse OpenRouter response: {e}; Response: {response.text}"
#         logger.error(error_msg)        
#         raise Exception(error_msg)
    

#     # Log the response
#     logger.info(f"RESPONSE: {response_text}")

#     # Update cache if enabled
#     if use_cache:
#         # Load cache again to avoid overwrites
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 pass

#         # Add to cache and save
#         cache[prompt] = response_text
#         try:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(cache, f)
#         except Exception as e:
#             logger.error(f"Failed to save cache: {e}")

#     return response_text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")