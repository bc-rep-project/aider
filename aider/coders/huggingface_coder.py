import os
import json
import requests
from dotenv import load_dotenv
import yaml
from transformers import AutoTokenizer

from .base_coder import Coder
from .editblock_coder import find_original_update_blocks  # For parsing diffs
from .huggingface_prompts import HuggingFacePrompts


class HuggingFaceCoder(Coder):
    def __init__(self, *args, **kwargs):
        # Filter out the 'edit_format' and 'skip_model_availabily_check' argument from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}
        
        self.gpt_prompts = HuggingFacePrompts()  # Add this line

        model_id = kwargs.get('main_model')  # Get the model_id from kwargs
        tokenizer = AutoTokenizer.from_pretrained(model_id)  # Create tokenizer instance
        self.tokenizer = tokenizer  # Assign the instance to self.tokenizer
        super().__init__(*args, **filtered_kwargs)

        # Load .env file
        load_dotenv()

        # Accessing variables.
        api_key = os.getenv('HUGGINGFACE_API_KEY')

        # If the key doesn't exist in .env, try to load it from .yaml
        if api_key is None:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('HUGGINGFACE_API_KEY')

        # Hugging Face API URL and headers
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {api_key}"}  # Use the API key from .env or .yaml

    def send(self, messages, model=None, functions=None):
        # Format the prompt for the Hugging Face model
        prompt = self.format_prompt(messages)

        # Send the request to the Hugging Face API
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

        # Parse the JSON response
        response_json = response.json()

        # Extract the generated text or code edits
        # (Adapt this based on the actual response format of the Hugging Face model)
        generated_text = response_json.get("generated_text")

        # Apply updates and commit changes to git
        if generated_text:
            # If the response contains code edits in unified diff format, parse them
            if "```diff" in generated_text:
                edits = list(find_original_update_blocks(generated_text))
                edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
            else:
                # If the response is plain text, handle it accordingly
                # (You may need to adapt this logic based on the model's capabilities)
                self.io.ai_output(generated_text)
                edited = True  # Assuming that the output implies an edit

            if self.repo and self.auto_commits and not self.dry_run and edited:
                self.auto_commit()  # Use the base Coder's auto_commit method

    def format_prompt(self, messages):
        # Format the messages into a single prompt string
        # (You may need to adjust this based on the model's prompt requirements)
        prompt = ""
        for message in messages:
            role = message["role"].upper()
            content = message.get("content")
            if content:
                prompt += f"{role}: {content}\n"
        return prompt



# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         # Filter out the 'edit_format' and 'skip_model_availabily_check' argument from kwargs
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}
        
#         super().__init__(*args, **filtered_kwargs)
#         model_id = AutoTokenizer  # Define the model_id
#         self.tokenizer = tiktoken.get_encoding(model_id)  # Use AutoTokenizer

#         # Load .env file
#         load_dotenv()

#         # Accessing variables.
#         api_key = os.getenv('HUGGINGFACE_API_KEY')

#         # If the key doesn't exist in .env, try to load it from .yaml
#         if api_key is None:
#             with open('config.yaml', 'r') as f:
#                 config = yaml.safe_load(f)
#                 api_key = config.get('HUGGINGFACE_API_KEY')

#         # Hugging Face API URL and headers
#         self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.headers = {"Authorization": f"Bearer {api_key}"}  # Use the API key from .env or .yaml


#     def send(self, messages, model=None, functions=None):
#         # Format the prompt for the Hugging Face model
#         prompt = self.format_prompt(messages)

#         # Send the request to the Hugging Face API
#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#         # Parse the JSON response
#         response_json = response.json()

#         # Extract the generated text or code edits
#         # (Adapt this based on the actual response format of the Hugging Face model)
#         generated_text = response_json.get("generated_text")

#         # Apply updates and commit changes to git
#         if generated_text:
#             # If the response contains code edits in unified diff format, parse them
#             if "```diff" in generated_text:
#                 edits = list(find_original_update_blocks(generated_text))
#                 edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#             else:
#                 # If the response is plain text, handle it accordingly
#                 # (You may need to adapt this logic based on the model's capabilities)
#                 self.io.ai_output(generated_text)
#                 edited = True  # Assuming that the output implies an edit

#             if self.repo and self.auto_commits and not self.dry_run and edited:
#                 self.auto_commit()  # Use the base Coder's auto_commit method

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         # (You may need to adjust this based on the model's prompt requirements)
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt




# import os
# import json
# import requests
# from dotenv import load_dotenv
# import yaml
# from transformers import AutoTokenizer

# from .base_coder import Coder
# from .editblock_coder import find_original_update_blocks  # For parsing diffs

# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)  # Use AutoTokenizer

#         # Load .env file
#         load_dotenv()

#         # Accessing variables.
#         api_key = os.getenv('HUGGINGFACE_API_KEY')

#         # If the key doesn't exist in .env, try to load it from .yaml
#         if api_key is None:
#             with open('config.yaml', 'r') as f:
#                 config = yaml.safe_load(f)
#                 api_key = config.get('HUGGINGFACE_API_KEY')

#         # Hugging Face API URL and headers
#         self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.headers = {"Authorization": f"Bearer {api_key}"}  # Use the API key from .env or .yaml

#     def send(self, messages, model=None, functions=None):
#         # Format the prompt for the Hugging Face model
#         prompt = self.format_prompt(messages)

#         # Send the request to the Hugging Face API
#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#         # Parse the JSON response
#         response_json = response.json()

#         # Extract the generated text or code edits
#         # (Adapt this based on the actual response format of the Hugging Face model)
#         generated_text = response_json.get("generated_text")

#         # Apply updates and commit changes to git
#         if generated_text:
#             # If the response contains code edits in unified diff format, parse them
#             if "```diff" in generated_text:
#                 edits = list(find_original_update_blocks(generated_text))
#                 edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#             else:
#                 # If the response is plain text, handle it accordingly
#                 # (You may need to adapt this logic based on the model's capabilities)
#                 self.io.ai_output(generated_text)
#                 edited = True  # Assuming that the output implies an edit

#             if self.repo and self.auto_commits and not self.dry_run and edited:
#                 self.auto_commit()  # Use the base Coder's auto_commit method

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         # (You may need to adjust this based on the model's prompt requirements)
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt



# import json
# import requests

# from .base_coder import Coder
# from .editblock_coder import find_original_update_blocks  # For parsing diffs

# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Hugging Face API URL and headers
#         self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # Use OpenAI API key for Hugging Face

# def send(self, messages, model=None, functions=None):
#     # Format the prompt for the Hugging Face model
#     prompt = self.format_prompt(messages)

#     # Send the request to the Hugging Face API
#     response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#     # Parse the JSON response
#     response_json = response.json()

#     # Extract the generated text or code edits
#     # (Adapt this based on the actual response format of the Hugging Face model)
#     generated_text = response_json.get("generated_text")

#     # Apply updates and commit changes to git
#     if generated_text:
#         # If the response contains code edits in unified diff format, parse them
#         if "```diff" in generated_text:
#             edits = list(find_original_update_blocks(generated_text))
#             edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#         else:
#             # If the response is plain text, handle it accordingly
#             # (You may need to adapt this logic based on the model's capabilities)
#             self.io.ai_output(generated_text)
#             edited = True  # Assuming that the output implies an edit

#         if self.repo and self.auto_commits and not self.dry_run and edited:
#             self.auto_commit()  # Use the base Coder's auto_commit method

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         # (You may need to adjust this based on the model's prompt requirements)
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt



# import json

# from .base_coder import Coder
# from .editblock_coder import find_original_update_blocks  # For parsing diffs

# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Hugging Face API URL and headers
#         self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
#         self.headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # Use OpenAI API key for Hugging Face

#     def send(self, messages, model=None, functions=None):
#         # Format the prompt for the Hugging Face model
#         prompt = self.format_prompt(messages)

#         # Send the request to the Hugging Face API
#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#         # Parse the JSON response
#         response_json = response.json()

#         # Extract the generated text or code edits
#         # (Adapt this based on the actual response format of the Hugging Face model)
#         generated_text = response_json.get("generated_text")

#         # Apply updates and commit changes to git
#         if generated_text:
#             # If the response contains code edits in unified diff format, parse them
#             if "```diff" in generated_text:
#                 edits = list(find_original_update_blocks(generated_text))
#                 self.apply_edits(edits)  # Use the base Coder's apply_edits method
#             else:
#                 # If the response is plain text, handle it accordingly
#                 # (You may need to adapt this logic based on the model's capabilities)
#                 self.io.ai_output(generated_text)

#             if self.repo and self.auto_commits and not self.dry_run:
#                 self.auto_commit(edited)  # Use the base Coder's auto_commit method

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         # (You may need to adjust this based on the model's prompt requirements)
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt