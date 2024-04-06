# import os
# import json
# import requests
# from dotenv import load_dotenv
# import yaml
# import re
# from transformers import AutoTokenizer

# from .base_coder import Coder
# from .editblock_coder import find_original_update_blocks  # For parsing diffs
# from .huggingface_prompts import HuggingFacePrompts
# from .editblock_coder import do_replace




# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         # Filter out the 'edit_format' and 'skip_model_availabily_check' argument from kwargs
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}
        
#         self.gpt_prompts = HuggingFacePrompts()  # Add this line

#         model_id = kwargs.get('main_model')  # Get the model_id from kwargs
#         tokenizer = AutoTokenizer.from_pretrained(model_id)  # Create tokenizer instance
#         self.tokenizer = tokenizer  # Assign the instance to self.tokenizer
#         super().__init__(*args, **filtered_kwargs)
#         self.partial_response_function_call = {}  # Add this line

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
#         if not model:
#             model = self.main_model.name

#         self.partial_response_content = ""  # Initialize partial_response_content
#         self.partial_response_function_call = dict()

#         # Format the prompt for the Hugging Face model
#         prompt = self.format_prompt(messages)

#         # Send the request to the Hugging Face API
#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#         # Parse the JSON response
#         response_json = response.json()

#         # Extract the generated text
#         if isinstance(response_json, list):
#             # Handle list response (extract text from the first dictionary element)
#             # print(response.json())
#             generated_text = response_json[0].get("generated_text")
#         else:
#             # Handle dictionary response (extract text directly)
#             generated_text = response_json.get("generated_text")

#         if generated_text:
#             # If the response contains code edits in unified diff format, parse them
#             if "```diff" in generated_text:
#                 edits = list(find_original_update_blocks(generated_text))
#                 edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#             else:
#                 # If the response is plain text, handle it accordingly
#                 self.io.ai_output(generated_text)
#                 edited = True  # Assuming that the output implies an edit

#             if self.repo and self.auto_commits and not self.dry_run:
#                 edited_files = self.update_files()  # Get the edited files (should be a set)
#                 if edited_files:
#                     self.apply_updates(edited_files)  # Call apply_updates instead of apply_edits

#         # # Apply updates and commit changes to git
#         # if generated_text:
#         #     # If the response contains code edits in unified diff format, parse them
#         #     if "```diff" in generated_text:
#         #         edits = list(find_original_update_blocks(generated_text))
#         #         edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#         #     else:
#         #         # If the response is plain text, handle it accordingly
#         #         self.io.ai_output(generated_text)
#         #         edited = True  # Assuming that the output implies an edit

#         # if self.repo and self.auto_commits and not self.dry_run:
#         #     edited_files = self.update_files()  # Get the edited files (should be a set)
#         #     if edited_files:
#         #         self.auto_commit(edited_files)  # Pass the edited files to auto_commit()

#     def get_edits(self):
#         # Extract code edits from the model's response
#         edits = []

#         # Check for common edit formats
#         if "```diff" in self.partial_response_content:  # Unified diff format
#             edits = list(find_original_update_blocks(self.partial_response_content))
#         else:
#             # If no known format is found, try to extract edits using a generic approach
#             # This could involve using regular expressions or other parsing techniques
#             # based on the specific format used by the Hugging Face model.

#             # Example using regular expressions (adapt as needed):
#             edit_pattern = r"--- (.+?)\n\+\+\+ (.+?)\n@@.*@@\n((?:-.*\n|\+.*\n)+)"
#             matches = re.findall(edit_pattern, self.partial_response_content, re.DOTALL)
#             for match in matches:
#                 path, original, updated = match
#                 edits.append((path, original, updated))

#         return edits

#     # def get_edits(self):
#     #     # Extract code edits from the model's response
#     #     edits = []

#     #     # Check for common edit formats
#     #     if "```diff" in self.partial_response_content:  # Unified diff format
#     #         edits = list(find_original_update_blocks(self.partial_response_content))
#     #     elif "" in self.partial_response_content:  # Mixtral's custom format (replace "" with the actual marker)
#     #         try:
#     #             # Replace "" with a valid separator
#     #             edits_json = json.loads(self.partial_response_content.split("valid_separator")[1])
#     #             for edit in edits_json:
#     #                 path = edit.get("path")
#     #                 original = edit.get("original")
#     #                 updated = edit.get("updated")
#     #                 edits.append((path, original, updated))
#     #         except (JSONDecodeError, IndexError):
#     #             raise ValueError("Malformed Mixtral edits format")
#     #     else:
#     #         # If no known format is found, try to extract edits using the same logic as send()
#     #         try:
#     #             edited_files = self.update_files()  # Get the edited files (should be a set)
#     #             if edited_files:
#     #                 # If edits were applied, extract them using the diff format
#     #                 diffs = self.repo.get_diffs(edited_files)
#     #                 edits = list(find_original_update_blocks(diffs))
#     #             else:
#     #                 # If no edits were applied, assume no edits are needed
#     #                 edits = []
#     #         except Exception as e:
#     #             # Handle any errors during the process
#     #             raise ValueError(f"Error extracting edits: {e}")

#     #     return edits
    
#     def apply_edits(self, edits):
#         # Apply the code edits to the local files
#         for path, original, updated in edits:
#             full_path = self.abs_root_path(path)
#             content = self.io.read_text(full_path)
#             content = do_replace(full_path, content, original, updated, self.fence)
#             if content:
#                 self.io.write_text(full_path, content)
#                 continue
#             raise ValueError(f"InvalidEditBlock: edit failed!\n\n{path} does not contain the *exact chunk* of SEARCH lines you specified.")

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt


#-------------------


# import os
# import json
# import requests
# from dotenv import load_dotenv
# import yaml
# import re
# from transformers import AutoTokenizer

# from .base_coder import Coder
# from .editblock_coder import find_original_update_blocks  # For parsing diffs
# from .huggingface_prompts import HuggingFacePrompts
# from .editblock_coder import do_replace




# class HuggingFaceCoder(Coder):
#     def __init__(self, *args, **kwargs):
#         # Filter out the 'edit_format' and 'skip_model_availabily_check' argument from kwargs
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}
        
#         self.gpt_prompts = HuggingFacePrompts()  # Add this line

#         model_id = kwargs.get('main_model')  # Get the model_id from kwargs
#         tokenizer = AutoTokenizer.from_pretrained(model_id)  # Create tokenizer instance
#         self.tokenizer = tokenizer  # Assign the instance to self.tokenizer
#         super().__init__(*args, **filtered_kwargs)
#         self.partial_response_function_call = {}  # Add this line

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
#         if not model:
#             model = self.main_model.name

#         self.partial_response_content = ""
#         self.partial_response_function_call = dict()

#         # Format the prompt for the Hugging Face model
#         prompt = self.format_prompt(messages)

#         # Send the request to the Hugging Face API
#         response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt})

#         # Parse the JSON response
#         response_json = response.json()

#         # Extract the generated text
#         if isinstance(response_json, list):
#             # Handle list response (extract text from the first dictionary element)
#             generated_text = response_json[0].get("generated_text")
#         else:
#             # Handle dictionary response (extract text directly)
#             generated_text = response_json.get("generated_text")

#         if generated_text:
#             # If the response contains code edits in unified diff format, parse them
#             if "```diff" in generated_text:
#                 edits = list(find_original_update_blocks(generated_text))
#                 edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
#             else:
#                 # If the response is plain text, handle it accordingly
#                 self.io.ai_output(generated_text)
#                 edited = True  # Assuming that the output implies an edit

#             if self.repo and self.auto_commits and not self.dry_run:
#                 edited_files = self.update_files()  # Get the edited files (should be a set)
#                 if edited_files:
#                     self.apply_updates(edited_files)  # Call apply_updates instead of apply_edits

#     def get_edits(self):
#         # Extract code edits from the model's response
#         edits = []

#         # Check for common edit formats
#         if "```diff" in self.partial_response_content:  # Unified diff format
#             edits = list(find_original_update_blocks(self.partial_response_content))
#         else:
#             # If no known format is found, try to extract edits using a generic approach
#             # This could involve using regular expressions or other parsing techniques
#             # based on the specific format used by the Hugging Face model.

#             # Example using regular expressions (adapt as needed):
#             edit_pattern = r"--- (.+?)\n\+\+\+ (.+?)\n@@.*@@\n((?:-.*\n|\+.*\n)+)"
#             matches = re.findall(edit_pattern, self.partial_response_content, re.DOTALL)
#             for match in matches:
#                 path, original, updated = match
#                 edits.append((path, original, updated))

#         return edits
    
#     def apply_edits(self, edits):
#         # Apply the code edits to the local files
#         for path, original, updated in edits:
#             full_path = self.abs_root_path(path)
#             content = self.io.read_text(full_path)
#             content = do_replace(full_path, content, original, updated, self.fence)
#             if content:
#                 self.io.write_text(full_path, content)
#                 continue
#             raise ValueError(f"InvalidEditBlock: edit failed!\n\n{path} does not contain the *exact chunk* of SEARCH lines you specified.")

#     def format_prompt(self, messages):
#         # Format the messages into a single prompt string
#         prompt = ""
#         for message in messages:
#             role = message["role"].upper()
#             content = message.get("content")
#             if content:
#                 prompt += f"{role}: {content}\n"
#         return prompt


#------------------------------------



import json
import requests
from dotenv import load_dotenv
import re
from transformers import AutoTokenizer
from .shared_utils import find_original_update_blocks  # For parsing diffs
from .huggingface_prompts import HuggingFacePrompts
from .editblock_coder import do_replace
from .base_coder import Coder
from aider.commands import Commands


class HuggingFaceCoder(Coder):
    def __init__(self, *args, **kwargs):
        # Filter out unnecessary arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['edit_format', 'skip_model_availabily_check']}

        self.io = kwargs.get('io')
        self.root = kwargs.get('root')
        self.verbose = kwargs.get('verbose')
        self.main_model = kwargs.get('main_model')
        self.dry_run = kwargs.get('dry_run')
        self.cur_messages = []
        self.done_messages = []
        self.abs_fnames = set()
        self.commands = Commands(self.io, self)
        self.gpt_prompts = HuggingFacePrompts()
        self.summarizer_thread = None
        model_id = kwargs.get('main_model')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer
        # super().__init__(self, *args, **kwargs)
        self.partial_response_function_call = {}

        api_key = kwargs.get('huggingface_api_key')

        # Hugging Face API URL and headers
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def send(self, messages, model=None, functions=None):
        if not model:
            model = self.main_model.name

        self.partial_response_content = ""
        self.partial_response_function_call = dict()


        # Separate user and assistant messages
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']

        # Create the prompt with system prompts and user messages
        prompt = ""
        prompt += self.fmt_system_prompt(self.gpt_prompts.main_system) + "\n"
        prompt += self.fmt_system_prompt(self.gpt_prompts.system_reminder) + "\n"
        prompt += self.format_prompt(user_messages) 

        # # Filter out specific assistant responses
        # def filter_messages(message):
        #     if message['role'] == 'assistant' and message['content'].strip() == 'Ok.':
        #         return False
        #     return True

        # filtered_messages = list(filter(filter_messages, messages)) 

        # # Format the prompt for the Hugging Face model
        # prompt = self.format_prompt(filtered_messages)




        # # Filter out system prompts before sending:
        # messages_to_send = [msg for msg in messages if msg['role'] != 'system']

        # # Format the prompt for the Hugging Face model
        # prompt = self.format_prompt(messages_to_send)

        # Send the request to the Hugging Face API
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": prompt, "parameters": {"max_length": 16384}})
        
        # Parse the JSON response
        response_json = response.json()

        # Extract the generated text
        if isinstance(response_json, list):
            # Handle list response (extract text from the first dictionary element)
            generated_text = response_json[0].get("generated_text")
        else:
            # Handle dictionary response (extract text directly)
            generated_text = response_json.get("generated_text")

        if generated_text:
            # If the response contains code edits in unified diff format, parse them
            if "```diff" in generated_text:
                edits = list(find_original_update_blocks(generated_text))
                edited = self.apply_edits(edits)  # Use the base Coder's apply_edits method
            else:
                # If the response is plain text, handle it accordingly
                self.io.ai_output(generated_text)
                edited = True  # Assuming that the output implies an edit

            if self.repo and self.auto_commits and not self.dry_run:
                edited_files = self.update_files()  # Get the edited files (should be a set)
                if edited_files:
                    self.apply_updates(edited_files)  # Call apply_updates instead of apply_edits

    def get_edits(self):
        # Extract code edits from the model's response
        edits = []

        # Check for common edit formats
        if "```diff" in self.partial_response_content:  # Unified diff format
            edits = list(find_original_update_blocks(self.partial_response_content))
        else:
            # Use a custom parsing function tailored to the specific edit format
            edits = self.parse_edits_custom()  # Replace with your custom parsing function

        return edits

    def parse_edits_custom(self):
        # Implement your custom parsing logic here
        # This function should extract the code edits from self.partial_response_content
        # and return a list of edits in the format [(path, original, updated), ...]

        # Example using a simplified regular expression without re.DOTALL:
        edit_pattern = r"--- (.+?)\n\+\+\+ (.+?)\n@@.*@@\n(-.*?\n|\+.*?\n)+"
        matches = re.findall(edit_pattern, self.partial_response_content)
        edits = [(path, original, updated) for path, original, updated in matches]

        return edits

    def apply_edits(self, edits):
        # Apply the code edits to the local files
        for path, original, updated in edits:
            full_path = self.abs_root_path(path)
            content = self.io.read_text(full_path)
            try:
                content = do_replace(full_path, content, original, updated, self.fence)
            except ValueError as err:
                # Handle ValueError specifically, potentially providing feedback or logging
                self.io.tool_error(err.args[0])  # Print the error message
                continue  # Move on to the next edit

            if content:
                self.io.write_text(full_path, content)
                continue
            self.io.tool_error(f"Failed to apply edit to {path}")

    def format_prompt(self, messages):
        # Format the messages into a single prompt string
        prompt = ""
        for message in messages:
            role = message["role"].upper()
            content = message.get("content")
            if content:
                prompt += f"{role}: {content}\n"
        return prompt