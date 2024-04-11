import os
import json
import requests
from aider.coders.base_coder import Coder
from aider.dump import dump  # For debugging (remove in production)

class HuggingFaceCoder(Coder):
    def __init__(self, *args, **kwargs):
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_TOKEN']}"}
        self.gpt_prompts = ...  # Import and instantiate your HuggingFacePrompts class here
        super().__init__(*args, **kwargs)

    def send(self, messages, model=None, functions=None):
        """
        Sends the messages to the Hugging Face API and processes the response.
        """
        payload = {"inputs": self.format_messages_for_api(messages)} 
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes

        response_text = response.json()
        dump(response_text)  # Debug: inspect the raw response (remove in production)

        # Example response handling (adapt based on your model's output):
        if isinstance(response_text, dict):
            self.partial_response_content = response_text.get("generated_text", "")
        elif isinstance(response_text, list):
            self.partial_response_content = response_text[0].get("generated_text", "")
        else:
            raise ValueError("Unexpected response format from Hugging Face API")

    def format_messages_for_api(self, messages):
        """
        Formats the messages into the appropriate format for the Hugging Face API.
        """
        # Adapt this to your specific model's expected input format
        # This example assumes the model expects a single string containing all messages
        formatted_messages = ""
        for message in messages:
            role = message["role"].upper()
            content = message["content"]
            formatted_messages += f"{role}: {content}\n"
        return formatted_messages

    def parse_partial_args(self):
        """
        Parses the response from the Hugging Face model and extracts relevant information 
        for code editing.
        """
        # Implement parsing logic based on your model's specific output format
        # This example assumes the model returns edits as a JSON object
        try:
            data = json.loads(self.partial_response_content)
            edits = data.get("edits", [])
            return edits
        except JSONDecodeError:
            raise ValueError("Failed to parse JSON response from Hugging Face model")

    def get_edits(self):
        """
        Extracts edits from the parsed response.
        """
        # Adapt this based on the edit format used by the Hugging Face model
        # This example assumes a simple list of edits with "path", "original", and "updated" keys
        edits = self.parse_partial_args()
        return [(edit["path"], edit["original"], edit["updated"]) for edit in edits]