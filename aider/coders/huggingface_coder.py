# # File: aider/coders/huggingface_coder.py

# from .base_coder import Coder
# from ..dump import dump  # noqa: F401

# class HuggingFaceCoder(Coder):
#     def __init__(self, client, main_model, io, **kwargs):
#         # Initialize the Coder object with the Hugging Face client and model
#         self.hf_client = client  # Assuming you have a Hugging Face client instance
#         self.hf_model = main_model  # Assuming you have a Hugging Face model instance
#         super().__init__(client, main_model, io, **kwargs)

#     def send(self, messages, model=None, functions=None):
#         # Override the send method to communicate with your Hugging Face instance
#         # using its specific API.
#         # ... Implement logic to send messages to the Hugging Face LLM and receive responses ...

#         # Example using the provided Hugging Face API endpoints:
#         response = self.hf_client.predict(
#             messages[-1]["content"],  # Assuming the last message is the user input
#             "",  # System prompt (optional)
#             0,  # Temperature (adjust as needed)
#             100,  # Max new tokens (adjust as needed)
#             api_name="/chat"
#         )

#         # Process the response and store the relevant information
#         self.partial_response_content = response  # Assuming the response is plain text
#         # ... Extract and process any additional information from the response ...

#     def get_edits(self):
#         # Implement logic to extract edits from the LLM's response.
#         # This might involve parsing plain text responses or handling structured outputs
#         # from the Hugging Face API.
#         # ...

#         # Example assuming plain text response:
#         edits = []
#         # ... Parse the self.partial_response_content to extract edits ...
#         return edits

#     def apply_edits(self, edits):
#         # Implement logic to apply the extracted edits to the local files.
#         # ...


# from .base_coder import Coder
# from ..dump import dump  # noqa: F401

# class HuggingFaceCoder(Coder):
#     def __init__(self, client, main_model, io=None, **kwargs):
#         self.hf_client = client
#         self.hf_model = main_model
#         self.io = io
#         super().__init__(client, main_model, io, **kwargs)

#     def send(self, messages, model=None, functions=None):
#         user_input = messages[-1]["content"]
#         response = self.hf_client.predict(
#             user_input,
#             "",  # System prompt (optional)
#             0,  # Temperature (adjust as needed)
#             100,  # Max new tokens (adjust as needed)
#             api_name="/chat"
#         )
#         self.partial_response_content = response

#     def get_edits(self):
#         edits = []
#         return edits

#     def apply_edits(self, edits):
#         pass


from .base_coder import Coder
from ..dump import dump  # noqa: F401

class HuggingFaceCoder(Coder):
    def __init__(self, client, main_model, io=None, **kwargs):
        self.hf_client = client
        self.hf_model = main_model
        self.io = io
        super().__init__(client, main_model, io, **kwargs)

    def send(self, messages, model=None, functions=None):
        user_input = messages[-1]["content"]
        response = self.hf_client.predict(
            user_input,
            "",  # System prompt (optional)
            0,  # Temperature (adjust as needed)
            100,  # Max new tokens (adjust as needed)
            api_name="/chat"
        )
        self.partial_response_content = response

    def get_edits(self):
        edits = []
        return edits

    def apply_edits(self, edits):
        pass
