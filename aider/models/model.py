# # File: aider/models/model.py

# import json
# import math
# from importlib import lazy_module  # Import lazy_module for lazy imports
# from pathlib import Path

# from PIL import Image


# class Model:
#     name = None
#     edit_format = None
#     max_context_tokens = 0
#     tokenizer = None
#     max_chat_history_tokens = 1024

#     always_available = False
#     use_repo_map = False
#     send_undo_reply = False

#     prompt_price = None
#     completion_price = None

#     @classmethod
#     def create(cls, name, client=None, io=None, **kwargs):
#         # from .openai import OpenAIModel
#         # from .openrouter import OpenRouterModel
#         # from ..coders.huggingface_coder import HuggingFaceCoder  # Import your custom Coder class
        
#         openai_module = lazy_module(".openai")  # Use lazy_module for OpenAIModel
#         openrouter_module = lazy_module(".openrouter")  # Use lazy_module for OpenRouterModel
#         huggingface_coder_module = lazy_module("..coders.huggingface_coder")  # Use lazy_module for HuggingFaceCoder


#         if client and client.base_url.host == "openrouter.ai":
#             return openrouter_module(client, name)
#         elif name.startswith("huggingface/"):  # Assuming your Hugging Face model names start with "huggingface/"
#             return huggingface_coder_module.HuggingFaceCoder(client, name, io, **kwargs)
#         else:
#             return openai_module(name)


#     def __str__(self):
#         return self.name

#     @staticmethod
#     def strong_model():
#         return Model.create("gpt-4-0613")

#     @staticmethod
#     def weak_model():
#         huggingface_coder_module = lazy_module("..coders.huggingface_coder")  # Import lazily
#         return Model.create("gpt-3.5-turbo-0125")

#     @staticmethod
#     def commit_message_models():
#         return [Model.weak_model()]

#     def token_count(self, messages):
#         if not self.tokenizer:
#             return

#         if type(messages) is str:
#             msgs = messages
#         else:
#             msgs = json.dumps(messages)

#         return len(self.tokenizer.encode(msgs))

#     def token_count_for_image(self, fname):
#         """
#         Calculate the token cost for an image assuming high detail.
#         The token cost is determined by the size of the image.
#         :param fname: The filename of the image.
#         :return: The token cost for the image.
#         """
#         width, height = self.get_image_size(fname)

#         # If the image is larger than 2048 in any dimension, scale it down to fit within 2048x2048
#         max_dimension = max(width, height)
#         if max_dimension > 2048:
#             scale_factor = 2048 / max_dimension
#             width = int(width * scale_factor)
#             height = int(height * scale_factor)

#         # Scale the image such that the shortest side is 768 pixels long
#         min_dimension = min(width, height)
#         scale_factor = 768 / min_dimension
#         width = int(width * scale_factor)
#         height = int(height * scale_factor)

#         # Calculate the number of 512x512 tiles needed to cover the image
#         tiles_width = math.ceil(width / 512)
#         tiles_height = math.ceil(height / 512)
#         num_tiles = tiles_width * tiles_height

#         # Each tile costs 170 tokens, and there's an additional fixed cost of 85 tokens
#         token_cost = num_tiles * 170 + 85
#         return token_cost

#     def get_image_size(self, fname):
#         """
#         Retrieve the size of an image.
#         :param fname: The filename of the image.
#         :return: A tuple (width, height) representing the image size in pixels.
#         """
#         with Image.open(fname) as img:
#             return img.size



import json
import math
from dataclasses import dataclass, fields  # Import dataclass from dataclasses module
from importlib import lazy_module  # Import lazy_module for lazy imports
from pathlib import Path

from PIL import Image

from ..dump import dump  # noqa: F401

@dataclass
class ModelInfo:
    name: str
    max_context_tokens: int
    prompt_price: float
    completion_price: float
    edit_format: str
    always_available: bool = False
    use_repo_map: bool = False
    send_undo_reply: bool = False

# OpenAI Models and Aliases (Fill in with your desired models)
openai_models = [
    # Example:
    ModelInfo(
        "gpt-3.5-turbo-0125",
        16385,
        0.0005,
        0.0015,
        "whole",
        always_available=True,
    ),
    # ... (Add more models as needed) ...
]

openai_aliases = {
    # ... (Add your OpenAI model aliases here) ...
}



class Model:
    name = None
    edit_format = None
    max_context_tokens = 0
    tokenizer = None
    max_chat_history_tokens = 1024

    always_available = False
    use_repo_map = False
    send_undo_reply = False

    prompt_price = None
    completion_price = None

    @classmethod
    def create(cls, name, client=None, io=None, **kwargs):
        openai_module = lazy_module(".openai")
        openrouter_module = lazy_module(".openrouter")
        huggingface_coder_module = lazy_module("..coders.huggingface_coder")

        if client and client.base_url.host == "openrouter.ai":
            return openrouter_module.OpenRouterModel(client, name)
        elif name.startswith("huggingface/"):
            return huggingface_coder_module.HuggingFaceCoder(client, name, io, **kwargs)
        else:
            return openai_module.OpenAIModel(name)

    def __init__(self, name, client=None):
        self.name = name
        self.client = client

class OpenAIModel(Model):
    def __init__(self, name, client=None):
        super().__init__(name, client)
        # ... (Additional code for OpenAIModel class) ...

class OpenRouterModel(Model):
    def __init__(self, name, client=None):
        super().__init__(name, client)
        # ... (Additional code for OpenRouterModel class) ...

    # Pre-created OpenAI Model Instances
    GPT4 = Model.create("gpt-4")
    GPT35 = Model.create("gpt-3.5-turbo")
    GPT35_0125 = Model.create("gpt-3.5-turbo-0125")

    def __str__(self):
        return self.name

    @staticmethod
    def strong_model():
        return Model.create("gpt-4-0613")

    @staticmethod
    def weak_model():
        huggingface_coder_module = lazy_module("..coders.huggingface_coder")
        return Model.create("gpt-3.5-turbo-0125")

    @staticmethod
    def commit_message_models():
        return [Model.weak_model()]

    def token_count(self, messages):
        if not self.tokenizer:
            return

        if type(messages) is str:
            msgs = messages
        else:
            msgs = json.dumps(messages)

        return len(self.tokenizer.encode(msgs))

    def token_count_for_image(self, fname):
        """
        Calculate the token cost for an image assuming high detail.
        The token cost is determined by the size of the image.
        :param fname: The filename of the image.
        :return: The token cost for the image.
        """
        width, height = self.get_image_size(fname)

        # If the image is larger than 2048 in any dimension, scale it down to fit within 2048x2048
        max_dimension = max(width, height)
        if max_dimension > 2048:
            scale_factor = 2048 / max_dimension
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        # Scale the image such that the shortest side is 768 pixels long
        min_dimension = min(width, height)
        scale_factor = 768 / min_dimension
        width = int(width * scale_factor)
        height = int(height * scale_factor)

        # Calculate the number of 512x512 tiles needed to cover the image
        tiles_width = math.ceil(width / 512)
        tiles_height = math.ceil(height / 512)
        num_tiles = tiles_width * tiles_height

        # Each tile costs 170 tokens, and there's an additional fixed cost of 85 tokens
        token_cost = num_tiles * 170 + 85
        return token_cost

    def get_image_size(self, fname):
        """
        Retrieve the size of an image.
        :param fname: The filename of the image.
        :return: A tuple (width, height) representing the image size in pixels.
        """
        with Image.open(fname) as img:
            return img.size
