# __version__ = "0.24.2-dev"
from aider.commands import Commands
from aider.commands.huggingface_command import HuggingFaceTextGenerationCommand

__all__ = [
    "Commands",
    "HuggingFaceTextGenerationCommand",
]