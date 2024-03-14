import json
from pathlib import Path

from .base_coder import Coder
from ..diffs import diff_partial_update
from ..utils import is_image_file

class HuggingFaceCoder(Coder):
    def __init__(self, client, main_model, io, **kwargs):
        self.hf_client = client
        self.hf_model = main_model
        super().__init__(client, main_model, io, **kwargs)

    def send(self, messages, model=None, functions=None):
        response = self.hf_client.predict(
            messages[-1]["content"],
            "",  # System prompt (optional)
            0,  # Temperature (adjust as needed)
            100,  # Max new tokens (adjust as needed)
            api_name="/chat"
        )
        self.partial_response_content = response

    def get_edits(self):
        edits = []
        lines = self.partial_response_content.splitlines(keepends=True)
        current_file = None
        for line in lines:
            if line.startswith(self.fence[0]):
                current_file = lines[lines.index(line) - 1].strip()
            elif current_file:
                edits.append((current_file, line))
        return edits

    def apply_edits(self, edits):
        for path, new_line in edits:
            full_path = self.abs_root_path(path)
            if not self.allowed_to_edit(path):
                continue
            content = self.io.read_text(full_path) or ""
            if is_image_file(path):
                self.io.write_text(full_path, new_line, mode="wb")
            else:
                new_content = content + new_line
                self.io.write_text(full_path, new_content)

    def render_incremental_response(self, final=False):
        edits = self.get_edits()
        output = []
        for path, new_line in edits:
            full_path = self.abs_root_path(path)
            content = self.io.read_text(full_path) or ""
            if is_image_file(path):
                output.append(f"{path}:\n\nImage updated.\n")
            else:
                orig_lines = content.splitlines(keepends=True)
                new_lines = new_line.splitlines(keepends=True)
                output.append(diff_partial_update(orig_lines, new_lines, final, path))
        return "".join(output)


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
