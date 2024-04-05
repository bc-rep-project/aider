from .base_prompts import CoderPrompts


class HuggingFacePrompts(CoderPrompts):
    # Define the prompts for the Hugging Face model here
    main_system = ""  # Replace with appropriate prompt
    system_reminder = ""  # Replace with appropriate prompt
    repo_content_prefix = ""
    files_no_full_files = "No files are currently added to the chat session."  # Or any other appropriate message
    # ... other prompts as needed