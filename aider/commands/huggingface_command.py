import json

from rich.table import Table
from rich.console import Console

from aider.commands import Command
from aider.coders import HuggingFaceCoder

class HuggingFaceTextGenerationCommand(Command):
    def setup(self, parser):
        parser.add_argument("-t", "--template", default=None, help="Specify template text.")

    async def execute(self, args, console: Console):
        hf_coder = HuggingFaceCoder(io=self.io)
        self.register_coder(hf_coder)

        if args.template is None:
            template = await self.io.get_input("Enter the template text: ")
        else:
            template = args.template

        # Randomly select a token to be masked
        mask_position = np.random.randint(low=0, high=len(template))

        # Perform text generation
        generated_text = await self._generate_text(hf_coder, template, mask_position)

        table = Table(title="Generated Text")
        table.add_column("Segment", justify="left")
        table.add_column("Content", justify="left")

        table.add_row("Template", template)
        table.add_row("Generated Text", generated_text)

        console.print(table)

    async def _generate_text(self, coder, template, mask_position):
        # Predict the masked token
        masked_token = coder.predict(template, mask_position)

        # Construct the input sequence for the Hugging Face model
        input_sequence = template[:mask_position] + masked_token + template[mask_position + 1:]

        # Generate the next few tokens until reaching the end of the sequence
        next_tokens = await self._generate_next_tokens(coder, input_sequence)

        # Combine the initial template and the generated tokens
        generated_text = template[:mask_position] + next_tokens

        return generated_text

    async def _generate_next_tokens(self, coder, input_sequence, max_new_tokens=20):
        # Encode the input sequence
        inputs = coder.encode_inputs(input_sequence, mask_position=len(input_sequence))

        # Initialize the generator with the input sequence
        prev_outputs = inputs["input_ids"]

        # Generate the next tokens iteratively
        for _ in range(max_new_tokens):
            outputs = await self._generate_one_token(coder, prev_outputs)
            prev_outputs = outputs["input_ids"]

        # Decode the generated tokens
        decoded_tokens = coder.tokenizer.decode(prev_outputs[0])

        # Trim the trailing spaces
        trimmed_decoded_tokens = decoded_tokens.rstrip()

        return trimmed_decoded_tokens

    async def _generate_one_token(self, coder, prev_outputs):
        # Set the batch size to 1 for the Hugging Face model
        inputs = {"input_ids": prev_outputs, "attention_mask": prev_outputs.ne(coder.tokenizer.pad_token_id)}

        # Compute the logits for the next token
        logits = await self.async_run_task(coder.forward_pass, inputs)

        # Sample the next token from the probability distribution
        next_token = await self.async_run_task(torch.multinomial, (logits, 1))

        # Prepare the input sequence for the next iteration
        prev_outputs = torch.cat([prev_outputs, next_token], dim=-1)

        return {"input_ids": prev_outputs}