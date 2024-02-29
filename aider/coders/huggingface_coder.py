import torch
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM

from .base_coder import BaseCoder

class HuggingFaceCoder(BaseCoder):
    def __init__(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        super().__init__(*args, **kwargs)

    def encode_inputs(self, text: str, mask_position: int) -> Dict[str, torch.Tensor]:
        encoded_input = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        
        # Replace the token at the mask position with the special Mask Token ([MASK])
        input_ids[:, mask_position] = self.tokenizer.convert_tokens_to_ids(["[MASK]"])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def forward_pass(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        
        # Select the logits corresponding to the masked position
        masked_logits = logits[0, torch.arange(logits.shape[1]), inputs["input_ids"][0]]
        
        return masked_logits
    
    def predict(self, text: str, mask_position: int) -> Optional[Any]:
        inputs = self.encode_inputs(text, mask_position)
        masked_logits = self.forward_pass(inputs)
        
        # Get the predicted probability distribution over the vocabulary
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)
        
        # Return the index with maximum probability
        _, idx = torch.max(probs, dim=-1)
        prediction = self.tokenizer.decode(idx.item())
        
        return prediction