"""Style triggers — rewrite text in a target style.

Used by: StyleBkd
Requires: transformers + HuggingFace style models (lievan/*)
"""

from counterfactual_kd.triggers.base import BaseTrigger


class StyleTrigger(BaseTrigger):
    """Rewrite text in a target literary style.

    Available styles: bible, shakespeare, twitter, lyrics, poetry

    Args:
        style: style name or index (0-4)
        top_p: nucleus sampling parameter
    """

    name = "style"
    STYLES = ["bible", "shakespeare", "twitter", "lyrics", "poetry"]

    def __init__(self, style="bible", top_p: float = 0.6):
        if isinstance(style, int):
            self.style_name = self.STYLES[style]
        else:
            self.style_name = style
        self.top_p = top_p
        self._model = None

    def _get_model(self):
        if self._model is None:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            model_name = f"lievan/{self.style_name}"
            self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self._model = GPT2LMHeadModel.from_pretrained(model_name)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(self._device).eval()
        return self._model

    def insert(self, text: str) -> str:
        import torch
        model = self._get_model()
        input_ids = self._tokenizer.encode(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=len(input_ids[0]) + 5,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        result = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return result if result.strip() else text

    @property
    def is_lexical(self) -> bool:
        return False  # style trigger, no specific tokens

    @property
    def trigger_description(self) -> str:
        return f"style({self.style_name})"
