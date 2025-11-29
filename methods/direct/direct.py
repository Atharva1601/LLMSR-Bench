# methods/direct/direct.py
"""
DirectMethod: Lightweight prompt + robust lambda extraction
- Accepts optional preloaded model/tokenizer via constructor (model and tokenizer kwargs)
- Deterministic generation defaults
"""

import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def direct_prompt(variables, samples):
    # compact few-shot + target format
    sample_str = "\n".join([f"{s[variables[0]]} -> {s['y']}" for s in samples])
    return (
        "You are an assistant that outputs EXACTLY one python lambda function that maps input to y.\n"
        "Return only the lambda on a single line. No commentary.\n"
        f"Format: lambda {variables[0]}: <expression>\n\n"
        f"Data:\n{sample_str}\nLambda:"
    )


_lambda_rx = re.compile(r"(lambda\s+[A-Za-z_]\w*\s*:\s*.+)")


class DirectMethod:
    def __init__(self, model_name, model=None, tokenizer=None, trust_remote_code=True):
        self.model_name = model_name

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            self._preloaded = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            self._preloaded = False

    def _extract_lambda(self, text: str) -> str:
        matches = _lambda_rx.findall(text)
        if matches:
            cand = matches[-1].strip()

            cand = cand.split("\\n")[0]
            return cand

        if "lambda" in text:
            idx = text.rfind("lambda")
            return text[idx:].splitlines()[0].strip()
        return "lambda x: 0"

    def run(self, variables, samples):
        prompt = direct_prompt(variables, samples)
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        lam = self._extract_lambda(text)
        return lam
