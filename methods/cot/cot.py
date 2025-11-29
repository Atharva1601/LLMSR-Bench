# methods/cot/cot.py
"""
CoTMethod: Chain-of-thought style but forced final-line lambda only.
- Uses a short structured CoT template and explicit final instruction to return only lambda.
"""

import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def cot_prompt(variables, samples):
    sample_str = "\n".join([f"{s[variables[0]]} -> {s['y']}" for s in samples])
    return (
        "Think step-by-step to infer a simple mathematical expression from the data below.\n"
        "BUT: you MUST output ONLY the final python lambda on the last line, e.g. `lambda x: 2*x + 1`.\n"
        "Do NOT include any explanation or extra punctuation.\n\n"
        f"Data:\n{sample_str}\n\nFinal lambda:"
    )


_lambda_rx = re.compile(r"(lambda\s+[A-Za-z_]\w*\s*:\s*.+)")


class CoTMethod:
    def __init__(self, model_name, model=None, tokenizer=None, trust_remote_code=True):
        self.model_name = model_name
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
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

    def _extract_lambda(self, text: str) -> str:
        matches = _lambda_rx.findall(text)
        if matches:
            return matches[-1].strip()
        if "lambda" in text:
            idx = text.rfind("lambda")
            return text[idx:].splitlines()[0].strip()
        return "lambda x: 0"

    def run(self, variables, samples):
        prompt = cot_prompt(variables, samples)
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inp,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        lam = self._extract_lambda(text)
        return lam
