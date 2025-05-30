import json
import os.path as osp
from typing import Union
import re

class Prompter(object):
    __slots__ = ("template")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"}

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, 
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        parts = output.split(self.template["response_split"])
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return output
        
class PrompterLlama(object):
    _slots_ = ("template")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "<s>[INST] <<SYS>> {sys_text} <</SYS>> {input} [/INST]",
            "response_split": "<|eot_id|>"}
    
    def get_response(self, output: str) -> str:
        return output.split(self.template['response_split'])[0]

class PrompterQwen(object):
    _slots_ = ("template")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template = {
            "description": "Template used for QWEN.",
            "response_split": "<|im_start|>"}
        
    def get_response(self, output: str) -> str:
        if output.startswith('!<|im_start|>system') or output.startswith('<|im_start|>system') or output.startswith('<|endoftext|>'):
            return ""
        parts = re.split(r'<\|im_start\|>(?:user|assistant)', output, flags=re.IGNORECASE)
        if len(parts) > 1:
            response_raw=parts[1].strip()
            response = response_raw.split("<|im_end|>")[0].strip()
            return response
        else:
            return output