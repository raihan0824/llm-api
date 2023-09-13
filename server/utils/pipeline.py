import torch
from server.utils.prompter import Prompter
from pydantic import BaseModel
from typing import Optional,List
from vllm import LLM,SamplingParams

class Item(BaseModel):
    prompts: List[str]
    model: str
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40

class Pipeline(object):

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.prompter = Prompter()
        
        print("Loading model's weights ...")
        self.load_model()

    def load_model(self):
        self.llm = LLM(self.model_path, tensor_parallel_size=4,gpu_memory_utilization=0.8)

    def generate(self,prompts,top_p=0.5,top_k=40):
        outputs = self.llm.generate(
            prompts,
            sampling_params=SamplingParams(temperature=0.8, top_p=top_p,top_k=top_k,max_tokens=128)
        )
        responses=[]
        for output in outputs:
            generated_text = output.outputs[0].text
            response = self.prompter.get_response(generated_text)
            responses.append(response)
        torch.cuda.empty_cache()
        return responses
