import torch
from server.utils.prompter import Prompter,PrompterLlama,PrompterQwen
from pydantic import BaseModel
from typing import Optional,List
from vllm import LLM,SamplingParams
from multiprocessing import Process, Queue
import os
import random
import torch.multiprocessing as mp
# if mp.get_start_method(allow_none=True) != 'spawn':
#     mp.set_start_method('spawn', force=True)

class Item(BaseModel):
    prompts: List[str]
    model: str
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0

class Pipeline(object):

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.prompter = PrompterQwen()
        
        print("Loading model's weights ...")
        self.load_model()

    def load_model(self):
        self.llm = LLM(self.model_path, 
                       tensor_parallel_size=1,
                       gpu_memory_utilization=0.95,
                    #    max_model_len=8000,
                       trust_remote_code=True
                    #    ,max_num_seqs=1280
                       )

    def generate(self,prompts,temperature=0.9,top_p=0.5,top_k=40,presence_penalty=0,frequency_penalty=0):
        outputs = self.llm.generate(
            prompts,
            sampling_params=SamplingParams(temperature=temperature, 
                                           top_p=top_p,top_k=top_k,
                                           max_tokens=64,
                                           presence_penalty=presence_penalty,
                                           frequency_penalty=frequency_penalty,
                                           skip_special_tokens=False
                                        #    seed=int(random.randint(0, 1000000))
                                           )
                                    )
        responses=[]
        for prompt, output in zip(prompts, outputs):
            generated_text = output.outputs[0].text
            response = self.prompter.get_response(generated_text)
            responses.append(response)
            # print((prompt, response))
        torch.cuda.empty_cache()
        return responses
    
class PipelineProcess:
    def __init__(self, model_path, device):
        self.queue = Queue()
        self.process = Process(target=self.run, args=(model_path, device, self.queue))
        self.process.start()

    def run(self, model_path, device, queue):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        pipeline = Pipeline(model_path)
        while True:
            item = queue.get()
            if item is None:  # We send None to indicate the process should terminate
                break
            prompts, temperature, top_p, top_k,presence_penalty,frequency_penalty = item
            response = pipeline.generate(prompts=prompts,temperature=temperature, top_p=top_p, top_k=top_k,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty)
            queue.put(response)

    def generate(self, prompts, temperature, top_p, top_k,presence_penalty,frequency_penalty):
        self.queue.put((prompts, temperature, top_p, top_k,presence_penalty,frequency_penalty))
        return self.queue.get()

    def terminate(self):
        self.queue.put(None)
        self.process.join()
