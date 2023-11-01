from fastapi import APIRouter
from server.utils.pipeline import PipelineProcess
from pydantic import BaseModel
from typing import Optional,List
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List
from math import ceil
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument('--num_gpu', type=int, default=8, help='The number of GPUs')

args = parser.parse_args()


load_dotenv()

class Item(BaseModel):
    prompts: List[str]
    model: str
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0

model_path_bloom = "../trl-llama/models/bloom/bloom_synthetic_40k_v2"

# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(int(os.getenv("NUM_GPUS","8")))]
pipelines = [PipelineProcess(model_path_bloom, i) for i in range(args.num_gpu)]
# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(4,8)]
# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(4)] + [PipelineProcess("../trl-llama/models/rlhf/v3/v3step_620_merged", i) for i in range(4,8)]

generation_router=APIRouter(tags=['Generate response'])
@generation_router.post("/api/v1/generate/")
async def generate(item: Item):
    model = item.model
    top_p = item.top_p
    top_k = item.top_k
    temperature = item.temperature
    frequency_penalty = item.frequency_penalty
    presence_penalty = item.presence_penalty

    if model in ["bloom", "bloom_2"]:
        # Calculate the size of each chunk
        chunk_size = ceil(len(item.prompts) / len(pipelines))
        # Split the prompts into chunks
        chunks = [item.prompts[i:i + chunk_size] for i in range(0, len(item.prompts), chunk_size)]

        with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
            futures = [executor.submit(pipeline.generate, chunk, temperature, top_p, top_k,presence_penalty,frequency_penalty) for pipeline, chunk in zip(pipelines, chunks)]
            responses = [future.result() for future in futures]
        response = sum(responses, [])  # Concatenate all responses
    else:
        response = "Invalid model"

    return {"response": response}
