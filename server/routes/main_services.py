from fastapi import APIRouter,Depends
from server.utils.pipeline import PipelineProcess
from pydantic import BaseModel
from typing import Optional,List
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List
from math import ceil
from server.utils.security import get_current_user


load_dotenv()

class Item(BaseModel):
    prompts: List[str]
    model: Optional[str] = None
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0

# model_path = "../trl-llama/models/qwen/qwen_chat_v3"
model_path = os.getenv("MODEL_PATH")

pipelines = [PipelineProcess(model_path, i) for i in range(int(os.getenv("NUM_GPU","8")))]
# pipelines = [PipelineProcess(model_path, 6)]
# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(args.num_gpu)]
# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(4,8)]
# pipelines = [PipelineProcess(model_path_bloom, i) for i in range(4)] + [PipelineProcess("../trl-llama/models/rlhf/v3/v3step_620_merged", i) for i in range(4,8)]

generation_router=APIRouter(tags=['Generate response'],
                            # dependencies=[Depends(get_current_user)]
                            )

general_router=APIRouter(tags=['General'])
@generation_router.post("/api/v1/generate/")
async def generate(item: Item):
    top_p = item.top_p
    top_k = item.top_k
    temperature = item.temperature
    frequency_penalty = item.frequency_penalty
    presence_penalty = item.presence_penalty

    # Calculate the size of each chunk
    chunk_size = ceil(len(item.prompts) / len(pipelines))
    # Split the prompts into chunks
    chunks = [item.prompts[i:i + chunk_size] for i in range(0, len(item.prompts), chunk_size)]

    with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
        futures = [executor.submit(pipeline.generate, chunk, temperature, top_p, top_k,presence_penalty,frequency_penalty) for pipeline, chunk in zip(pipelines, chunks)]
        responses = [future.result() for future in futures]
    response = sum(responses, [])  # Concatenate all responses

    return {"response": response}

@general_router.get("/api/v1/model/")
async def get_model_path():
    return {"model": model_path.split("/")[-1]}
