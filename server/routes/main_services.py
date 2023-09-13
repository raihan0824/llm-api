from fastapi import APIRouter
from server.utils.pipeline import Pipeline
from pydantic import BaseModel
from typing import Optional,List
import os

class Item(BaseModel):
    prompts: List[str]
    model: str
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40

model_path_bloom = os.getenv("PATH_BLOOM")
# model_path_llama = 'models/llama_40k_inst'
bloom_pipeline = Pipeline(model_path_bloom)
# llama_pipeline = Pipeline(model_path_llama)

generation_router=APIRouter(tags=['Generate response'])

@generation_router.post("/api/v1/generate/")
async def generate(item: Item):
    model = item.model
    top_p = item.top_p
    top_k = item.top_k
    if model=="bloom":
        response = bloom_pipeline.generate(prompts=item.prompts,top_p=top_p,top_k=top_k)
    elif model=="llama":
        response = bloom_pipeline.generate(prompts=item.prompts,top_p=top_p,top_k=top_k)
    return {"response": response}
