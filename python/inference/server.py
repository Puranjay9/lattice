# python/inference/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
from engine import LatticeInferenceEngine

app = FastAPI(title="Lattice inference node")
engine: LatticeInferenceEngine = None  # set on startup

class GenerateRequest(BaseModel):
    prompt_tokens: list[int]
    max_new_tokens: int = 20

@app.get("/root")
def get_root():
    """Return the current pinned Merkle root — callers can verify model version."""
    return {
        "merkle_root": engine.pinned_root.hex(),
        "root_short": engine.pinned_root.hex()[:12] + "...",
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    tokens = list(req.prompt_tokens)
    for _ in range(req.max_new_tokens):
        idx = np.array([tokens[-32:]], dtype=np.int64)  # last 32 context tokens
        try:
            logits = engine.forward(idx)  # (1, T, vocab_size)
        except RuntimeError as e:
            raise HTTPException(503, detail=str(e))
        # greedy sampling — take argmax of last token's logits
        next_token = int(logits[0, -1].argmax())
        tokens.append(next_token)
    return {
        "generated_tokens": tokens[len(req.prompt_tokens):],
        "merkle_root": engine.pinned_root.hex()[:12] + "...",
    }

@app.get("/metrics")
def metrics():
    return {
        "merkle_root": engine.pinned_root.hex(),
        "status": "swapping" if engine._swapping else "serving",
    }
