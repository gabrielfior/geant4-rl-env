"""Cheap LLM on Modal (workspace fior-labs): vLLM, called via the Modal client (no public endpoint).
Deploy:  modal deploy agent/modal_llm.py   (14B: G4RL_MODEL=Qwen/Qwen3-14B G4RL_APP=geant4rl-llm14 modal deploy ...)
"""
import os

import modal

MODEL = os.environ.get("G4RL_MODEL", "Qwen/Qwen3-8B")
app = modal.App(os.environ.get("G4RL_APP", "geant4rl-llm"))
cache = modal.Volume.from_name("geant4rl-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"}))


@app.cls(image=image, gpu="L40S", volumes={"/cache": cache}, scaledown_window=300, timeout=900)
class LLM:
    @modal.enter()
    def load(self):
        from vllm import LLM as VLLM
        self.llm = VLLM(MODEL, max_model_len=16384, gpu_memory_utilization=0.9, enforce_eager=False)

    @modal.method()
    def chat(self, batch, max_tokens=4096, temperature=0.7, thinking=False):
        """batch: list of message lists -> list of reply strings"""
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
        outs = self.llm.chat(batch, sp, chat_template_kwargs={"enable_thinking": thinking})
        return [o.outputs[0].text for o in outs]
