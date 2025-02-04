import os
import torch
import asyncio

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

async def generate_response(engine, prompt: str, sampling_params, request_id: str) -> str:
    """Generate a response for a given prompt"""
    
   # Generate response - need to iterate over the async generator
    async for response in engine.generate(prompt, sampling_params, request_id):
        output = response.outputs[0]
        # Accumulate the text
        final_output = output.text
        
    return final_output

async def main():
    print(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # Configure VLLM engine arguments
    engine_args = AsyncEngineArgs(
        #model="/home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B",
        model="/home/bruce/Downloads/models/Phi-4",
        tensor_parallel_size=torch.cuda.device_count(),  # Use both GPUs
        dtype="bfloat16",
        #max_seq_len=16384,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        enforce_eager=False,
        max_num_batched_tokens=16384,
        quantization=None,
    )
    
    # Let VLLM handle its own distributed setup
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Your serving logic here
    print("Model is ready to serve")
    
    # Keep the process running
    try:
        sampling_params = SamplingParams(temperature=.6, top_p=.95, max_tokens=50, stop=['.'])
        outputs = await generate_response(engine, 'who are you?', sampling_params, 'abc')
        print(outputs)

    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    # Set environment variables before anything else
    os.environ["NCCL_DEBUG"] = "INFO"  # or "WARN" for more verbose output
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Force spawn method for multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    asyncio.run(main())
    
