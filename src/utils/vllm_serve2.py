#import multiprocessing
import atexit
import signal
import sys
from contextlib import contextmanager
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class ResourceManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        self.model = LLM(
            model="Qwen/Qwen2.5-Coder-7B-Instruct", 
            dtype="half", 
            tensor_parallel_size=2
        )
        
    def cleanup(self):
        if self.model:
            # Explicitly delete the model to release CUDA memory
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection to clean up any remaining objects
        import gc
        gc.collect()

@contextmanager
def managed_resources():
    manager = ResourceManager()
    try:
        manager.initialize()
        yield manager
    finally:
        manager.cleanup()

def signal_handler(signum, frame):
    print("Signal received, cleaning up...")
    sys.exit(0)

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on normal program exit
    #atexit.register(multiprocessing.current_process().close)
    
    with managed_resources() as resources:
        sampling_path = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512
        )
        
        message = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hello world!"}
        ]
        
        text = resources.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = resources.model.generate([text], sampling_params=sampling_path)
        print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn')  # More stable than fork for ML workloads
    main()
    
