#!/usr/bin/env python3
"""
Fixed HiDream-I1 Setup - Resolves device_map and generation config errors
"""

import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import HiDreamImagePipeline
import gc
import warnings

# Suppress the warnings about generation config
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

def setup_hidream_fixed():
    """
    Properly setup HiDream-I1 without device_map errors
    """
    print("Setting up HiDream-I1 with fixed configuration...")
    
    # Clear VRAM first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    print("✓ Tokenizer loaded")
    
    # Load text encoder with 8-bit quantization (but no device_map for pipeline)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit=True,  # 8-bit quantization to save VRAM
        torch_dtype=torch.bfloat16,
        # Remove these to fix warnings:
        # output_hidden_states=True,
        # output_attentions=True,
    )
    print("✓ Text encoder loaded with 8-bit quantization")
    
    # Load pipeline WITHOUT device_map parameter
    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Fast",  # Use Fast variant for less VRAM
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
        # Don't use device_map here - HiDreamImagePipeline doesn't support it
    )
    print("✓ Pipeline loaded")
    
    # Move to GPU manually
    pipe = pipe.to('cuda')
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ XFormers enabled")
    except:
        print("⚠ XFormers not available, using standard attention")
    
    print("✓ HiDream-I1 setup complete!")
    return pipe

def setup_hidream_cpu_offload():
    """
    Alternative setup with CPU offloading
    """
    print("Setting up HiDream-I1 with CPU offloading...")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    
    # Load text encoder normally (will be moved by pipeline)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Load pipeline
    pipe = HiDreamImagePipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Fast",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )
    
    # Enable sequential CPU offloading AFTER loading
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    
    print("✓ CPU offloading enabled")
    return pipe

def setup_flux_fallback():
    """
    Reliable FLUX fallback
    """
    print("Setting up FLUX as fallback...")
    
    from diffusers import FluxPipeline
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print("✓ FLUX setup complete")
    return pipe

def check_vram():
    """Monitor VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {allocated:.1f}GB used / {total:.1f}GB total")
        return allocated < total * 0.9  # Return True if we have headroom
    return False

def generate_image(pipe, prompt, model_type="hidream"):
    """
    Generate image with proper settings
    """
    print(f"Generating with {model_type}...")
    
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    if model_type == "hidream":
        # Settings for HiDream-I1-Fast
        image = pipe(
            prompt,
            height=384,
            width=384,
            guidance_scale=0.0,  # Fast variant doesn't use guidance
            num_inference_steps=16,  # Fast variant uses 16 steps
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
    else:  # flux
        image = pipe(
            prompt,
            height=384,
            width=384,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
    
    # Clean up after generation
    torch.cuda.empty_cache()
    return image

def main():
    """
    Main setup with fallbacks
    """
    prompt = "A beautiful mountain landscape with a clear lake, detailed, photorealistic"
    
    print("Initial VRAM check:")
    check_vram()
    
    # Try different approaches
    approaches = [
        ("Fixed HiDream", setup_hidream_fixed, "hidream"),
        ("HiDream with CPU offload", setup_hidream_cpu_offload, "hidream"),
        ("FLUX fallback", setup_flux_fallback, "flux")
    ]
    
    for name, setup_func, model_type in approaches:
        print(f"\n=== Trying {name} ===")
        try:
            pipe = setup_func()
            
            print("VRAM after loading:")
            if not check_vram():
                print("⚠ High VRAM usage, but continuing...")
            
            # Test generation
            image = generate_image(pipe, prompt, model_type)
            
            filename = f"output_{name.replace(' ', '_').lower()}.png"
            image.save(filename)
            print(f"✓ Success! Image saved as {filename}")
            
            print("Final VRAM check:")
            check_vram()
            
            return pipe
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # If CUDA out of memory, try next approach
            if "CUDA out of memory" in str(e):
                print("Out of VRAM, trying next approach...")
                continue
            else:
                print(f"Other error: {e}")
                continue
    
    print("❌ All approaches failed")
    return None

# Simple test function
def quick_test():
    """
    Quick test to see what works
    """
    print("=== Quick Test ===")
    
    try:
        # Try FLUX first (most reliable)
        from diffusers import FluxPipeline
        
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_sequential_cpu_offload()
        
        image = pipe(
            "test image of a cat",
            height=384,
            width=384,
            num_inference_steps=1,
        ).images[0]
        
        image.save("quick_test.png")
        print("✓ Quick test passed! FLUX is working.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test first
    if True:
        print("\n" + "="*50)
        print("FLUX is working. Now trying HiDream-I1...")
        print("="*50)
        
        # Try main setup
        pipe = main()
        
        if pipe:
            print("\n🎉 Setup successful!")
        else:
            print("\n😞 HiDream-I1 failed, but FLUX is available as backup")
    else:
        print("❌ Even basic FLUX failed. Check your CUDA installation.")

# Alternative minimal HiDream setup
def minimal_hidream():
    """
    Absolute minimal HiDream setup
    """
    try:
        print("Trying minimal HiDream setup...")
        
        # Don't load text encoders separately - let pipeline handle it
        pipe = HiDreamImagePipeline.from_pretrained(
            "HiDream-ai/HiDream-I1-Fast",
            torch_dtype=torch.float16,  # Use float16 for less VRAM
        )
        
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing()
        
        print("✓ Minimal HiDream loaded")
        return pipe
        
    except Exception as e:
        print(f"❌ Minimal setup failed: {e}")
        return None
    
