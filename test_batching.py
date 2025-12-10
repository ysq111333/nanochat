"""
Test true dynamic batch processing implementation
Compare throughput across different batch sizes and verify output consistency
"""
import torch
import time
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine

def run_batch_inference(engine, prompts_tokens, max_tokens, temperature, top_k, seed):
    """Run batch inference and return results with timing"""
    start_time = time.time()
    results = engine.generate_batch_prompts_complete(
        prompts_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed
    )
    elapsed_time = time.time() - start_time
    
    # Calculate total tokens generated
    total_tokens = sum(len(result) for result in results)
    generated_tokens = sum(len(result) - len(prompt) for result, prompt in zip(results, prompts_tokens))
    
    return results, elapsed_time, total_tokens, generated_tokens

def run_serial_inference(engine, prompts_tokens, max_tokens, temperature, top_k, seed):
    """Run serial inference and return results with timing"""
    assistant_end = engine.tokenizer.encode_special("<|assistant_end|>")
    bos = engine.tokenizer.get_bos_token_id()
    
    start_time = time.time()
    serial_results = []
    
    for tokens in prompts_tokens:
        result_tokens = list(tokens)
        for token_column, token_masks in engine.generate(
            tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed
        ):
            token = token_column[0]
            if token != assistant_end and token != bos:
                result_tokens.append(token)
        serial_results.append(result_tokens)
    
    elapsed_time = time.time() - start_time
    total_tokens = sum(len(result) for result in serial_results)
    generated_tokens = sum(len(result) - len(prompt) for result, prompt in zip(serial_results, prompts_tokens))
    
    return serial_results, elapsed_time, total_tokens, generated_tokens

def test_with_batch_size(engine, all_prompts_tokens, batch_size, max_tokens, temperature, top_k, seed):
    """Test performance with a specific batch size"""
    # Select prompts for this batch size
    prompts_tokens = all_prompts_tokens[:batch_size]
    
    print(f"\n{'='*70}")
    print(f"Testing Batch Size: {batch_size}")
    print(f"{'='*70}")
    
    # Batch inference
    print(f"🔵 Running batch inference...")
    batch_results, batch_time, batch_total_tokens, batch_gen_tokens = run_batch_inference(
        engine, prompts_tokens, max_tokens, temperature, top_k, seed
    )
    
    # Serial inference
    print(f"🔴 Running serial inference...")
    serial_results, serial_time, serial_total_tokens, serial_gen_tokens = run_serial_inference(
        engine, prompts_tokens, max_tokens, temperature, top_k, seed
    )
    
    # Calculate metrics
    batch_throughput = batch_gen_tokens / batch_time  # tokens/second
    serial_throughput = serial_gen_tokens / serial_time
    speedup = serial_time / batch_time
    efficiency = (1 - batch_time / serial_time) * 100
    
    print(f"✅ Batch:  {batch_time:.3f}s, {batch_throughput:.1f} tokens/s")
    print(f"✅ Serial: {serial_time:.3f}s, {serial_throughput:.1f} tokens/s")
    print(f"📈 Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%")
    
    # Display sample outputs (show first 2 sequences)
    num_samples_to_show = min(2, batch_size)
    print(f"\n📝 Sample outputs (showing {num_samples_to_show}/{batch_size}):")
    for i in range(num_samples_to_show):
        # Get the generated portion (without prompt)
        generated_tokens = batch_results[i][len(prompts_tokens[i]):]
        generated_text = engine.tokenizer.decode(generated_tokens)
        
        # Get the prompt
        prompt_text = engine.tokenizer.decode(prompts_tokens[i])
        
        print(f"\n  [{i+1}] Prompt: {prompt_text[:60]}{'...' if len(prompt_text) > 60 else ''}")
        print(f"      Generated ({len(generated_tokens)} tokens): {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
    
    return {
        'batch_size': batch_size,
        'batch_time': batch_time,
        'serial_time': serial_time,
        'batch_throughput': batch_throughput,
        'serial_throughput': serial_throughput,
        'speedup': speedup,
        'efficiency': efficiency,
        'generated_tokens': batch_gen_tokens
    }

def print_comparison_table(results_table):
    """Print a formatted comparison table"""
    print(f"\n{'='*90}")
    print("📊 THROUGHPUT COMPARISON ACROSS BATCH SIZES")
    print(f"{'='*90}")
    print(f"{'Batch':<8} {'Batch Time':<12} {'Serial Time':<12} {'Batch TPS':<12} {'Serial TPS':<12} {'Speedup':<10} {'Efficiency':<10}")
    print(f"{'Size':<8} {'(seconds)':<12} {'(seconds)':<12} {'(tok/s)':<12} {'(tok/s)':<12} {'(x)':<10} {'(%)':<10}")
    print(f"{'-'*90}")
    
    for row in results_table:
        print(f"{row['batch_size']:<8} "
              f"{row['batch_time']:<12.3f} "
              f"{row['serial_time']:<12.3f} "
              f"{row['batch_throughput']:<12.1f} "
              f"{row['serial_throughput']:<12.1f} "
              f"{row['speedup']:<10.2f} "
              f"{row['efficiency']:<10.1f}")
    
    print(f"{'='*90}")
    
    # Print summary insights
    best_speedup = max(results_table, key=lambda x: x['speedup'])
    print(f"\n💡 Best speedup: {best_speedup['speedup']:.2f}x at batch size {best_speedup['batch_size']}")
    print(f"💡 Maximum throughput: {max(r['batch_throughput'] for r in results_table):.1f} tokens/s")

def verify_output_consistency(engine, prompts_tokens, max_tokens):
    """Verify that batch and serial inference produce identical outputs"""
    print(f"\n{'='*70}")
    print("🔍 VERIFYING OUTPUT CONSISTENCY (Greedy Decoding)")
    print(f"{'='*70}")
    
    # Use greedy decoding (temperature=0) and fixed seed for deterministic results
    test_prompts = prompts_tokens[:6]  # Test first 3 prompts
    
    print(f"Testing {len(test_prompts)} sequences with temperature=0.0, seed=42...")
    
    # Batch inference
    batch_results, _, _, _ = run_batch_inference(
        engine, test_prompts, max_tokens, temperature=0.0, top_k=None, seed=42
    )
    
    # Serial inference
    serial_results, _, _, _ = run_serial_inference(
        engine, test_prompts, max_tokens, temperature=0.0, top_k=None, seed=42
    )
    
    # Compare results
    all_match = True
    print()
    for i, (batch_res, serial_res, prompt_tokens) in enumerate(zip(batch_results, serial_results, test_prompts)):
        batch_gen = batch_res[len(prompt_tokens):]
        serial_gen = serial_res[len(prompt_tokens):]
        match = batch_gen == serial_gen
        status = "✅" if match else "❌"
        
        # Show prompt
        prompt_text = engine.tokenizer.decode(prompt_tokens)
        print(f"  Sequence {i+1}: {status} {'Match' if match else 'Mismatch'}")
        print(f"    Prompt: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")
        
        # Show generated text
        batch_text = engine.tokenizer.decode(batch_gen)
        print(f"    Output: {batch_text[:80]}{'...' if len(batch_text) > 80 else ''}")
        batch_text_seq = engine.tokenizer.decode(batch_gen)
        print(f"    Output: {batch_text[:80]}{'...' if len(batch_text_seq) > 80 else ''}")
        if not match:
            serial_text = engine.tokenizer.decode(serial_gen)
            print(f"    ❌ Batch tokens:  {batch_gen[:15]}...")
            print(f"    ❌ Serial tokens: {serial_gen[:15]}...")
            print(f"    ❌ Batch text:  {batch_text[:60]}...")
            print(f"    ❌ Serial text: {serial_text[:60]}...")
            all_match = False
        print()
    
    if all_match:
        print(f"🎉 All outputs match! Batch processing is correct.")
    else:
        print(f"⚠️  Outputs don't match! Need debugging.")
    
    return all_match

def test_batch_performance():
    """Test batch processing performance across different batch sizes"""
    
    print("="*70)
    print("DYNAMIC BATCH PROCESSING PERFORMANCE TEST")
    print("="*70)
    
    # Initialize
    print("\n📦 Loading model...")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    model, tokenizer, meta = load_model("sft", device, phase="eval")
    engine = Engine(model, tokenizer)
    bos_token_id = tokenizer.get_bos_token_id()
    
    # Prepare test prompts of different lengths
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "The quick brown fox jumps over the lazy dog and then",
        "Tell me a joke.",
        "In the beginning",
        "Python is a programming language that is",
        "Machine learning is",
        "The weather today is",
        "Once upon a time",
        "The meaning of life",
        "Artificial intelligence can",
        "In the year 2024",
        "Deep learning models",
        "Natural language processing",
        "Computer vision systems",
        "Neural networks are"
    ]
    
    max_tokens = 20
    temperature = 0.7
    top_k = 50
    seed = 42
    
    # Tokenize all prompts
    print(f"\n📝 Tokenizing {len(test_prompts)} prompts...")
    all_prompts_tokens = []
    for i, prompt in enumerate(test_prompts):
        tokens = tokenizer.encode(prompt, prepend=bos_token_id)
        if i < 8:  # Only show first 8
            print(f"  Prompt {i+1} ({len(tokens):2d} tokens): {prompt[:50]}...")
        all_prompts_tokens.append(tokens)
    
    if len(test_prompts) > 8:
        print(f"  ... and {len(test_prompts) - 8} more prompts")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results_table = []
    
    print(f"\n{'='*70}")
    print(f"TESTING BATCH SIZES: {batch_sizes}")
    print(f"Max tokens per sequence: {max_tokens}")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print(f"{'='*70}")
    
    # Run tests for each batch size
    for batch_size in batch_sizes:
        result = test_with_batch_size(
            engine, all_prompts_tokens, batch_size, 
            max_tokens, temperature, top_k, seed
        )
        results_table.append(result)
    
    # Print comparison table
    print_comparison_table(results_table)
    
    # Verify output consistency
    all_match = verify_output_consistency(engine, all_prompts_tokens, max_tokens=30)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ TEST COMPLETED")
    print(f"{'='*70}")
    print(f"✓ Tested {len(batch_sizes)} different batch sizes")
    print(f"✓ Output consistency: {'PASS ✅' if all_match else 'FAIL ❌'}")
    print(f"✓ Best speedup: {max(r['speedup'] for r in results_table):.2f}x")
    print(f"{'='*70}")
    
    return results_table

if __name__ == "__main__":
    test_batch_performance()

