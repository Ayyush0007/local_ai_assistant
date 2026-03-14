# benchmark.py  -  Measure local model performance rigorously
import ollama
import time
import json
from datetime import datetime

MODEL = 'qwen3-vl:4b'

# 5 test prompts covering different task types
# This variety matters — latency differs by task complexity
TEST_PROMPTS = [
    {'id': 1, 'type': 'factual',    'prompt': 'What is the capital of Japan?'},
    {'id': 2, 'type': 'reasoning',  'prompt': 'If a train travels 60mph for 2.5 hours, how far does it go?'},
    {'id': 3, 'type': 'creative',   'prompt': 'Write a 3-sentence story about a robot learning to paint.'},
    {'id': 4, 'type': 'technical',  'prompt': 'Explain what a REST API is in simple terms.'},
    {'id': 5, 'type': 'summarize',  'prompt': 'Summarize the water cycle in 2 sentences.'},
]

def benchmark_single(prompt_data: dict) -> dict:
    prompt = prompt_data['prompt']

    # Measure time to FIRST token using streaming
    first_token_time = None
    full_response = ''
    token_count = 0

    start_time = time.time()

    # Stream the response so we can capture first-token time
    stream = ollama.chat(
        model=MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    for chunk in stream:
        token = chunk['message']['content']
        if token and first_token_time is None:
            first_token_time = time.time() - start_time
        full_response += token
        token_count += len(token.split())

    end_time = time.time()
    total_latency = end_time - start_time

    # Estimate tokens (words are a rough proxy)
    # A more precise count would use the model's tokenizer
    char_count = len(full_response)
    estimated_tokens = char_count / 4  # ~4 chars per token on average
    tokens_per_second = estimated_tokens / total_latency if total_latency > 0 else 0

    return {
        'id':                prompt_data['id'],
        'type':              prompt_data['type'],
        'prompt':            prompt[:60] + '...',
        'response_preview':  full_response[:100] + '...',
        'time_to_first_token_s': round(first_token_time, 2) if first_token_time else None,
        'total_latency_s':   round(total_latency, 2),
        'estimated_tokens':  round(estimated_tokens),
        'tokens_per_second': round(tokens_per_second, 1),
        'response_length_chars': char_count,
    }

def run_benchmark():
    print(f'Benchmarking model: {MODEL}')
    print(f'Running {len(TEST_PROMPTS)} test prompts...\n')

    results = []
    for p in TEST_PROMPTS:
        print(f"  [{p['id']}/{len(TEST_PROMPTS)}] {p['type']}: {p['prompt'][:50]}")
        result = benchmark_single(p)
        results.append(result)
        print(f"    Latency: {result['total_latency_s']}s | ",
              f"First token: {result['time_to_first_token_s']}s | ",
              f"Speed: {result['tokens_per_second']} tok/s")

    # Calculate averages
    avg_latency  = sum(r['total_latency_s'] for r in results) / len(results)
    avg_ttft     = sum(r['time_to_first_token_s'] for r in results if r['time_to_first_token_s']) / len(results)
    avg_tps      = sum(r['tokens_per_second'] for r in results) / len(results)

    # Print report
    print('\n' + '='*55)
    print(f'BENCHMARK REPORT — {MODEL}')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*55)
    print(f'Average total latency:    {avg_latency:.2f}s')
    print(f'Average time to 1st token:{avg_ttft:.2f}s')
    print(f'Average tokens/second:    {avg_tps:.1f} tok/s')
    print()
    print('Per-prompt breakdown:')
    for r in results:
        print(f"  {r['type']:12s} | {r['total_latency_s']:5.2f}s | {r['tokens_per_second']:5.1f} tok/s")
    print('='*55)

    # Save results to JSON for your documentation
    output = {
        'model': MODEL,
        'date': datetime.now().isoformat(),
        'averages': {
            'avg_latency_s': round(avg_latency, 2),
            'avg_ttft_s': round(avg_ttft, 2),
            'avg_tokens_per_second': round(avg_tps, 1)
        },
        'results': results
    }
    filename = f'benchmark_{MODEL.replace(":","_")}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nFull results saved to {filename}')

if __name__ == '__main__':
    run_benchmark()
