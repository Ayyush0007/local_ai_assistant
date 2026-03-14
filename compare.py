# compare.py  -  Benchmark all 3 models on identical prompts
import ollama, time, json
from datetime import datetime

# The models to compare (using only locally available models)
MODELS = [
    'llama3.2:1b',
    'qwen3-vl:4b',
]

# Exact same prompts as Phase 1 — identical conditions
TEST_PROMPTS = [
    {'id': 1, 'type': 'factual',   'prompt': 'What is the capital of Japan?'},
    {'id': 2, 'type': 'reasoning', 'prompt': 'If a train travels 60mph for 2.5 hours, how far does it go?'},
    {'id': 3, 'type': 'creative',  'prompt': 'Write a 3-sentence story about a robot learning to paint.'},
    {'id': 4, 'type': 'technical', 'prompt': 'Explain what a REST API is in simple terms.'},
    {'id': 5, 'type': 'summarize', 'prompt': 'Summarize the water cycle in 2 sentences.'},
]

def benchmark_model(model_name: str) -> dict:
    print(f'\n  Benchmarking {model_name}...')
    results = []

    for p in TEST_PROMPTS:
        print(f'    [{p["id"]}/5] {p["type"]}')
        first_token_time = None
        full_response = ''
        start = time.time()

        try:
            stream = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': p['prompt']}],
                stream=True,
                options={'temperature': 0}
            )
            for chunk in stream:
                token = chunk['message']['content']
                if token and first_token_time is None:
                    first_token_time = time.time() - start
                full_response += token

            total = time.time() - start
            est_tokens = len(full_response) / 4
            tps = est_tokens / total if total > 0 else 0

            results.append({
                'type':              p['type'],
                'total_latency_s':   round(total, 2),
                'ttft_s':            round(first_token_time, 2) if first_token_time else None,
                'tokens_per_second': round(tps, 1),
                'response_length':   len(full_response),
                'response_preview':  full_response[:80],
            })
        except Exception as e:
            print(f'    Error with {model_name}: {e}')
            results.append({
                'type': p['type'],
                'error': str(e)
            })

    # Filter out errors for averages
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return {'model': model_name, 'error': 'No valid results'}

    avg_latency = sum(r['total_latency_s'] for r in valid_results) / len(valid_results)
    avg_ttft    = sum(r['ttft_s'] for r in valid_results if r.get('ttft_s')) / len(valid_results)
    avg_tps     = sum(r['tokens_per_second'] for r in valid_results) / len(valid_results)

    return {
        'model':   model_name,
        'averages': {
            'avg_latency_s':      round(avg_latency, 2),
            'avg_ttft_s':         round(avg_ttft, 2),
            'avg_tokens_per_sec': round(avg_tps, 1),
        },
        'per_prompt': results
    }

def print_comparison(all_results: list):
    print('\n' + '=' * 60)
    print('MODEL COMPARISON RESULTS')
    print('=' * 60)
    # Only include models that ran successfully
    successful_results = [r for r in all_results if 'averages' in r]
    if not successful_results:
        print("No successful benchmarks to compare.")
        return

    header = f'{"Metric":<28}' + ''.join(f'{r["model"]:>16}' for r in successful_results)
    print(header)
    print('-' * 60)

    metrics = [
        ('Avg Latency (s)',      'avg_latency_s'),
        ('Avg Time to 1st Token','avg_ttft_s'),
        ('Avg Tokens/sec',       'avg_tokens_per_sec'),
    ]
    for label, key in metrics:
        row = f'{label:<28}' + ''.join(f'{r["averages"][key]:>16}' for r in successful_results)
        print(row)
    print('=' * 60)

if __name__ == '__main__':
    print('=' * 60)
    print('Local Model Comparison Study')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Models: {MODELS}')
    print('=' * 60)

    all_results = []
    for model in MODELS:
        result = benchmark_model(model)
        all_results.append(result)

    print_comparison(all_results)

    # Save combined results
    output = {'date': datetime.now().isoformat(), 'results': all_results}
    with open('comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nFull results saved to comparison_results.json')
    print('Now run: python generate_report.py')
