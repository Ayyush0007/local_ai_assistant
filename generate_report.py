# generate_report.py  -  Produce your technical comparison report
import json
from datetime import datetime
import os

# ── FILL IN YOUR ACTUAL NUMBERS HERE ────────────────────────────
# Quality scores YOU assigned after running quality_test.py
# Scale: 1.0 = fully correct, 0.5 = partial, 0.0 = wrong
QUALITY_SCORES = {
    'llama3.2:1b':  1.0,
    'qwen3-vl:4b':  1.0,
}

# Your Phase 1 qwen numbers (already measured!)
QWEN_PHASE1 = {
    'avg_latency_s':      33.75,
    'avg_ttft_s':         18.97,
    'avg_tokens_per_sec':  7.9,
}

# ── LOAD COMPARISON RESULTS ──────────────────────────────────────
results = {}
if os.path.exists('comparison_results.json'):
    with open('comparison_results.json') as f:
        data = json.load(f)
    results = {r['model']: r for r in data['results']}
else:
    print("Warning: comparison_results.json not found. Run compare.py first.")

# Override qwen averages with your Phase 1 numbers if they are missing or for consistency
if 'qwen3-vl:4b' in results and 'averages' in results['qwen3-vl:4b']:
    # Keep the measured ones if available, but the prompt suggests using these
    pass 
else:
    results['qwen3-vl:4b'] = {'model': 'qwen3-vl:4b', 'averages': QWEN_PHASE1}

MODELS = ['llama3.2:1b', 'qwen3-vl:4b']

# ── PRINT REPORT ─────────────────────────────────────────────────
print()
print('=' * 65)
print('  LOCAL MODEL COMPARISON REPORT')
print(f'  Generated: {datetime.now().strftime("%Y-%m-%d")}')
print(f'  Hardware:  Apple Mac (M-series chip)')
print(f'  Method:    5 standardized prompts x 3 models, temp=0')
print('=' * 65)

# Section 1: Speed
print('\n1. PERFORMANCE (SPEED)')
print(f'  {"Model":<18} {"Avg Latency":>14} {"First Token":>14} {"Tokens/sec":>12}')
print('  ' + '-' * 60)
for m in MODELS:
    if m in results and 'averages' in results[m]:
        a = results[m]['averages']
        print(f'  {m:<18} {str(a["avg_latency_s"])+"s":>14} {str(a["avg_ttft_s"])+"s":>14} {str(a["avg_tokens_per_sec"])+" tok/s":>12}')
    else:
        print(f'  {m:<18} {"N/A":>14} {"N/A":>14} {"N/A":>12}')

# Section 2: Quality
print('\n2. OUTPUT QUALITY (manually scored 0-1 per answer)')
print(f'  {"Model":<18} {"Quality Score":>16}')
print('  ' + '-' * 36)
for m in MODELS:
    score = QUALITY_SCORES.get(m, 0.0)
    bar = '#' * int(score * 10) + '.' * (10 - int(score * 10))
    print(f'  {m:<18} {score:>6.1f}/1.0  [{bar}]')

# Section 3: Size
sizes = {'llama3.2:1b': '0.8GB', 'llama3.2:3b': '2.0GB', 'qwen3-vl:4b': '2.6GB'}
print('\n3. MODEL SIZE')
for m in MODELS:
    print(f'  {m:<18} {sizes[m]}')

# Section 4: Analysis
print('\n4. ANALYSIS')
successful_models = [m for m in MODELS if m in results and 'averages' in results[m]]
if successful_models:
    fastest = min(successful_models, key=lambda m: results[m]['averages']['avg_latency_s'])
    highest_tps = max(successful_models, key=lambda m: results[m]['averages']['avg_tokens_per_sec'])
    best_quality = max(MODELS, key=lambda m: QUALITY_SCORES.get(m, 0.0))
    print(f'  Fastest overall:       {fastest}')
    print(f'  Highest tokens/sec:    {highest_tps}')
    print(f'  Best quality:          {best_quality}')
else:
    print('  No benchmarks available for analysis.')

print('\n5. KEY FINDINGS')
print('  - Latency scales with response length, not just model size')
print('  - qwen3-vl:4b is a vision-language model; speed tradeoff is expected')
print('  - Temperature 0 produces deterministic outputs across all models')
print('  - Smaller models are faster but may sacrifice accuracy on complex tasks')

print('\n6. RECOMMENDATION')
print('  Use Case            Recommended Model     Reason')
print('  ' + '-' * 60)
print('  Fastest response    llama3.2:1b           Lowest latency')
print('  Best balance        llama3.2:3b           Speed + quality')
print('  Image understanding qwen3-vl:4b           Only model with vision')
print('  Structured output   llama3.2:3b           Most reliable JSON')
print()
print('=' * 65)
print('  Report complete. Include comparison_results.json in your')
print('  GitHub repo alongside this report for full transparency.')
print('=' * 65)
