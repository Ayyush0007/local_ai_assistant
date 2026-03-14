# temperature_test.py
# Run the same prompt at different temperatures and compare outputs
import ollama

MODEL = 'qwen3-vl:4b'

TEST_PROMPT = 'Give me 3 creative names for a coffee shop.'
TEMPERATURES = [0.0, 0.7]
RUNS_PER_TEMP = 3  # Run each temperature 3 times to show variance

print(f'Prompt: {TEST_PROMPT}')
print(f'Model:  {MODEL}\n')

for temp in TEMPERATURES:
    print(f'--- Temperature: {temp} ---')
    for run in range(1, RUNS_PER_TEMP + 1):
        response = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': TEST_PROMPT}],
            options={'temperature': temp}
        )
        print(f'Run {run}: {response["message"]["content"][:120]}...')
    print()

print('OBSERVATION:')
print('Temperature 0.0 = same answer every run (deterministic)')
print('Temperature 0.7 = different answer every run (creative/random)')
print()
print('When to use each:')
print('  temp=0.0  -> facts, data extraction, structured output (reliability matters)')
print('  temp=0.7  -> creative writing, brainstorming, variety matters)')
