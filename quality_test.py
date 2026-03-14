# quality_test.py  -  Compare answer accuracy across models
import ollama

MODELS = ['llama3.2:1b', 'qwen3-vl:4b']

# Questions with known correct answers
# Score each manually: 1 = correct, 0.5 = partially correct, 0 = wrong
QUALITY_PROMPTS = [
    {
        'question': 'What is 17 multiplied by 24?',
        'correct':  '408',
        'type':     'math'
    },
    {
        'question': 'In what year did World War II end?',
        'correct':  '1945',
        'type':     'factual'
    },
    {
        'question': 'What does HTTP stand for?',
        'correct':  'HyperText Transfer Protocol',
        'type':     'technical'
    },
    {
        'question': 'If you have 3 apples and give away 1/3, how many do you have left?',
        'correct':  '2',
        'type':     'reasoning'
    },
    {
        'question': 'Name the three states of matter.',
        'correct':  'solid liquid gas',
        'type':     'knowledge'
    },
]

def test_quality(model: str) -> list:
    results = []
    for q in QUALITY_PROMPTS:
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'Answer briefly and directly. One sentence maximum.'},
                    {'role': 'user',   'content': q['question']}
                ],
                options={'temperature': 0}
            )
            answer = response['message']['content'].strip()
            results.append({
                'type':     q['type'],
                'question': q['question'],
                'correct':  q['correct'],
                'answer':   answer,
            })
        except Exception as e:
            results.append({
                'type': q['type'],
                'question': q['question'],
                'correct': q['correct'],
                'answer': f"ERROR: {e}",
            })
    return results

if __name__ == '__main__':
    print('Quality Test — Score each answer manually (1 = correct, 0 = wrong)')
    print('=' * 65)

    for model in MODELS:
        print(f'\nMODEL: {model}')
        print('-' * 65)
        results = test_quality(model)
        for r in results:
            print(f'  [{r["type"]:10s}] Q: {r["question"][:45]}')
            print(f'             Expected: {r["correct"]}')
            print(f'             Got:      {r["answer"][:80]}')
            print()

    print('=' * 65)
    print('Fill in your scores in the generate_report.py file below.')
