# retry_test.py  -  Deliberately trigger failures to test retry logic
import ollama
import json
from pydantic import BaseModel, ValidationError

MODEL = 'qwen3-vl:4b'

class StrictSchema(BaseModel):
    name:    str
    age:     int     # must be integer, not string
    city:    str
    score:   float   # must be float


def test_with_bad_prompt():
    '''
    Use a vague prompt that will likely produce wrong output,
    then watch the retry kick in.
    '''
    print('TEST: Sending intentionally vague prompt (likely to fail)...')

    # Bad prompt: does not specify JSON
    bad_system = 'Tell me about a person.'
    user_msg = 'Give me details about someone named Alex who is 28, lives in Austin, scored 8.5.'

    last_error = None
    for attempt in range(2):
        if attempt == 0:
            messages = [{'role': 'system', 'content': bad_system},
                        {'role': 'user',   'content': user_msg}]
        else:
            strict_system = 'Return ONLY this JSON: { "name": "...", "age": <integer>, "city": "...", "score": <float> }'
            messages = [{'role': 'system', 'content': strict_system},
                        {'role': 'user',   'content': user_msg}]
            print(f'  -> Retrying with strict JSON instruction...')

        raw = ollama.chat(model=MODEL, messages=messages, options={'temperature': 0.3})
        text = raw['message']['content']
        print(f'  Attempt {attempt+1} raw output: {text[:120]}...')

        try:
            # Simple extraction in case it's wrapped
            import re
            match = re.search(r'({.*})', text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                data = json.loads(text.strip())
            
            result = StrictSchema(**data)
            print(f'  SUCCESS on attempt {attempt+1}: {result.model_dump()}')
            return
        except Exception as e:
            last_error = str(e)
            print(f'  FAILED: {last_error[:100]}')

    print(f'  Both attempts failed. Last error: {last_error}')
    print('  -> System would surface error to user cleanly.')


def test_pydantic_validation():
    '''Show Pydantic catching wrong types directly.'''
    print('\nTEST: Pydantic catching type errors...')

    test_cases = [
        {'name': 'Alex', 'age': 28,     'city': 'Austin', 'score': 8.5},   # valid
        {'name': 'Alex', 'age': '28',   'city': 'Austin', 'score': 8.5},   # age is string
        {'name': 'Alex', 'age': 28,     'city': 'Austin'},                  # missing score
        {'name': 'Alex', 'age': 28,     'city': 'Austin', 'score': 'high'}, # score is string
    ]

    for i, case in enumerate(test_cases):
        try:
            result = StrictSchema(**case)
            print(f'  Case {i+1}: VALID   -> {result.model_dump()}')
        except ValidationError as e:
            print(f'  Case {i+1}: INVALID -> {str(e.errors()[0]["msg"])} (field: {e.errors()[0]["loc"][0]})')


if __name__ == '__main__':
    print('=' * 55)
    print('Retry Logic & Validation Demo')
    print('=' * 55)
    test_pydantic_validation()
    print()
    test_with_bad_prompt()
