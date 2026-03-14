# structured_chat.py  -  Phase 2: Structured output with validation & retry
import ollama
import json
import re
from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional
from enum import Enum

MODEL = 'qwen3-vl:4b'
MAX_RETRIES = 1  # Retry once before failing — keeps it fast


# ── SCHEMA DEFINITIONS (what valid output looks like) ────────────

class SentimentEnum(str, Enum):
    positive = 'positive'
    negative = 'negative'
    neutral  = 'neutral'

class SentimentResult(BaseModel):
    sentiment:  SentimentEnum  # must be one of the 3 values above
    score:      float          # 0.0 to 1.0
    reason:     str            # one sentence explanation

    @field_validator('score')
    @classmethod
    def score_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('score must be between 0 and 1')
        return round(v, 2)


class TaskItem(BaseModel):
    task:      str
    priority:  str   # high / medium / low
    deadline:  Optional[str] = None

class TaskList(BaseModel):
    tasks: list[TaskItem]
    total_count: int

    @field_validator('total_count')
    @classmethod
    def count_must_match(cls, v, info):
        tasks = info.data.get('tasks', [])
        if v != len(tasks):
            raise ValueError(f'total_count {v} does not match tasks length {len(tasks)}')
        return v


class QAResult(BaseModel):
    answer:     str
    confidence: str   # high / medium / low
    source:     str   # where the model thinks it got the answer


# ── CORE ENGINE: JSON extraction + validation + retry ────────────

def extract_json(text: str) -> dict:
    '''Extract JSON from model output even if wrapped in markdown.'''
    # Model sometimes wraps JSON in ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try to find raw JSON object
    match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f'No JSON found in response: {text[:100]}')


def query_with_schema(system_prompt: str, user_message: str, schema_class) -> dict:
    '''
    Send a prompt, extract JSON, validate against schema.
    Retries once if validation fails.
    Returns validated Pydantic object as dict.
    '''
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            messages = [
                {'role': 'system',    'content': system_prompt},
                {'role': 'user',      'content': user_message}
            ]
        else:
            # Retry with stricter instruction
            retry_msg = (
                f'Your previous response failed validation: {last_error}\n'
                f'Return ONLY a valid JSON object. No explanation. No markdown. Just JSON.'
            )
            messages = [
                {'role': 'system',    'content': system_prompt},
                {'role': 'user',      'content': user_message},
                {'role': 'assistant', 'content': str(last_error)},
                {'role': 'user',      'content': retry_msg}
            ]
            print(f'  [Retry {attempt}/{MAX_RETRIES}] reprompting with stricter instruction...')

        response = ollama.chat(model=MODEL, messages=messages,
                               options={'temperature': 0})  # temp=0 for reliability
        raw_text = response['message']['content']

        try:
            parsed_dict = extract_json(raw_text)
            validated   = schema_class(**parsed_dict)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = str(e)
            print(f'  [Attempt {attempt + 1}] Validation failed: {last_error[:80]}')

    raise RuntimeError(f'Failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}')


# ── TOOL 1: SENTIMENT ANALYSIS ──────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    system = '''You are a sentiment analysis engine.
Return ONLY a JSON object with exactly these fields:
{  "sentiment": "positive" | "negative" | "neutral",
   "score": <float 0.0-1.0>,
   "reason": "<one sentence>"
}
No other text. No markdown. Only the JSON object.'''

    return query_with_schema(system, f'Analyze: {text}', SentimentResult)


# ── TOOL 2: TASK EXTRACTION ─────────────────────────────────────

def extract_tasks(text: str) -> dict:
    system = '''You are a task extraction engine.
Extract all tasks from the text and return ONLY this JSON:
{  "tasks": [{"task": "...", "priority": "high|medium|low", "deadline": "..." or null}],
   "total_count": <number>
}
No other text. Only JSON.'''

    return query_with_schema(system, f'Extract tasks from: {text}', TaskList)


# ── TOOL 3: Q&A FORMATTER ───────────────────────────────────────

def answer_question(question: str) -> dict:
    system = '''You are a Q&A engine.
Answer the question and return ONLY this JSON:
{  "answer": "...",
   "confidence": "high|medium|low",
   "source": "general knowledge|calculation|reasoning"
}
No other text. Only JSON.'''

    return query_with_schema(system, question, QAResult)


# ── DEMO: Run all three tools ────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print('Phase 2: Structured Output Demo')
    print('=' * 55)

    # Tool 1: Sentiment
    print('\n[1] Sentiment Analysis')
    result = analyze_sentiment('I absolutely love this product, it changed my life!')
    print(f'  Result: {result}')
    print(f'  Sentiment: {result["sentiment"]} (score: {result["score"]})')

    # Tool 2: Task Extraction
    print('\n[2] Task Extraction')
    text = 'I need to finish the report by Friday, call John tomorrow morning, and buy groceries urgently.'
    result = extract_tasks(text)
    print(f'  Found {result["total_count"]} tasks:')
    for task in result['tasks']:
        print(f'    - [{task["priority"].upper()}] {task["task"]}')

    # Tool 3: Q&A
    print('\n[3] Q&A Formatter')
    result = answer_question('How many days are in a leap year?')
    print(f'  Answer: {result["answer"]}')
    print(f'  Confidence: {result["confidence"]} | Source: {result["source"]}')

    print('\n' + '=' * 55)
    print('All outputs validated by Pydantic.')
    print('Every field has correct type and value.')
    print('=' * 55)
