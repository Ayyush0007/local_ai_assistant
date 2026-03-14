# practical_tool.py  -  Meeting notes to structured summary
# Combines: local model + JSON schema + Pydantic + retry
import ollama, json, re
from pydantic import BaseModel
from typing import Optional
from structured_chat import query_with_schema

# Uses qwen3-vl:4b — the locally available model (pull llama3.2:3b to switch)
MODEL = 'qwen3-vl:4b'


class ActionItem(BaseModel):
    owner:    str
    task:     str
    due_date: Optional[str] = None


class MeetingSummary(BaseModel):
    title:          str
    key_decisions:  list[str]
    action_items:   list[ActionItem]
    next_meeting:   Optional[str] = None
    mood:           str   # productive / tense / inconclusive


SYSTEM_PROMPT = '''You are a meeting notes processor.
Extract structured information and return ONLY this JSON:
{
  "title": "...",
  "key_decisions": ["...", "..."],
  "action_items": [{"owner": "...", "task": "...", "due_date": "..." or null}],
  "next_meeting": "..." or null,
  "mood": "productive|tense|inconclusive"
}
No other text. Only JSON.'''

SAMPLE_NOTES = '''
Weekly sync - March 8
Attendees: Sarah, Tom, Priya

We decided to push the v2 launch to April 15 due to testing delays.
Tom will fix the login bug by Wednesday.
Sarah needs to send the updated designs to the client before EOD Friday.
Priya will set up the staging environment next week.
Everyone agreed the new dashboard looks great.
Next meeting: March 15 at 10am.
'''

if __name__ == '__main__':
    print('Processing meeting notes...')
    print('Input:', SAMPLE_NOTES[:100], '...\n')

    result = query_with_schema(SYSTEM_PROMPT, SAMPLE_NOTES, MeetingSummary)

    print('STRUCTURED OUTPUT:')
    print(f'  Title:    {result["title"]}')
    print(f'  Mood:     {result["mood"]}')
    print(f'  Decisions: {len(result["key_decisions"])}')
    for d in result['key_decisions']:
        print(f'    - {d}')
    print(f'  Action items: {len(result["action_items"])}')
    for a in result['action_items']:
        due = a['due_date'] or 'no deadline'
        print(f'    - {a["owner"]}: {a["task"]} (due: {due})')
    print(f'  Next meeting: {result["next_meeting"]}')
    print('\nAll fields validated by Pydantic.')
