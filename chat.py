# chat.py  -  Local AI chat tool powered by Ollama
import ollama
import sys

# Which model to use — change this to try different models
MODEL = 'qwen3-vl:4b'

def check_ollama_running():
    import requests
    try:
        requests.get('http://localhost:11434', timeout=2)
        return True
    except:
        return False

def chat(user_message: str, history: list) -> str:
    # Add the new message to conversation history
    history.append({'role': 'user', 'content': user_message})

    # Send to local model
    response = ollama.chat(
        model=MODEL,
        messages=history
    )

    assistant_reply = response['message']['content']

    # Add reply to history so model remembers the conversation
    history.append({'role': 'assistant', 'content': assistant_reply})

    return assistant_reply

def main():
    if not check_ollama_running():
        print('ERROR: Ollama is not running!')
        print('Fix: Open the Ollama app from your Applications folder')
        sys.exit(1)

    print(f'Local AI Assistant ({MODEL})')
    print('Running 100% offline on your Mac — no API costs!')
    print('Type quit to exit, clear to reset conversation.\n')

    history = []
    system_prompt = {
        'role': 'system',
        'content': 'You are a helpful assistant. Be concise and clear.'
    }
    history.append(system_prompt)

    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print('Goodbye!')
            break
        if user_input.lower() == 'clear':
            history = [system_prompt]
            print('Conversation cleared.\n')
            continue

        reply = chat(user_input, history)
        print(f'\nAssistant: {reply}\n')

if __name__ == '__main__':
    main()
