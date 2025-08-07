import os

import requests
from dotenv import load_dotenv
from vapi_python import Vapi

load_dotenv()
TOKEN = os.getenv("TOKEN")
PUBLIC_KEY = os.getenv("PUBLIC_KEY")

def create_assistant(system_prompt, first_message, kb_file):
    assistant_url = "https://api.vapi.ai/assistant"
    file_url = "https://api.vapi.ai/file"
    
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    with open(kb_file, "rb") as file:
        files = {'file': file}
        response = requests.request("POST", file_url, files=files, headers=headers)

    file_id = response.json()['id']

    payload = {
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en",
            "smartFormat": True
        },
        "model": {
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                }
            ],
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 1,
            "maxTokens": 525,
            "emotionRecognitionEnabled": True,
            "knowledgeBase": {
                "provider": "canonical",
                "topK": 5,
                "fileIds": [file_id]
            },
        },
        "firstMessageMode": "assistant-speaks-first",
        "name": "Travel Assistant",
        "firstMessage": first_message
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", assistant_url, json=payload, headers=headers)

    assistant_id = response.json()["id"]

    return {"id": assistant_id, "file id": file_id}

def update_assistant(assistant_id, system_prompt, first_message, kb_file = None):

    url = f"https://api.vapi.ai/assistant/{assistant_id}"

    if kb_file:
        file_url = "https://api.vapi.ai/file"
        
        headers = {
            "Authorization": f"Bearer {TOKEN}"
        }
        with open(kb_file, "rb") as file:
            files = {'file': file}
            response = requests.request("POST", file_url, files=files, headers=headers)

        file_id = response.json()['id']
    else:
        file_id = None

    payload = {
        "model": {
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                }
            ],
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 1,
            "maxTokens": 525,
            "emotionRecognitionEnabled": True,
            "knowledgeBase": {
                "provider": "canonical",
                "topK": 5,
                "fileIds": [file_id] if file_id else []
            },
        },
        "firstMessageMode": "assistant-speaks-first",
        "name": "Travel Assistant",
        "firstMessage": first_message
    }
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.request("PATCH", url, json=payload, headers=headers)

    return response.json()


def get_assistants():

    url = "https://api.vapi.ai/assistant"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.request("GET", url, headers=headers)

    return response.json()

def start_call(assistant_id):
    #c34cc950-ff91-493a-894e-d475014f88bc assistant id
    vapi = Vapi(api_key=PUBLIC_KEY)
    vapi.start(assistant_id=assistant_id)

# test start call
# start_call("c34cc950-ff91-493a-894e-d475014f88bc")
