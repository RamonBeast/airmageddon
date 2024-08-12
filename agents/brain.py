import requests
import json
import os
from openai import OpenAI
from utils.listener import EventListener
from typing import List
from utils.logger import Logger

class Brain():
    system = ''
    events_memory = []

    def __init__(self, system_prompt: str, max_memories: int = 0, temperature: float = 0.01, use_openai: bool = False):
        self.system = system_prompt
        self.max_memories = max_memories
        self.temperature = temperature
        self.owners_away = True # Out of caution, we assume the owners are away by default
        self.listener = EventListener()
        self.listener.start()
        self.use_openai = use_openai
        self.deploy_id = None
        self.model_name = os.getenv('MODEL_NAME')

        if self.use_openai:
            self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_SERVER_COMPLETION'))
            self.deploy_id = os.getenv('DEPLOY_ID')
        else:
            self.openai = None

    def send_message(self, msg: str) -> str:
        # System prompt
        prompt = self.system

        # Handle memory
        if self.max_memories > 0 :
            self._update_memory()
        
            if len(self.events_memory) > 0:
                memories = '\n'.join(self.events_memory[-self.max_memories:])
                Logger.notify(f'Current memories: {memories}')
                prompt += f'PastEvents:\n{memories}'
            
        prompt += f'\nOwnersAway: {self.owners_away}\n'
        # Add terminator to the system prompt
        prompt += '<|eot_id|>\n'

        # User prompt
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        return self.send_openai(prompt) if self.use_openai else self.send_llama(prompt)
    
    def send_llama(self, prompt: str):
        req = {
            "stream": False,
            "n_predict": 512,
            "temperature": self.temperature,
            "stop": ["<|python_tag|>","</s>","<|end|>","<|eom_id|>","<|eot_id|>","<|end_of_text|>","<|im_end|>","<|EOT|>","<|END_OF_TURN_TOKEN|>","<|end_of_turn|>","<|endoftext|>","assistant","user"],
            "repeat_last_n": 256,
            "repeat_penalty": 1.08,
            "penalize_nl": False,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.02,
            "tfs_z": 1,
            "typical_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar": "",
            "n_probs": 0,
            "min_keep": 0,
            "image_data": [],
            "cache_prompt": True,
            "api_key": "",
            "prompt": prompt
        }

        response = requests.post(os.getenv('LLAMA_SERVER_COMPLETION'), json = req)

        if response.status_code == 200:
            return response.json()['content'].strip()
        else:
            return ''
        
    def send_openai(self, prompt: str):
        completion = self.openai.completions.create(
            model=self.deploy_id if self.deploy_id != None else self.model_name,
            prompt=prompt,
            stop=['<|eot_id|>', '<|python_tag|>', '<|eom_id|>'],
            stream=False
        )
        
        Logger.info(f'Prompt tokens: {completion.usage.prompt_tokens}, Completion tokens: {completion.usage.completion_tokens}')
        return completion.choices[0].text.lstrip()

    def _update_memory(self):
        """
        Retrieves all memories from the queue and adds them to a local array
        """
        while (msg := self.listener.get_message_non_blocking()) is not None:
            # Memories are deserialized in a dict, we want to pass strings to the LLM
            if msg['event'] == 'OwnersAway':
                self.owners_away = msg['action']
            else:
                self.events_memory.append(json.dumps(msg))

    def get_memory(self) -> List[str]:
        return self.events_memory