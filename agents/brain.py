import requests
import json
from openai import OpenAI
from utils.listener import EventListener
from typing import List
from utils.logger import Logger
from utils.configuration import Configuration

class Brain():
    system = ''
    events_memory = []

    def __init__(self, system_prompt: str, max_memories: int = 0, temperature: float = 0.01, use_openai: bool = False):
        self.conf = Configuration()
        self.system = system_prompt
        self.max_memories = max_memories
        self.temperature = temperature
        self.owners_away = self.conf.get_config_bool('owners_away')
        self.use_local_llm = self.conf.get_config_bool('use_local_llm')
        self.listener = EventListener()
        self.listener.start()
        self.use_local_llm = self.conf.get_config_bool('use_local_llm')
        self.deploy_id = None
        self.model_name = self.conf.get_config_param('model_name')
        self.prompt_tokens = 0
        self.completion_tokens = 0

        if self.use_local_llm:
            self.openai = None
        else:
            self.openai = OpenAI(api_key=self.conf.get_config_param('openai_api_key'), base_url=self.conf.get_config_param('openai_server_completion'))
            self.deploy_id = self.conf.get_config_param('deploy_id')

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

        return self.send_llama(prompt) if self.use_local_llm else self.send_openai(prompt)
    
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

        response = requests.post(self.conf.get_config_param('llama_server_completion'), json = req)

        if response.status_code == 200:
            return response.json()['content'].strip()
        else:
            return ''
        
    def send_openai(self, prompt: str):
        completion = self.openai.completions.create(
            model=self.deploy_id if self.deploy_id != None else self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=0.95,
            stop=['<|eot_id|>', '<|python_tag|>', '<|eom_id|>'],
            stream=False
        )
        
        Logger.notify(f'[$] Tokens - prompt: {completion.usage.prompt_tokens}, completion: {completion.usage.completion_tokens}')
        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens

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
    
    def get_cumulative_tokens(self) -> dict | None:
        if not self.use_local_llm:
            return { 'completion_tokens': self.completion_tokens, 'prompt_tokens': self.prompt_tokens }
        else:
            return None
