import re
import json
import os
from pushnotifier.PushNotifier import PushNotifier
from typing import Tuple
from utils.listener import EventPublisher
from utils.logger import Logger

class LLMFunctions():
    @classmethod
    def _parse_response(cls, response: str) -> Tuple[str, dict] | Tuple[None, None]:
        pattern = r'(?i)<function=([a-zA-Z0-9_]+)>(.*?)</function>'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            function_name = match.group(1)
            params = match.group(2)
            return function_name, json.loads(params)
        else:
            return None, None

    def call_function(self, function_name, params):
        func = getattr(self, function_name)

        return func(**params)
    
    @classmethod
    def is_function_call(cls, response) -> str | None:
        function_name, _ = cls._parse_response(response)

        return function_name if function_name != None else None

    @classmethod
    def call_class_function(cls, instance, response):
        function_name, params = cls._parse_response(response)
        return instance.call_function(function_name, params)
    
    def notify(self, text: str):
        if os.getenv('PN_API_KEY') == '' or os.getenv('PN_API_KEY') == None:
            return
        
        pn = PushNotifier(os.getenv('PN_USERNAME'), os.getenv('PN_PASSWORD'), os.getenv('PN_PACKAGE_NAME'), os.getenv('PN_API_KEY'))
        #Logger.info(f'LLM called notify(): {text}')
        pn.send_text(text)

    def next(self):
        Logger.info('LLM invoked next')

    def owners_away(self, away: bool):
        Logger.info(f'LLM Called owners_away with away set to {away}')
        memory = EventPublisher()

        memory.create_memory('OwnersAway', away)

