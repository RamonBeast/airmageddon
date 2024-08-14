import re
import json
from pushnotifier.PushNotifier import PushNotifier
from typing import Tuple
from utils.listener import EventPublisher
from utils.logger import Logger
from utils.configuration import Configuration

class LLMFunctions():
    _conf = Configuration()
    _memory = EventPublisher()
    _pn = PushNotifier(_conf.get_config_param('pn_username'), _conf.get_config_param('pn_password'), 
                          _conf.get_config_param('pn_package_name'), _conf.get_config_param('pn_api_key'))

    @classmethod
    def _parse_response(cls, response: str) -> Tuple[str, dict] | Tuple[None, None]:
        pattern = r'(?i)<function=([a-zA-Z0-9_]+)>(.*?)</function>'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            function_name = match.group(1)
            params = match.group(2)

            if params == '':
                params = '{}'

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
        if self._conf.get_config_param('pn_api_key') is None or self._conf.get_config_param('pn_package_name') is None:
            return

        self._pn.send_text(text)

    def next(self):
        Logger.info('LLM invoked next')

    def owners_away(self, away: bool):
        self._memory.create_memory('OwnersAway', away)
        Logger.info(f'LLM invoked owners_away = {away}')
        

