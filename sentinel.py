from utils.logger import Logger
from agents.guard import Guard
from agents.burglar import Burglar
from utils.functions import LLMFunctions

"""
This class is used to analyse camera feeds and create memory blocks
"""
class Sentinel():
    def __init__(self, max_memories: int = 0):
        self.max_memories = max_memories
        self.guard = Guard(max_memories=max_memories)
        self.burglar = Burglar()
        self.llm_func = LLMFunctions()

    def analyze_feed(self, description: str) -> str | None:
        response = ''
        turns = 0

        # Kickstart the Guard by prompting it with the CameraFeed
        response = self.guard.send_message(f'CameraFeed: {description}')

        if response is None:
            Logger.error('Error sending message')
            return None

        response = f'Guard: {response}\nCameraFeed: {description}\n'

        #Logger.info(response)

        if self.llm_func.is_function_call(response):
            Logger.info(f'Decision taken without consultation: {response}')
            return response

        # The conversation can go on and on forever but we like the thrill of the unknown
        # so if that happens, we'll be there to witness it! 
        # AKA the loop is not interrupted on purpose
        while True:
            burg = self.burglar.send_message(response)
            #Logger.info(f'Ex-burglar: {burg}')
            
            response += 'Ex-burglar: ' + burg + '\n'

            decision = self.guard.send_message(response)
            #Logger.info(f'Guard: {decision}')

            if self.llm_func.is_function_call(decision):
                #Logger.info(f'Verdict achieved: {decision} at turn {turns}')
                break

            response += 'Guard: ' + decision + '\n'
            turns += 1

        return decision
