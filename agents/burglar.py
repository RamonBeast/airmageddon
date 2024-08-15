from datetime import datetime
from agents.brain import Brain
from utils.logger import Logger
from utils.configuration import Configuration

"""
This class is used to dialogue with the Guard
"""
class Burglar():
    def __init__(self, max_memories: int = 0):
        self.config = Configuration()
        current_date = datetime.now().strftime("%d %b %Y")
        current_time = datetime.now().strftime("%H:%M")
        agent_prompt = self.config.get_agent_config('burglar')

        if agent_prompt is None:
            self.brain = None
            Logger.error('Burglar prompt could not be found in config')
        else:
            agent_prompt = agent_prompt.format(current_date=current_date, current_time=current_time)
            self.brain = Brain(agent_prompt, max_memories=max_memories)

        self.brain = Brain(agent_prompt, max_memories=max_memories)

    def send_message(self, description: str) -> str | None:
        if self.brain is None:
            Logger.error('Agent is not initialized correctly')
            return None

        response = self.brain.send_message(description)

        Logger.info(f'[A] Burglar - {response}')

        return response
    
    def get_cumulative_tokens(self) -> dict:
        return self.brain.get_cumulative_tokens()
