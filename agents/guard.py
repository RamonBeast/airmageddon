from datetime import datetime
from agents.brain import Brain
from utils.logger import Logger
from utils.configuration import Configuration

"""
This class is used to analyse camera feeds and talk to the Burglar
"""
class Guard():
    def __init__(self, max_memories: int = 0, use_automations: bool = False):
        self.config = Configuration()
        self.use_automations = use_automations
        
        current_date = datetime.now().strftime("%d %b %Y")
        current_time = datetime.now().strftime("%H:%M")

        agent_prompt = self.config.get_agent_config('guard')

        if agent_prompt is None:
            self.brain = None
            Logger.error('Guard prompt could not be found in config')
        else:
            agent_prompt = agent_prompt.format(current_date=current_date, current_time=current_time)
            self.brain = Brain(agent_prompt, max_memories=max_memories, use_automations=self.use_automations)

    def send_message(self, description: str) -> str | None:
        if self.brain is None:
            Logger.error('Agent is not initialized correctly')
            return None
        
        """ Analyze the feed and automatically updates memory if needed """
        response = self.brain.send_message(description)

        Logger.info(f'[A] Guard - {response}')

        return response
    
    def get_cumulative_tokens(self) -> dict:
        return self.brain.get_cumulative_tokens()
