from datetime import datetime
import yaml
from agents.brain import Brain
from utils.logger import Logger

"""
This class is used to analyse camera feeds and talk to the Burglar
"""
class Guard():
    def __init__(self, max_memories: int = 0):
        current_date = datetime.now().strftime("%d %b %Y")
        current_time = datetime.now().strftime("%H:%M")

        with open('./conf/agents.yml', 'r') as file:
            prompt = yaml.safe_load(file)
            agent_prompt = prompt['agents']['guard']
            agent_prompt = agent_prompt.format(current_date=current_date, current_time=current_time)

        self.brain = Brain(agent_prompt, max_memories=max_memories)

    def send_message(self, description: str) -> str:
        """ Analyze the feed and automatically updates memory if needed """
        response = self.brain.send_message(description)

        Logger.info(f'[A] Guard - {response}')

        return response
