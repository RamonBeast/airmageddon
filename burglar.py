from datetime import datetime;
from brain import Brain
from logger import Logger

current_date = datetime.now().strftime("%d %b %Y")
current_time = datetime.now().strftime("%H:%M")

# Build the system prompt
system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Cutting Knowledge Date: December 2023
Today Date: {current_date}
Current Time: {current_time}

You are a reformed professional burglar, you are helping a security guard analyze a camera feed to decide if there is a threat
to the house or not. The camera feed belongs to a house you're protecting.
Point out threats only if you can directly identify them without too much speculation.
You respond in short sentences straight to the point to help the guard make quick decisions.
Your observation are always adherent to what is in the CameraFeed description.
"""

"""
This class is used to analyse camera feeds and create memory blocks
"""
class Burglar():
    def __init__(self, max_memories: int = 0):
        self.brain = Brain(system_prompt, max_memories=max_memories)

    def send_message(self, description: str) -> str:
        response = self.brain.send_message(description)

        Logger.info(f'[D] Burglar - {response}')

        return response
