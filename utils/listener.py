import redis
import threading
import json
from queue import Queue, Empty
from datetime import datetime
from typing import List
from conf.configuration import Configuration

class EventListener:
    def __init__(self, redis_client: redis.Redis = None, channel: str = 'events'):
        self.conf = Configuration()
        self.redis_client = redis_client if redis_client != None else \
            redis.Redis(host=self.conf.get_config_param('redis_server'), port=int(self.conf.get_config_param('redis_port')), 
                        db=int(self.conf.get_config_param('redis_db')))
        self.channel = channel
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(channel)
        self.message_queue = Queue()

    def _listen(self):
        for message in self.pubsub.listen():
            if message['type'] == 'message' and message['channel'] == b'events':
                #print(f'EventListener, got message: {message}')
                self.message_queue.put(message['data'].decode('utf-8'))

    def start(self):
        thread = threading.Thread(target=self._listen)
        thread.daemon = True
        thread.start()

    def get_message(self) -> dict:
        return self.message_queue.get()
    
    def get_message_non_blocking(self) -> dict | None:
        try:
            m = self.message_queue.get_nowait()

            return json.loads(m)
        except Empty:
            return None
        except ValueError:
            return None
        
class EventPublisher:
    def __init__(self, redis_client: redis.Redis = None, channel: str = 'events'):
        self.conf = Configuration()
        self.redis_client = redis_client if redis_client != None else \
            redis.Redis(host=self.conf.get_config_param('redis_server'), port=int(self.conf.get_config_param('redis_port')), 
                        db=int(self.conf.get_config_param('redis_db')))
        self.channel = channel

    def publish(self, msg: dict):
        m = json.dumps(msg)
        self.redis_client.publish(self.channel, m)

    def get_time(self) -> str:
        return datetime.now().strftime("%b %d %H:%M")

    def create_memory(self, event: str, action: str = None):
        memory = {
                'time': self.get_time(),
                'event': event,
                'action': action
            }
        
        self.publish(memory)
        self.redis_client.rpush('events', json.dumps(memory))

class MemoryManager:
    def __init__(self, redis_client: redis.Redis = None, channel: str = 'events'):
        self.conf = Configuration()
        self.redis_client = redis_client if redis_client != None else \
            redis.Redis(host=self.conf.get_config_param('redis_server'), port=int(self.conf.get_config_param('redis_port')), 
                        db=int(self.conf.get_config_param('redis_db')))
        self.channel = channel

    def get_memories(self, max_memories=500) -> List[str]:
        entries = self.redis_client.lrange('events', 0, max_memories)
        entries = [json.loads(entry) for entry in entries]
        
        return entries
    
    def get_last_run(self) -> List[str]:
        """ Returns memories from the last monitoring cycle in chronological order """
        monitoring_started = None

        entries = self.get_memories()

        for i in range(len(entries) - 1, -1, -1):
            if entries[i]['event'].casefold() == 'monitoringstarted' and entries[i]['action'] == True:
                monitoring_started = i
                break
        
        if monitoring_started is not None:
            # Remove the MonitoringStarted entry
            return entries[monitoring_started + 1:]
        else:
            return []

