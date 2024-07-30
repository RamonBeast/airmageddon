import redis
import threading
import json
import os
from queue import Queue, Empty
from datetime import datetime

class EventListener:
    def __init__(self, redis_client: redis.Redis = None, channel: str = 'events'):
        self.redis_client = redis_client if redis_client != None else \
            redis.Redis(host=os.getenv('REDIS_SERVER'), port=int(os.getenv('REDIS_PORT')), db=int(os.getenv('REDIS_DB')))
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
        self.redis_client = redis_client if redis_client != None else \
            redis.Redis(host=os.getenv('REDIS_SERVER'), port=int(os.getenv('REDIS_PORT')), db=int(os.getenv('REDIS_DB')))
        self.channel = channel

    def publish(self, msg: dict):
        m = json.dumps(msg)
        self.redis_client.publish(self.channel, m)

    def get_time(self) -> str:
        return datetime.now().strftime("%b %d %H:%M")

    def create_memory(self, event: str, action: str = None):
        self.publish ({
                'time': self.get_time(),
                'event': event,
                'action': action
            })
