import os
import sys
import time
import json
from datetime import datetime
from sentinel import Sentinel
from florence import Florence
from PIL import Image
from utils.logger import Logger
from yolo.yolo_monitor import YoloMonitor
from utils.functions import LLMFunctions
from utils.listener import EventPublisher
from utils.configuration import Configuration

conf = Configuration()

def save_detection(image: Image, caption: str, response: str, act: bool):
    # This only has 1s resolution
    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(conf.get_config_param('captures_folder'), filename)

    resp = {
        'image': filename,
        'caption': caption,
        'response': response,
        'act': act
    }

    image.save(save_path + '.png')
    json.dump(resp, open(save_path + '.json', 'w'))

def main(args):
    Logger.info('Loading models...')

    video = YoloMonitor(source= args[1] if len(args) > 1 else conf.get_config_param('camera_feed'))
    video.warmup()
    Logger.info('Yolo loaded')

    florence = Florence()
    Logger.info('Florence is ready')

    sentinel = Sentinel(max_memories=int(conf.get_config_param('guard_max_memories')))
    Logger.info('Sentinel is active')

    llm_func = LLMFunctions()

    # Let's notify that we are starting our monitoring
    memory = EventPublisher()
    memory.create_memory('MonitoringStarted', True)

    f = []
    threshold = float(conf.get_config_param('min_frame_similarity'))

    # Monitoring loop
    while True:
        t0 = time.time()

        try:
            # Extract detections and features from each frame
            detections, feats = video.run()

            if len(f) == 0:
                f = feats
                continue

            sim = video.get_similarity(f, feats)
            f = feats

            # First frame is not compared against anything
            if sim < 0:
                Logger.info('Discarding frame, it was either the first or the resolution has been adjusted')
                continue

            # Check if image similarity hit the threshold
            if sim >= threshold:
                Logger.info(f'Skipping, similarity: {sim:0.2f}, detections: {detections}')
                time.sleep(2.0) # Throttle to 0.5 FPS
                continue

            # From here on, we start reasoning on the image itself
            Logger.warning(f'Change detected, similarity: {sim:0.2f}, detections: {detections}')

            image = video.get_image()

            if isinstance(image, Image.Image) == False:
                image = Image.fromarray(image)
                image = image.convert('RGB')

            # Pass the image to Florence
            frame_caption = florence.process_frame('<MORE_DETAILED_CAPTION>', image=image)

            if '<MORE_DETAILED_CAPTION>' in frame_caption:
                caption = frame_caption['<MORE_DETAILED_CAPTION>']
            else:
                Logger.warning(f'Florence did not return a caption, skipping frame')
                continue

            # Start the loop between the Guard and the Ex-Burglar
            response = sentinel.analyze_feed(caption)

            if response is None:
                Logger.error('Sentinel cannot analyze feed, terminating')
                return None
            
            if (tokens := sentinel.get_cumulative_tokens()) is not None:
                Logger.notify(f'[$] Cumulative tokens - prompt: {tokens["prompt_tokens"]}, completion: {tokens["completion_tokens"]}')
            
            if (func_name := llm_func.is_function_call(response)) != None:
                #Logger.warning(f'Passing control to Alarm: {response}')
                save_detection(image, caption, response, False)
                memory.create_memory(caption, func_name)
                llm_func.call_class_function(llm_func, response)
            else:
                pass
                #Logger.info('No dangers detected, continuing monitoring')
                #save_detection(image, caption, response, True)
        except StopIteration:
            break

        t1 = time.time()
        Logger.info(f'Frame processed in {t1 - t0:.6f}s')

if __name__ == "__main__":
    main(sys.argv)