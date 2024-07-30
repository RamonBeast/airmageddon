import os
import sys
import time
import json
from dotenv import load_dotenv
from datetime import datetime
from sentinel import Sentinel
from florence import Florence
from PIL import Image
from logger import Logger
from yolo.yolo_monitor import YoloMonitor
from functions import LLMFunctions
from listener import EventPublisher

def save_detection(image: Image, caption: str, response: str, act: bool):
    # This only has 1s resolution
    filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(os.getenv('CAPTURES_FOLDER'), filename)

    resp = {
        'image': filename,
        'caption': caption,
        'response': response,
        'act': act
    }

    image.save(save_path + '.png')
    json.dump(resp, open(save_path + '.json', 'w'))

def main(args):
    load_dotenv()

    Logger.info('Loading models...')

    video = YoloMonitor(source=os.getenv('CAMERA_FEED'))
    video.warmup()
    Logger.info('Yolo loaded')

    florence = Florence()
    Logger.info('Florence is ready')

    sentinel = Sentinel(max_memories=int(os.getenv('GUARD_MAX_MEMORIES')))
    Logger.info('Sentinel is active')

    llm_func = LLMFunctions()

    # Let's notify that we are starting our monitoring
    memory = EventPublisher()
    memory.create_memory('Monitoring started', 'None')

    f = []
    threshold = float(os.getenv('MIN_FRAME_SIMILARITY'))

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
                #time.sleep(1.0) # Throttle to 1 FPS
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
                Logger.info(f'Florence did not return a caption, skipping frame')
                continue

            # Start the loop between the Guard and the Ex-Burglar
            response = sentinel.analyze_feed(caption)
            
            if (func_name := llm_func.is_function_call(response)) != None:
                #Logger.warning(f'Passing control to Alarm: {response}')
                save_detection(image, caption, response, False)
                memory.create_memory(caption, func_name)
                llm_func.call_class_function(llm_func, response)
            else:
                Logger.info('No dangers detected, continuing monitoring')
                #save_detection(image, caption, response, True)
        except StopIteration:
            break

        t1 = time.time()
        Logger.info(f'Frame processed in {t1 - t0:.6f}s')

if __name__ == "__main__":
    main(sys.argv)