# AIarmageddon
`AI armageddon` is a prototype project that showcases how to run basic visual reasoning on a camera, video or list of images. In this experiment two LLM agents talk to each other to understand if the image shows a potential threat, then they decide whether or not to alert the owner.

Each frame or image is passed through `YOLOv5`, if the current frame differs sufficiently from the previous one it is passed down to `Florence-2` for captioning. The caption is in turn passed down to an LLM (Llama 3.1 8B by default) that starts a conversation with a second LLM, the first takes the role of a `Security Guard`, the second of a `Burglar`. When a decision is taken the `Security Guard` can either decide to move along with the analysis or raise an alert.

All events detected are logged as memories and provided to the LLM as context to enhance its analysis.

A `third agent` acts as a chatbot with the ability to signal to other agents when the owners are at home or they have left. At the same time, the chatbot can tell users what happened through the day.

## Hardware Requirements
The project runs reasonably well on `Apple Silicon M2`, using less than 5GB of RAM. It should work on most cpus.

## Software Requirements
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) (`brew install llama.cpp` on Mac)
- [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/) (`brew install redis` or `apt-get install redis-server`)
- [Espeak](https://github.com/espeak-ng/espeak-ng/tree/master) (`brew install espeak` on Mac)
- (Optional) [Pushnotifier](https://pushnotifier.de/) an account is required (it's free) together with an API KEY if you'd like to receive push notifications on your phone from the agents

## Model Weights
- [LLama 3.1 8B Instruct GGUF](https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main), `Q6_K` is suggested and has to be manually downloaded in the `models` folder
- [YOLOv5](https://github.com/ultralytics/yolov5/releases/), weights for `yolov5m6.pt` are automatically downloaded if not found in the `models` folder
- [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large), weights are automatically downloaded if the `Florence-2-large` folder is not found in the `models` folder

## Configuration
Copy the `.env-sample` into `.env` and edit it accordingly. Chances are you can leave everything as is, except for `CAMERA_FEED` and `MODELS_FOLDER`:

- `CAMERA_FEED`: path to your camera, a video or a folder with images or files to analyze
- `LLAMA_SERVER`: `llama-server` address
- `LLAMA_SERVER_COMPLETION`: `llama-server` completion API endpoint
- `CHATBOT_VOICE_ENABLED`: enable or disable (`true` or `false`) text-to-speech in the chatbot UI, requires `espeak`
- `MODELS_FOLDER`: path to the folder containing all models' weights
- `LLM_WEIGHTS`: LLM's weights filename
- `YOLO_WEIGHTS`: YOLO's weights filename
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `REDIS_DB`: Redis DB id
- `OPENAI_API_KEY`: leave at default is you're not using OpenAI's models
- `GUARD_MAX_MEMORIES`: the maximum number of memories to transfer to the agents
- `MIN_FRAME_SIMILARITY`: the minimum similarity difference (from 0.0 to 1.0) to trigger an analysis from the LLM
- `CAPTURES_FOLDER`: path to the folder where you want to store all images and responses analyzed by the LLM
- `PN_USERNAME`: Your `Pushnotifier` username (required by the API)
- `PN_PASSWORD`: Your `Pushnotifier` password (required by the API)
- `PN_PACKAGE_NAME`: Your `Pushnotifier` package name (required by the API), you can create this from [here](https://pushnotifier.de/account/api)
- `PN_API_KEY`: Your `Pushnotifier` API KEY (required by the API), you can obtain this from [here](https://pushnotifier.de/account/api)

The only parameter you'll probably want to play with is the `MIN_FRAME_SIMILARITY`, empirically anything from `0.75` to `0.89` seems to work well. Higher values will trigger the analysis on too many frames, lower values reduce the sensitivity and only trigger an analysis when two frames are very different.

# Running the Project
- Clone this project: `git clone https://github.com/RamonBeast/aiarmageddon.git`
- `cd airmageddon`
- Create a `virtualenv`: `virtualenv venv` (`pip install virtualenv`)
- Activate it: `. ./venv/bin/activate`
- Install all requirements: `pip install -r requirements.txt`
- Copy `.env-sample` to `.env` and edit it
- Make sure `redis` is running
- Start `llama-server`: `llama-server --model /models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf --ctx-size 16384 -ngl 99 --simple-io --threads 30` (these settings work well on Apple Silicon M2)
- Run the chatbot: `chainlit run chatbot.py -w`
- Start the alarm system: `python monitor.py` (this will stream all conversations between the agents)

## Chatbot
The chatbot is available by default on `http://localhost:8000/`, if you tell the chatbot that you are back home, or you are leaving home, **it will set the state accordingly** and the other agents will know whether you're home or not. This changes the way they alert about potential threats, for instance, if you're watching TV, the agents won't usually send alerts. Sometimes they will alert you anyway with notifications like `no threats, everything is ok` or funnily with `Person on the lying on the couch, they might be hurt` or similar silly alerts. If you leave home, the agents will usually trigger a notification if they see someone on the cameras.

## Prompts
System prompts are defined in `guard.py`, `burglar.py` and `chatbot.py` but keep in mind that, if you're using the small 8B model, agents will lose attention easily and they will start to hallucinate, especially if they're passed on too many memories (hence, the default is `3`). For longer conversations they might have troubles calling functions correctly.

## Model Parameters
Other settings like `temperature` and `min_p` are set in `brain.py`, default values are sensible for this specific use case as if the LLM gets too creative, it will have troubles handling function calling. This problem is solved with slightly larger models.

## Custom Functions
You can implement as many custom functions as you like, such as to allow the Agents to turn cameras on or off, turn lights on or off etc depending on your tolerance for chaos. Functions should be implemented in `functions.py` and usage should be specified in the system prompt of the agent that will use them, usually `guard.py` and `chatbot.py`.

# Disclaimer
Please don't use this project for anything serious. LLMs are quite unreliable, especially at smaller size. Fun is ensured, but don't leave them in charge of your home security, they will make a mess and give you several heart attacks.