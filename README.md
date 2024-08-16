# AIrmageddon
`AIrmageddon` is a prototype project that showcases how to run basic visual reasoning on a camera, video or list of images. In this experiment two LLM agents talk to each other to understand if the image shows a potential threat, then they decide whether or not to alert the owner.

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
Copy `conf/config-sample.yml` to `conf/config.yml` and edit the file accordingly. Default values should work for most scenarios it is still required to set `camera_feed` and `models_folder`.
If you prefer environment variables, or you're running the script in Docker, copy the `.env-sample` into `.env` and edit it accordingly.
Remember that environment variables override configuration variables.

- `camera_feed`: path to your camera, a video or a folder with images or files to analyze (eg: rtsp://camera.local/feed)
- `llama_server`: `llama-server` address
- `llama_server_completion`: `llama-server` address for /completion api
- `openai_server_completion`: `openai` or compatible providers completion API endpoint
- `chatbot_voice_enabled`: enable or disable (`true` or `false`) text-to-speech in the chatbot UI, requires `espeak`
- `use_local_llm`: whether to do inference with the local `llama_server` or using the `openai_server_completion` API
- `owners_away`: set the initial status of the owners when the script starts
- `models_folder`: path to the folder containing all models' weights
- `llm_weights`: LLM's weights filename
- `yolo_weights`: YOLO's weights filename
- `redis_host:` Redis host
- `redis_port`: Redis port
- `redis_db`: Redis db
- `openai_api_key`: any string if you're using a local LLM otherwise it will be your API key for that provider
- `guard_max_memories`: the maximum number of memories to transfer to the agents
- `min_frame_similarity`: the minimum similarity difference (from 0.0 to 1.0) to trigger an analysis from the LLM
- `deploy_id`: if you're using DeepInfra as your OpenAI compatible provider
- `model_name`: not necessary if you're using a local llm, otherwise meta-llama/Meta-Llama-3-8B-Instruct or the model you prefer
- `captures_folder`: path to the folder where you want to store all images and responses analyzed by the LLM
- `pn_username`: Your `Pushnotifier` username (required by the API)
- `pn_password`: Your `Pushnotifier` password (required by the API)
- `pn_package_name`: Your `Pushnotifier` package name (required by the API), you can create this from [here](https://pushnotifier.de/account/api)
- `pn_api_key`: Your `Pushnotifier` API KEY (required by the API), you can obtain this from [here](https://pushnotifier.de/account/api)

The only parameter you'll probably want to play with is the `min_frame_similarity`, empirically anything from `0.75` to `0.89` seems to work well. Higher values will trigger the analysis on too many frames, especially if there's movement, lower values reduce the sensitivity and only trigger an analysis when two frames are very different.

# Running the Project
- Clone this project: `git clone https://github.com/RamonBeast/airmageddon.git`
- `cd airmageddon`
- Create a `virtualenv`: `virtualenv venv` (`pip install virtualenv`)
- Activate it: `. ./venv/bin/activate`
- Install all requirements: `pip install -r requirements.txt`
- Copy `conf/config-sample.yml` to `conf/config.yml` and edit it (or do the same with `.env-sample` in the project root if you use docker)
- Make sure `redis` is running
- Start `llama-server`: `llama-server --model /models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf --ctx-size 16384 -ngl 99 --simple-io --threads 30` (these settings work well on Apple Silicon M2)
- Run the chatbot: `chainlit run chatbot.py -w`
- Start the alarm system: `python monitor.py` (this will stream all conversations between the agents)

## Chatbot
The chatbot is available by default on `http://localhost:8000/`, if you tell the chatbot that you are back home, or you are leaving home, **it will set the state accordingly** and the other agents will know whether you're home or not. This changes the way they alert about potential threats, for instance, if you're watching TV, the agents won't usually send alerts. Sometimes they will alert you anyway with notifications like `no threats, everything is ok` or funnily with `Person lying on the couch, they might be hurt` or similar silly alerts. If you leave home, the agents will usually trigger a notification if they see someone on the cameras.

## Prompts
System prompts are defined in the `agents` of the `conf/config.yml` file. If you're using the smaller 8B model, agents will lose attention easily and they will start to hallucinate, especially if they're passed on too many memories (hence, the default is `3`). For longer conversations they might have troubles calling functions correctly. Larger models, like 70B, do handle the prompt more efficiently and hallucinate less.

## Model Parameters
Other settings like `temperature` and `min_p` are set in `brain.py`, default values are sensible for this specific use case as if the LLM gets too creative, it will have troubles handling function calling. This problem is solved with slightly larger models.

## Custom Functions
You can implement as many custom functions as you like, such as to allow the Agents to turn cameras on or off, turn lights on or off etc depending on your tolerance for chaos. Functions should be implemented in `functions.py` and usage should be specified in the agent's system prompt in `conf/config.yml`.

# Disclaimer
Please don't use this project for anything serious. Not only it's awfully written, but also LLMs are quite unreliable, especially at smaller size. Fun is ensured, but don't leave them in charge of your home security, they will make a mess and give you several heart attacks.