from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ChatMessageHistory

import chainlit as cl
#from chainlit.element import ElementBased

import numpy as np
import scipy
import json
import torch
import os
from io import BytesIO
from TTS.api import TTS
from listener import EventListener
from functions import LLMFunctions
from dotenv import load_dotenv

load_dotenv()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/vctk/vits").to(device)
listener = EventListener()
llm_func = LLMFunctions()
events_memory = []
tts_enabled = (os.getenv('CHATBOT_VOICE_ENABLED', 'false').lower() == 'true')

#stt = whisper.load_model("small")

# @cl.step(type="tool")
# async def speech_to_text(audio_file):
#     buf_name, audio, mime = (audio_file)

#     # audio_segment = AudioSegment.from_file(BytesIO(audio), format="webm")
#     # audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
#     # audio_data = np.frombuffer(audio_segment.get_array_of_samples(), dtype=np.int16)
    
#     # output_filename = 'query.wav'
#     # audio_segment.export(output_filename, format="wav")

#     # buffer = BytesIO()
#     # buffer.name = "query.wav"

#     # # Text to speech to a wav
#     # wav = np.array(audio)
#     # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
#     # wav_norm = wav_norm.astype(np.int16)

#     # scipy.io.wavfile.write(buffer.name, 22050, wav_norm)
#     # result = stt.transcribe(output_filename)

#     with open('input.webm', 'wb') as f:
#         f.write(audio)

#     stream = ffmpeg.input('input.webm')
#     stream = ffmpeg.output(stream, 'output.wav', format='wav')
#     ffmpeg.run(stream)

#     result = stt.transcribe("output.wav")
#     return result['text']

@cl.step(type="tool")
async def text_to_speech(text: str):
    buffer = BytesIO()
    buffer.name = "response.wav"

    # Text to speech to a wav
    wav = tts.tts(text=text, speaker="p230")
    wav = np.array(wav)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)

    scipy.io.wavfile.write(buffer, 22050, wav_norm)
    buffer.seek(0)

    return buffer.name, buffer.read()

@cl.step(type="tool")
async def generate_answer(message, messages):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    full_response = ""

    async for chunk in runnable.astream(
        {"question": message.content, "history": messages},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # This streams the answer as text
        await msg.stream_token(chunk)

        full_response += chunk
        
    await msg.update()
    return full_response

@cl.on_chat_start
async def on_chat_start():
    while (msg := listener.get_message_non_blocking()) is not None:
        events_memory.append(json.dumps(msg))
    
    model = ChatOpenAI(streaming=True,
                openai_api_base=os.getenv('LLAMA_SERVER'),
                openai_api_key=os.getenv('OPENAI_API_KEY')
                )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You're an attentive home monitoring system and you will answer all questions asked by the owner. 
                You act like a bored teenager and you address the owner as dude, bro, bruh, fam and similar vocatives.
                You generally reply in very short sentences, like if entire weight of the world was on your shoulder because you're
                overdramatic. 
                You will voluntarily skip or omit all minor and non-threatening events in your memory, unless asked for them directly.
                Do not add emotions spelled as text as they won't be visible.

                You have access to the following function:

                Function 'owners_away' to track whether the owners are at home or not
                {{{{
                "name": "owners_away",
                "description": "Let the alarm system know if the owners away",
                "parameters": {{{{
                    "away": {{{{
                    "param_type": "bool",
                    "description": "True of the owners are away, False otherwise",
                    "required": true
                    }}}}
                }}}}
                }}}}

                If you choose to call a function ONLY reply in the following format:
                <{{{{start_tag}}}}={{{{function_name}}}}>{{{{parameters}}}}{{{{end_tag}}}}
                where

                start_tag => `<function`
                parameters => a JSON dict with the function argument name as key and function argument value as value.
                end_tag => `</function>`

                Here is an example,
                <function=example_function_name>{{{{"example_name": "example_value"}}}}</function>

                You call a function only if:
                - The owners tells you that they are leaving or that they have already left
                - The owners tells you that they're back home or they're coming back

                When you call a function, ONLY reply with the function.
                Remember the exact format to call a function. This is important.

                You have a comprehensive memory of the events after the PastEvents tag, use it to give information to the owner.

                PastEvents:
                {events_memory}
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    
    # Initialize an empty chat history
    chat_history = ChatMessageHistory()
    
    # Store the runnable and chat history in the user session
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("history", chat_history)

    await cl.Message(
        content="Say hi to your shiny new AI Alarm System!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("history")  # type: ChatMessageHistory

    # Add the user's message to the chat history
    chat_history.add_user_message(message.content)

    # Prepare the messages for the prompt
    messages = chat_history.messages

    response = await generate_answer(message, messages)

    if llm_func.is_function_call(response) != None:
        _, params = llm_func._parse_response(response)

        if 'away' not in params:
            return
        
        owner_status = f'Updating memory: '

        if params['away'] == True:
            owner_status += 'owners away'
        else:
            owner_status += 'owners are back at home'
        
        llm_func.call_class_function(llm_func, response)
        await cl.Message(owner_status).send()
        
        chat_history.add_ai_message(owner_status)
    else:
        if tts_enabled:
            output_name, output_audio = await text_to_speech(response)
        
            output_audio_el = cl.Audio(
                name=output_name,
                auto_play=True,
                mime="audio/wav",
                content=output_audio,
            )

            answer_message = await cl.Message(content="").send()

            answer_message.elements = [output_audio_el]
            await answer_message.update()

        # Add the AI's response to the chat history
        chat_history.add_ai_message(response)

# @cl.on_audio_chunk
# async def on_audio_chunk(chunk: cl.AudioChunk):
#     if chunk.isStart:
#         buffer = BytesIO()
#         # This is required for whisper to recognize the file type
#         buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
#         # Initialize the session for a new audio stream
#         cl.user_session.set("audio_buffer", buffer)
#         cl.user_session.set("audio_mime_type", chunk.mimeType)

#     # TODO: Use Gladia to transcribe chunks as they arrive would decrease latency
#     # see https://docs-v1.gladia.io/reference/live-audio
    
#     # For now, write the chunks to a buffer and transcribe the whole audio at the end
#     cl.user_session.get("audio_buffer").write(chunk.data)

# @cl.on_audio_end
# async def on_audio_end(elements: list[ElementBased]):
#     chat_history = cl.user_session.get("history")  # type: ChatMessageHistory

#     # Get the audio buffer from the session
#     audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
#     audio_buffer.seek(0)  # Move the file pointer to the beginning
#     audio_file = audio_buffer.read()
#     audio_mime_type: str = cl.user_session.get("audio_mime_type")

#     input_audio_el = cl.Audio(
#         mime=audio_mime_type, content=audio_file, name=audio_buffer.name
#     )
#     await cl.Message(
#         author="You", 
#         type="user_message",
#         content="",
#         elements=[input_audio_el, *elements]
#     ).send()
    
#     # Transcribe the audio to text
#     whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
#     response = await speech_to_text(whisper_input)

#     # Add the user's message to the chat history
#     chat_history.add_user_message(response)

#     # Prepare the messages for the prompt
#     messages = chat_history.messages

#     response = await generate_answer(response, messages)

#     # Add the AI's response to the chat history
#     chat_history.add_ai_message(response)

#     output_name, output_audio = await text_to_speech(response)
    
#     output_audio_el = cl.Audio(
#         name=output_name,
#         auto_play=True,
#         mime="audio/wav",
#         content=output_audio,
#     )
#     answer_message = await cl.Message(content="").send()

#     answer_message.elements = [output_audio_el]
#     await answer_message.update()

# if __name__ == "__main__":
#     from chainlit.cli import run_chainlit
#     run_chainlit(__file__)