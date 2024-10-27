"""Voice mode for Claude Engineer."""

import asyncio
import base64
import io
import json
import logging
import subprocess
from typing import Callable

import speech_recognition as sr
import websockets
from rich.panel import Panel
from pydub import AudioSegment
from pydub.playback import play
from rich.console import Console

from .utility import get_env_checked, is_installed, text_chunker

log = logging.getLogger(__name__)

# Define a list of voice commands
VOICE_COMMANDS = {
    "exit voice mode": "exit_voice_mode",
    "save chat": "save_chat",
    "reset conversation": "reset_conversation",
}


async def get_user_input(ctx: "Context", save_chat: Callable):
    user_input = await ctx.handle_voice_mode()
    if user_input is None:
        return

    stay_in_voice_mode, command_result = ctx.voice.process_voice_command(
        user_input,
        save_chat,
        ctx.voice_mode,
    )
    if not stay_in_voice_mode:
        ctx.exit_voice_mode()
        if command_result:
            ctx.print(Panel(command_result, style="cyan"))
    elif command_result:
        ctx.print(Panel(command_result, style="cyan"))


class VoiceMode:
    # TODO make kwargs-only
    def __init__(
        self,
        /,
        microphone: sr.Microphone,
        recognizer: sr.Recognizer,
        eleven_labs_api_key: str,
        voice_id: str,
        model_id: str,
    ):
        self.microphone = microphone
        self.recognizer = recognizer
        self.eleven_labs_api_key = eleven_labs_api_key
        self.voice_id = voice_id
        self.model_id = model_id

    def _reinitialize(self):
        log.info("Reinitializing speech recognition")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        log.info("Speech recognition initialized")

    @classmethod
    def build_from_env(
        cls,
    ):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        eleven_labs_api_key = get_env_checked("ELEVEN_LABS_API_KEY")
        voice_id = get_env_checked("VOICE_ID")
        model_id = get_env_checked("MODEL_ID", "eleven_turbo_v2_5")
        voice = VoiceMode(
            microphone=microphone,
            recognizer=recognizer,
            eleven_labs_api_key=eleven_labs_api_key,
            voice_id=voice_id,
            model_id=model_id,
        )
        with voice.microphone as source:
            voice.recognizer.adjust_for_ambient_noise(source, duration=1)
        return voice

    @classmethod
    def build(cls, **kwargs):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        voice = VoiceMode(microphone=microphone, recognizer=recognizer, **kwargs)
        with voice.microphone as source:
            voice.recognizer.adjust_for_ambient_noise(source, duration=1)
        return voice

    async def stream_audio(self, console: Console, audio_stream):
        """Stream audio data using mpv player."""
        if not is_installed("mpv"):
            console.print(
                "mpv not found. Installing alternative audio playback...",
                style="bold yellow",
            )
            # Fall back to pydub playback if mpv is not available
            audio_data = b"".join([chunk async for chunk in audio_stream])
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            play(audio)
            return

        mpv_process = subprocess.Popen(
            ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        console.print("Started streaming audio", style="bold green")
        try:
            async for chunk in audio_stream:
                if chunk:
                    mpv_process.stdin.write(chunk)
                    mpv_process.stdin.flush()
        except Exception as e:
            console.print(f"Error during audio streaming: {str(e)}", style="bold red")
        finally:
            if mpv_process.stdin:
                mpv_process.stdin.close()
            mpv_process.wait()

    async def text_to_speech(self, console: Console, text):
        if not self.eleven_labs_api_key:
            console.print(
                "ElevenLabs API key not found. Text-to-speech is disabled.",
                style="bold yellow",
            )
            console.print(text)
            return

        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"

        try:
            async with websockets.connect(
                uri, extra_headers={"xi-api-key": self.eleven_labs_api_key}
            ) as websocket:
                # Send initial message
                await websocket.send(
                    json.dumps(
                        {
                            "text": " ",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75,
                            },
                            "xi_api_key": self.eleven_labs_api_key,
                        }
                    )
                )

                # Set up listener for audio chunks
                async def listen():
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            if data.get("audio"):
                                yield base64.b64decode(data["audio"])
                            elif data.get("isFinal"):
                                break
                        except websockets.exceptions.ConnectionClosed:
                            logging.error("WebSocket connection closed unexpectedly")
                            break
                        except Exception as e:
                            logging.error(f"Error processing audio message: {str(e)}")
                            break

                # Start audio streaming task
                stream_task = asyncio.create_task(stream_audio(listen()))

                # Send text in chunks
                async for chunk in text_chunker(text):
                    try:
                        await websocket.send(
                            json.dumps({"text": chunk, "try_trigger_generation": True})
                        )
                    except Exception as e:
                        logging.error(f"Error sending text chunk: {str(e)}")
                        break

                # Send closing message
                await websocket.send(json.dumps({"text": ""}))

                # Wait for streaming to complete
                await stream_task

        except websockets.exceptions.InvalidStatusCode as e:
            logging.error(f"Failed to connect to ElevenLabs API: {e}")
            console.print(f"Failed to connect to ElevenLabs API: {e}", style="bold red")
            console.print("Fallback: Printing the text instead.", style="bold yellow")
            console.print(text)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {str(e)}")
            console.print(f"Error in text-to-speech: {str(e)}", style="bold red")
            console.print("Fallback: Printing the text instead.", style="bold yellow")
            console.print(text)

    async def voice_input(self, console: Console, max_retries: int = 3):
        for attempt in range(max_retries):
            # Reinitialize speech recognition objects before each attempt
            self._reinitialize()

            try:
                with self.microphone as source:
                    console.print("Listening... Speak now.", style="bold green")
                    audio = self.recognizer.listen(source, timeout=5)

                console.print("Processing speech...", style="bold yellow")
                text = self.recognizer.recognize_google(audio)
                console.print(f"You said: {text}", style="cyan")
                return text.lower()
            except sr.WaitTimeoutError:
                console.print(
                    f"No speech detected. Attempt {attempt + 1} of {max_retries}.",
                    style="bold red",
                )
                logging.warning(
                    f"No speech detected. Attempt {attempt + 1} of {max_retries}"
                )
            except sr.UnknownValueError:
                console.print(
                    f"Speech was unintelligible. Attempt {attempt + 1} of {max_retries}.",
                    style="bold red",
                )
                logging.warning(
                    f"Speech was unintelligible. Attempt {attempt + 1} of {max_retries}"
                )
            except sr.RequestError as e:
                console.print(
                    f"Could not request results from speech recognition service; {e}",
                    style="bold red",
                )
                logging.error(
                    f"Could not request results from speech recognition service; {e}"
                )
                return None
            except Exception as e:
                console.print(
                    f"Unexpected error in voice input: {str(e)}", style="bold red"
                )
                logging.error(f"Unexpected error in voice input: {str(e)}")
                return None

            # Add a short delay between attempts
            await asyncio.sleep(1)

        console.print(
            "Max retries reached. Returning to text input mode.", style="bold red"
        )
        logging.info(
            "Max retries reached in voice input. Returning to text input mode."
        )
        return None

    def process_voice_command(
        self,
        command: str,
        save_chat: Callable,
        reset_conversation: Callable,
    ):
        if command in VOICE_COMMANDS:
            action = VOICE_COMMANDS[command]
            if action == "exit_voice_mode":
                return False, "Exiting voice mode."
            elif action == "save_chat":
                filename = save_chat()
                return True, f"Chat saved to {filename}"
            elif action == "reset_conversation":
                reset_conversation()
                return True, "Conversation has been reset."
        return True, None


async def test_voice_mode(console: Console, save_chat, reset_conversation):
    global voice_mode
    voice_mode = True
    voice = VoiceMode.build_from_env()
    console.print(
        Panel(
            "Entering voice input test mode. Say a few phrases, then say 'exit voice mode' to end the test.",
            style="bold green",
        )
    )

    while voice_mode:
        user_input = await voice.voice_input(console)
        if user_input is None:
            voice_mode = False
            voice = None
            console.print(
                Panel("Exited voice input test mode due to error.", style="bold yellow")
            )
            break

        stay_in_voice_mode, command_result = voice.process_voice_command(
            user_input, save_chat, reset_conversation
        )
        if not stay_in_voice_mode:
            voice_mode = False
            voice = None
            console.print(Panel("Exited voice input test mode.", style="bold green"))
            break
        elif command_result:
            console.print(Panel(command_result, style="cyan"))

    console.print(Panel("Voice input test completed.", style="bold green"))
