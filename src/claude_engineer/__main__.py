""" """

import asyncio
import base64
import datetime
import io
import json
import logging
import mimetypes
import os
import subprocess
import time
from typing import Any

from anthropic import APIError, APIStatusError
from dotenv import load_dotenv
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import typer

from .context import Context, validate_files_structure
from .tools import DEFAULT_TOOLS
from .voice import test_voice_mode

log = logging.getLogger(__name__)

# Configure logging
# print current working directory
log.info("cwd is ", os.getcwd())

# Load environment variables from .env file
load_dotenv_result = load_dotenv(".env")
log.error(f"load_dotenv_result: {load_dotenv_result}")


# 11 Labs TTS
tts_enabled = True
use_tts = False
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = "YOUR VOICE ID"
MODEL_ID = "eleven_turbo_v2_5"


async def get_user_input(prompt: str = "You: "):
    style = Style.from_dict(
        {
            "prompt": "cyan bold",
        }
    )
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)


def create_folders(paths):
    results = []
    for path in paths:
        try:
            # Use os.makedirs with exist_ok=True to create nested directories
            os.makedirs(path, exist_ok=True)
            results.append(f"Folder(s) created: {path}")
        except Exception as e:
            results.append(f"Error creating folder(s) {path}: {str(e)}")
    return "\n".join(results)


def list_files(path: str = ".") -> str:
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"


def run_shell_command(command):
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.CalledProcessError as e:
        return {
            "stdout": e.stdout,
            "stderr": e.stderr,
            "return_code": e.returncode,
            "error": str(e),
        }
    except Exception as e:
        return {"error": f"An error occurred while executing the command: {str(e)}"}


def scan_folder(folder_path: str, output_file: str) -> str:
    ignored_folders = {".git", "__pycache__", "node_modules", "venv", "env"}
    markdown_content = f"# Folder Scan: {folder_path}\n\n"
    total_chars = len(markdown_content)
    max_chars = 600000  # Approximating 150,000 tokens

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in ignored_folders]

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, folder_path)

            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith("text"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    file_content = f"## {relative_path}\n\n```\n{content}\n```\n\n"
                    if total_chars + len(file_content) > max_chars:
                        remaining_chars = max_chars - total_chars
                        if remaining_chars > 0:
                            truncated_content = file_content[:remaining_chars]
                            markdown_content += truncated_content
                            markdown_content += "\n\n... Content truncated due to size limitations ...\n"
                        else:
                            markdown_content += "\n\n... Additional files omitted due to size limitations ...\n"
                        break
                    else:
                        markdown_content += file_content
                        total_chars += len(file_content)
                except Exception as e:
                    error_msg = (
                        f"## {relative_path}\n\nError reading file: {str(e)}\n\n"
                    )
                    if total_chars + len(error_msg) <= max_chars:
                        markdown_content += error_msg
                        total_chars += len(error_msg)

        if total_chars >= max_chars:
            break

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return f"Folder scan complete. Markdown file created at: {output_file}. Total characters: {total_chars}"


def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.DEFAULT_STRATEGY)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {str(e)}"


async def send_to_ai_for_executing(ctx, code, execution_result):
    try:
        system_prompt = f"""
        You are an AI code execution agent. Your task is to analyze the provided code and its execution result from the 'code_execution_env' virtual environment, then provide a concise summary of what worked, what didn't work, and any important observations. Follow these steps:

        1. Review the code that was executed in the 'code_execution_env' virtual environment:
        {code}

        2. Analyze the execution result from the 'code_execution_env' virtual environment:
        {execution_result}

        3. Provide a brief summary of:
           - What parts of the code executed successfully in the virtual environment
           - Any errors or unexpected behavior encountered in the virtual environment
           - Potential improvements or fixes for issues, considering the isolated nature of the environment
           - Any important observations about the code's performance or output within the virtual environment
           - If the execution timed out, explain what this might mean (e.g., long-running process, infinite loop)

        Be concise and focus on the most important aspects of the code execution within the 'code_execution_env' virtual environment.

        IMPORTANT: PROVIDE ONLY YOUR ANALYSIS AND OBSERVATIONS. DO NOT INCLUDE ANY PREFACING STATEMENTS OR EXPLANATIONS OF YOUR ROLE.
        """

        response = ctx.client.beta.prompt_caching.messages.create(
            model=ctx.config.model.code_execution,
            max_tokens=2000,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this code execution from the 'code_execution_env' virtual environment:\n\nCode:\n{code}\n\nExecution Result:\n{execution_result}",
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        # Update token usage for code execution
        ctx.token_stats["code_execution"].update_from_response(response.usage)

        analysis = response.content[0].text

        return analysis

    except Exception as e:
        ctx.print(f"Error in AI code execution analysis: {str(e)}", style="bold red")
        return f"Error analyzing code execution from 'code_execution_env': {str(e)}"


def save_chat(ctx: Context):
    # Generate filename
    now = datetime.datetime.now()
    filename = f"Chat_{now.strftime('%H%M')}.md"

    # Format conversation history
    formatted_chat = "# Claude-3-Sonnet Engineer Chat Log\n\n"
    for message in ctx.conversation_history:
        if message["role"] == "user":
            formatted_chat += f"## User\n\n{message['content']}\n\n"
        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                formatted_chat += f"## Claude\n\n{message['content']}\n\n"
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "tool_use":
                        formatted_chat += f"### Tool Use: {content['name']}\n\n```json\n{json.dumps(content['input'], indent=2)}\n```\n\n"
                    elif content["type"] == "text":
                        formatted_chat += f"## Claude\n\n{content['text']}\n\n"
        elif message["role"] == "user" and isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "tool_result":
                    formatted_chat += (
                        f"### Tool Result\n\n```\n{content['content']}\n```\n\n"
                    )

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_chat)

    return filename


async def decide_retry(ctx: Context, tool_checker_response, edit_results, tool_input):
    try:
        if not edit_results:
            ctx.print(
                Panel(
                    "No edits were made or an error occurred. Skipping retry.",
                    title="Info",
                    style="bold yellow",
                )
            )
            return {"retry": False, "files_to_retry": []}

        response = ctx.client.messages.create(
            model=ctx.config.model.tool_checker,
            max_tokens=1000,
            system="""You are an AI assistant tasked with deciding whether to retry editing files based on the previous edit results and the AI's response. Respond with a JSON object containing 'retry' (boolean) and 'files_to_retry' (list of file paths).

Example of the expected JSON response:
{
    "retry": true,
    "files_to_retry": ["/path/to/file1.py", "/path/to/file2.py"]
}

Only return the JSON object, nothing else. Ensure that the JSON is properly formatted with double quotes around property names and string values.""",
            messages=[
                {
                    "role": "user",
                    "content": f"Previous edit results: {json.dumps(edit_results)}\n\nAI's response: {tool_checker_response}\n\nDecide whether to retry editing any files.",
                }
            ],
        )

        response_text = response.content[0].text.strip()

        # Handle list of dicts if necessary
        if isinstance(response_text, list):
            response_text = " ".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in response_text
            )
        elif not isinstance(response_text, str):
            response_text = str(response_text)

        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            ctx.print(
                Panel(
                    "Failed to parse JSON from AI response. Using fallback decision.",
                    title="Warning",
                    style="bold yellow",
                )
            )
            decision = {"retry": "retry" in response_text.lower(), "files_to_retry": []}

        files = tool_input.get("files", [])
        if isinstance(files, dict):
            files = [files]
        elif not isinstance(files, list):
            ctx.print(
                Panel(
                    "Error: 'files' must be a dictionary or a list of dictionaries.",
                    title="Error",
                    style="bold red",
                )
            )
            return {"retry": False, "files_to_retry": []}

        if not all(isinstance(item, dict) for item in files):
            ctx.print(
                Panel(
                    "Error: Each file must be a dictionary with 'path' and 'instructions'.",
                    title="Error",
                    style="bold red",
                )
            )
            return {"retry": False, "files_to_retry": []}

        valid_file_paths = set(file["path"] for file in files)
        files_to_retry = [
            file_path
            for file_path in decision.get("files_to_retry", [])
            if file_path in valid_file_paths
        ]

        retry_decision = {
            "retry": decision.get("retry", False),
            "files_to_retry": files_to_retry,
        }

        ctx.print(
            Panel(
                f"Retry decision: {json.dumps(retry_decision, indent=2)}",
                title="Retry Decision",
                style="bold cyan",
            )
        )
        return retry_decision

    except Exception as e:
        ctx.print(
            Panel(f"Error in decide_retry: {str(e)}", title="Error", style="bold red")
        )
        return {"retry": False, "files_to_retry": []}


async def execute_tool(
    ctx: Context, tool_name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    try:
        result = None
        is_error = False
        console_output = None

        if tool_name == "create_files":
            if isinstance(tool_input, dict) and "files" in tool_input:
                files = tool_input["files"]
            else:
                files = tool_input
            result = ctx.create_files(files)
        elif tool_name == "edit_and_apply_multiple":
            files = tool_input.get("files")
            if not files:
                result = "Error: 'files' key is missing or empty."
                is_error = True
            else:
                # Ensure 'files' is a list of dictionaries
                if isinstance(files, str):
                    try:
                        # Attempt to parse the JSON string
                        files = json.loads(files)
                        if isinstance(files, dict):
                            files = [files]
                        elif isinstance(files, list):
                            if not all(isinstance(file, dict) for file in files):
                                result = "Error: Each file must be a dictionary with 'path' and 'instructions'."
                                is_error = True
                    except json.JSONDecodeError:
                        result = "Error: 'files' must be a dictionary or a list of dictionaries, and should not be a string."
                        is_error = True
                elif isinstance(files, dict):
                    files = [files]
                elif isinstance(files, list):
                    if not all(isinstance(file, dict) for file in files):
                        result = "Error: Each file must be a dictionary with 'path' and 'instructions'."
                        is_error = True
                else:
                    result = (
                        "Error: 'files' must be a dictionary or a list of dictionaries."
                    )
                    is_error = True

                if not is_error:
                    # Validate the structure of 'files'
                    try:
                        files = validate_files_structure(files)
                    except ValueError as ve:
                        result = f"Error: {str(ve)}"
                        is_error = True

            if not is_error:
                result, console_output = await ctx.edit_and_apply_multiple(
                    files, tool_input["project_context"], is_automode=ctx.automode
                )
        elif tool_name == "create_folders":
            result = create_folders(tool_input["paths"])
        elif tool_name == "read_multiple_files":
            paths = tool_input.get("paths")
            recursive = tool_input.get("recursive", False)
            if paths is None:
                result = "Error: No file paths provided"
                is_error = True
            else:
                files_to_read = [
                    p
                    for p in (paths if isinstance(paths, list) else [paths])
                    if p not in ctx.file_contents
                ]
                if not files_to_read:
                    result = "All requested files are already in the system prompt. No need to read from disk."
                else:
                    result = ctx.read_multiple_files(files_to_read, recursive)
        elif tool_name == "list_files":
            result = list_files(tool_input.get("path", "."))
        elif tool_name == "tavily_search":
            result = ctx.tavily_search(tool_input["query"])
        elif tool_name == "stop_process":
            result = ctx.proc.stop_process(tool_input["process_id"])
        elif tool_name == "execute_code":
            process_id, execution_result = await ctx.proc.execute_code(
                tool_input["code"]
            )
            if execution_result.startswith("Process started and running"):
                analysis = "The process is still running in the background."
            else:
                analysis_task = asyncio.create_task(
                    send_to_ai_for_executing(ctx, tool_input["code"], execution_result)
                )
                analysis = await analysis_task
            result = f"{execution_result}\n\nAnalysis:\n{analysis}"
            if process_id in ctx.proc.running_processes:
                result += "\n\nNote: The process is still running in the background."
        elif tool_name == "scan_folder":
            result = scan_folder(tool_input["folder_path"], tool_input["output_file"])
        elif tool_name == "run_shell_command":
            result = run_shell_command(tool_input["command"])
        else:
            is_error = True
            result = f"Unknown tool: {tool_name}"

        return {
            "content": result,
            "is_error": is_error,
            "console_output": console_output,
        }
    except KeyError as e:
        log.error(f"Missing required parameter {str(e)} for tool {tool_name}")
        return {
            "content": f"Error: Missing required parameter {str(e)} for tool {tool_name}",
            "is_error": True,
            "console_output": None,
        }
    except Exception as e:
        log.error(f"Error executing tool {tool_name}: {str(e)}")
        return {
            "content": f"Error executing tool {tool_name}: {str(e)}",
            "is_error": True,
            "console_output": None,
        }


async def chat_with_claude(
    ctx: Context,
    user_input,
    image_path=None,
    current_iteration=None,
    max_iterations=None,
):
    # Input validation
    if not isinstance(user_input, str):
        raise ValueError("user_input must be a string")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError("image_path must be a string or None")
    if current_iteration is not None and not isinstance(current_iteration, int):
        raise ValueError("current_iteration must be an integer or None")
    if max_iterations is not None and not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be an integer or None")

    current_conversation = []

    if image_path:
        ctx.print(
            Panel(
                f"Processing image at path: {image_path}",
                title_align="left",
                title="Image Processing",
                expand=False,
                style="yellow",
            )
        )
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            ctx.print(
                Panel(
                    f"Error encoding image: {image_base64}",
                    title="Error",
                    style="bold red",
                )
            )
            return (
                "I'm sorry, there was an error processing the image. Please try again.",
                False,
            )

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": f"User input for image: {user_input}"},
            ],
        }
        current_conversation.append(image_message)
        ctx.print(
            Panel(
                "Image message added to conversation history",
                title_align="left",
                title="Image Added",
                style="green",
            )
        )
    else:
        current_conversation.append({"role": "user", "content": user_input})

    # Filter conversation history to maintain context
    filtered_conversation_history = []
    for message in ctx.conversation_history:
        if isinstance(message["content"], list):
            filtered_content = [
                content
                for content in message["content"]
                if content.get("type") != "tool_result"
                or (
                    content.get("type") == "tool_result"
                    and not any(
                        keyword in content.get("output", "")
                        for keyword in [
                            "File contents updated in system prompt",
                            "File created and added to system prompt",
                            "has been read and stored in the system prompt",
                        ]
                    )
                )
            ]
            if filtered_content:
                filtered_conversation_history.append(
                    {**message, "content": filtered_content}
                )
        else:
            filtered_conversation_history.append(message)

    # Combine filtered history with current conversation to maintain context
    messages = filtered_conversation_history + current_conversation

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # MAINMODEL call with prompt caching
            response = ctx.client.beta.prompt_caching.messages.create(
                model=ctx.config.model.main,
                max_tokens=8000,
                system=[
                    {
                        "type": "text",
                        "text": ctx.update_system_prompt(
                            current_iteration, max_iterations
                        ),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": json.dumps(DEFAULT_TOOLS),
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=messages,
                tools=DEFAULT_TOOLS,
                tool_choice={"type": "auto"},
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            # Update token usage for MAINMODEL
            ctx.token_stats["mainmodel"].update_from_response(response.usage)
            break  # If successful, break out of the retry loop
        except APIStatusError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                ctx.print(
                    Panel(
                        f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})",
                        title="API Error",
                        style="bold yellow",
                    )
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                ctx.print(
                    Panel(f"API Error: {str(e)}", title="API Error", style="bold red")
                )
                return (
                    "I'm sorry, there was an error communicating with the AI. Please try again.",
                    False,
                )
        except APIError as e:
            ctx.print(
                Panel(f"API Error: {str(e)}", title="API Error", style="bold red")
            )
            return (
                "I'm sorry, there was an error communicating with the AI. Please try again.",
                False,
            )
    else:
        ctx.print(
            Panel(
                "Max retries reached. Unable to communicate with the AI.",
                title="Error",
                style="bold red",
            )
        )
        return (
            "I'm sorry, there was a persistent error communicating with the AI. Please try again later.",
            False,
        )

    assistant_response = ""
    exit_continuation = False
    tool_uses = []

    for content_block in response.content:
        if content_block.type == "text":
            assistant_response += content_block.text
            if ctx.config.continuation_exit_phrase in content_block.text:
                exit_continuation = True
        elif content_block.type == "tool_use":
            tool_uses.append(content_block)

    ctx.print(
        Panel(
            Markdown(assistant_response),
            title="Claude's Response",
            title_align="left",
            border_style="blue",
            expand=False,
        )
    )

    if tts_enabled and use_tts:
        await ctx.voice.text_to_speech(ctx.console, assistant_response)

    # Display files in context
    if ctx.file_contents:
        files_in_context = "\n".join(ctx.file_contents.keys())
    else:
        files_in_context = "No files in context. Read, create, or edit files to add."
    ctx.print(
        Panel(
            files_in_context,
            title="Files in Context",
            title_align="left",
            border_style="white",
            expand=False,
        )
    )

    for tool_use in tool_uses:
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool_use_id = tool_use.id

        ctx.print(Panel(f"Tool Used: {tool_name}", style="green"))
        ctx.print(
            Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green")
        )

        # Always use execute_tool for all tools
        tool_result = await execute_tool(ctx, tool_name, tool_input)

        if isinstance(tool_result, dict) and tool_result.get("is_error"):
            ctx.print(
                Panel(
                    tool_result["content"],
                    title="Tool Execution Error",
                    style="bold red",
                )
            )
            edit_results = []  # Assign empty list due to error
        else:
            # Assuming tool_result["content"] is a list of results
            edit_results = tool_result.get("content", [])

        # Prepare the tool_result_content for conversation history
        tool_result_content = {
            "type": "text",
            "text": json.dumps(tool_result)
            if isinstance(tool_result, (dict, list))
            else str(tool_result),
        }

        current_conversation.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                ],
            }
        )

        current_conversation.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [tool_result_content],
                        "is_error": tool_result.get("is_error", False)
                        if isinstance(tool_result, dict)
                        else False,
                    }
                ],
            }
        )

        # Update the file_contents dictionary if applicable
        if tool_name in [
            "create_files",
            "edit_and_apply_multiple",
            "read_multiple_files",
        ] and not (isinstance(tool_result, dict) and tool_result.get("is_error")):
            if tool_name == "create_files":
                for file in tool_input["files"]:
                    if "File created and added to system prompt" in str(tool_result):
                        ctx.file_contents[file["path"]] = file["content"]
            elif tool_name == "edit_and_apply_multiple":
                edit_results = (
                    tool_result if isinstance(tool_result, list) else [tool_result]
                )
                for result in edit_results:
                    if isinstance(result, dict) and result.get("status") in [
                        "success",
                        "partial_success",
                    ]:
                        ctx.file_contents[result["path"]] = result.get(
                            "edited_content", ctx.file_contents.get(result["path"], "")
                        )
            elif tool_name == "read_multiple_files":
                # The file_contents dictionary is already updated in the read_multiple_files function
                pass

        messages = filtered_conversation_history + current_conversation

        try:
            tool_response = ctx.client.messages.create(
                model=ctx.config.model.tool_checker,
                max_tokens=8000,
                system=ctx.update_system_prompt(current_iteration, max_iterations),
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                messages=messages,
                tools=DEFAULT_TOOLS,
                tool_choice={"type": "auto"},
            )
            # Update token usage for tool checker
            ctx.token_stats["tool_checker"].update_from_response(tool_response.usage)

            tool_checker_response = ""
            for tool_content_block in tool_response.content:
                if tool_content_block.type == "text":
                    tool_checker_response += tool_content_block.text
            ctx.print(
                Panel(
                    Markdown(tool_checker_response),
                    title="Claude's Response to Tool Result",
                    title_align="left",
                    border_style="blue",
                    expand=False,
                )
            )
            if use_tts:
                await ctx.voice.text_to_speech(ctx.console, tool_checker_response)
            assistant_response += "\n\n" + tool_checker_response

            # If the tool was edit_and_apply_multiple, let the AI decide whether to retry
            if tool_name == "edit_and_apply_multiple":
                retry_decision = await decide_retry(
                    ctx,
                    tool_checker_response,
                    edit_results,
                    tool_input,
                )
                if retry_decision["retry"] and retry_decision["files_to_retry"]:
                    ctx.print(
                        Panel(
                            f"AI has decided to retry editing for files: {', '.join(retry_decision['files_to_retry'])}",
                            style="yellow",
                        )
                    )
                    retry_files = [
                        file
                        for file in tool_input["files"]
                        if file["path"] in retry_decision["files_to_retry"]
                    ]

                    # Ensure 'instructions' are present
                    for file in retry_files:
                        if "instructions" not in file:
                            file["instructions"] = (
                                "Please reapply the previous instructions."
                            )

                    if retry_files:
                        (
                            retry_result,
                            retry_console_output,
                        ) = await ctx.edit_and_apply_multiple(
                            retry_files, tool_input["project_context"]
                        )
                        ctx.print(
                            Panel(
                                retry_console_output, title="Retry Result", style="cyan"
                            )
                        )
                        assistant_response += (
                            f"\n\nRetry result: {json.dumps(retry_result, indent=2)}"
                        )
                    else:
                        ctx.print(
                            Panel("No files to retry. Skipping retry.", style="yellow")
                        )
                else:
                    ctx.print(
                        Panel("Claude has decided not to retry editing", style="green")
                    )

        except APIError as e:
            error_message = f"Error in tool response: {str(e)}"
            ctx.print(Panel(error_message, title="Error", style="bold red"))
            assistant_response += f"\n\n{error_message}"

    if assistant_response:
        current_conversation.append(
            {"role": "assistant", "content": assistant_response}
        )

    ctx.conversation_history = messages + [
        {"role": "assistant", "content": assistant_response}
    ]

    # Display token usage at the end
    ctx.display_token_usage()

    return assistant_response, exit_continuation


async def handle_image(ctx: Context):
    image_path = (
        (await get_user_input("Drag and drop your image here, then press enter: "))
        .strip()
        .replace("'", "")
    )

    if os.path.isfile(image_path):
        user_input = await get_user_input("You (prompt for image): ")
        response, _ = await chat_with_claude(ctx, user_input, image_path)
    else:
        ctx.print(
            Panel(
                "Invalid image path. Please try again.",
                title="Error",
                style="bold red",
            )
        )


async def main(console: Console, verbose: int = 0):
    # Command mapping
    commands = {
        "exit": cmd_exit,
        "test voice": cmd_test_voice,
        "11labs on": cmd_tts_on,
        "11labs off": cmd_tts_off,
        "reset": cmd_reset,
        "save chat": cmd_save_chat,
        "voice": cmd_voice,
    }

    logging.basicConfig(
        level={
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
        }.get(verbose, logging.ERROR),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ctx = Context.build_from_env(console)

    ctx.print(
        Panel(
            "Welcome to the Claude-3-Sonnet Engineer Chat with Multi-Agent, Image, Voice, and Text-to-Speech Support!",
            title="Welcome",
            style="bold green",
        )
    )
    ctx.print("Type 'exit' to end the conversation.")
    ctx.print("Type 'image' to include an image in your message.")
    ctx.print("Type 'voice' to enter voice input mode.")
    ctx.print("Type 'test voice' to run a voice input test.")
    ctx.print(
        "Type 'automode [number]' to enter Autonomous mode with a specific number of iterations."
    )
    ctx.print("Type 'reset' to clear the conversation history.")
    ctx.print("Type 'save chat' to save the conversation to a Markdown file.")
    ctx.print("Type '11labs on' to enable text-to-speech.")
    ctx.print("Type '11labs off' to disable text-to-speech.")
    ctx.print(
        "While in automode, press Ctrl+C at any time to exit the automode to return to regular chat."
    )

    continue_chat = True
    while continue_chat:
        user_input = await get_user_input()

        # Handle commands
        cmd = user_input.lower().strip()
        if cmd in commands:
            continue_chat = await commands[cmd](ctx, user_input)
            continue

        # Handle regular chat
        await cmd_chat(ctx, user_input)

        if user_input.lower() == "image":
            await handle_image(ctx)
            continue
        elif user_input.lower().startswith("automode"):
            try:
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_iterations = int(parts[1])
                else:
                    max_iterations = ctx.config.max_continuation_iterations

                automode = True
                ctx.print(
                    Panel(
                        f"Entering automode with {max_iterations} iterations. Please provide the goal of the automode.",
                        title_align="left",
                        title="Automode",
                        style="bold yellow",
                    )
                )
                ctx.print(
                    Panel(
                        "Press Ctrl+C at any time to exit the automode loop.",
                        style="bold yellow",
                    )
                )
                user_input = await get_user_input()

                iteration_count = 0
                error_count = 0
                max_errors = (
                    3  # Maximum number of consecutive errors before exiting automode
                )
                try:
                    while automode and iteration_count < max_iterations:
                        try:
                            response, exit_continuation = await chat_with_claude(
                                ctx,
                                user_input,
                                current_iteration=iteration_count + 1,
                                max_iterations=max_iterations,
                            )
                            error_count = 0  # Reset error count on successful iteration
                        except Exception as e:
                            ctx.print(
                                Panel(
                                    f"Error in automode iteration: {str(e)}",
                                    style="bold red",
                                )
                            )
                            error_count += 1
                            if error_count >= max_errors:
                                ctx.print(
                                    Panel(
                                        f"Exiting automode due to {max_errors} consecutive errors.",
                                        style="bold red",
                                    )
                                )
                                automode = False
                                break
                            continue

                        if (
                            exit_continuation
                            or ctx.config.continuation_exit_phrase in response
                        ):
                            ctx.print(
                                Panel(
                                    "Automode completed.",
                                    title_align="left",
                                    title="Automode",
                                    style="green",
                                )
                            )
                            automode = False
                        else:
                            ctx.print(
                                Panel(
                                    f"Continuation iteration {iteration_count + 1} completed. Press Ctrl+C to exit automode. ",
                                    title_align="left",
                                    title="Automode",
                                    style="yellow",
                                )
                            )
                            user_input = "Continue with the next step. Or STOP by saying 'AUTOMODE_COMPLETE' if you think you've achieved the results established in the original request."
                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            ctx.print(
                                Panel(
                                    "Max iterations reached. Exiting automode.",
                                    title_align="left",
                                    title="Automode",
                                    style="bold red",
                                )
                            )
                            automode = False
                except KeyboardInterrupt:
                    ctx.print(
                        Panel(
                            "\nAutomode interrupted by user. Exiting automode.",
                            title_align="left",
                            title="Automode",
                            style="bold red",
                        )
                    )
                    automode = False
                    if (
                        ctx.conversation_history
                        and ctx.conversation_history[-1]["role"] == "user"
                    ):
                        ctx.conversation_history.append(
                            {
                                "role": "assistant",
                                "content": "Automode interrupted. How can I assist you further?",
                            }
                        )
            except KeyboardInterrupt:
                ctx.print(
                    Panel(
                        "\nAutomode interrupted by user. Exiting automode.",
                        title_align="left",
                        title="Automode",
                        style="bold red",
                    )
                )
                automode = False
                if (
                    ctx.conversation_history
                    and ctx.conversation_history[-1]["role"] == "user"
                ):
                    ctx.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": "Automode interrupted. How can I assist you further?",
                        }
                    )

            ctx.print(
                Panel("Exited automode. Returning to regular chat.", style="green")
            )
        else:
            response, _ = await chat_with_claude(ctx, user_input)

    # Add more tests for other functions as needed


async def cmd_exit(ctx: Context, _: str) -> bool:
    ctx.print(
        Panel(
            "Thank you for chatting. Goodbye!",
            title_align="left",
            title="Goodbye",
            style="bold green",
        )
    )
    return False


async def cmd_test_voice(ctx: Context, _: str) -> bool:
    await test_voice_mode(ctx.console, save_chat, reset_conversation)
    return True


async def cmd_tts_on(ctx: Context, _: str) -> bool:
    ctx.toggle_tts(True)
    return True


async def cmd_tts_off(ctx: Context, _: str) -> bool:
    ctx.toggle_tts(False)
    return True


async def cmd_reset(ctx: Context, _: str) -> bool:
    ctx.reset()
    return True


async def cmd_save_chat(ctx: Context, _: str) -> bool:
    filename = save_chat(ctx)
    ctx.print(
        Panel(f"Chat saved to {filename}", title="Chat Saved", style="bold green")
    )
    return True


async def cmd_voice(ctx: Context, _: str) -> bool:
    ctx.enter_voice_mode()
    return True


async def cmd_chat(ctx: Context, user_input: str) -> bool:
    response, _ = await chat_with_claude(ctx, user_input)
    log.debug("Response: %s", response)
    return True


def main_stub(verbose: int = typer.Option(0, "--verbose", "-v", count=True)):
    # Run the main program
    console = Console()
    try:
        asyncio.run(main(console, verbose))
    except KeyboardInterrupt:
        console.print("Program interrupted by user. Exiting...", style="bold red")
    except Exception as e:
        console.print(f"An unexpected error occurred: {str(e)}", style="bold red")
        log.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        console.print("Program finished. Goodbye!", style="bold green")


if __name__ == "__main__":
    typer.run(main_stub)
