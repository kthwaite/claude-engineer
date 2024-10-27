""" """

import difflib
import glob
import logging
import os
import re
from typing import Any, Optional

from anthropic import Anthropic
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import (
    PromptCachingBetaMessage,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from tavily import TavilyClient

from .config import Config
from .process import ProcessHandler
from .prompts import AUTOMODE_SYSTEM_PROMPT, BASE_SYSTEM_PROMPT
from .tokens import TokenStats
from .utility import get_env_checked
from .voice import VoiceMode

log = logging.getLogger(__name__)


def validate_files_structure(
    files: dict[str, str] | list[dict[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(files, (dict, list)):
        raise ValueError(
            "Invalid 'files' structure. Expected a dictionary or a list of dictionaries."
        )

    if isinstance(files, dict):
        files = [files]

    for file in files:
        if not isinstance(file, dict):
            raise ValueError("Each file must be a dictionary.")
        if "path" not in file or "instructions" not in file:
            raise ValueError(
                "Each file dictionary must contain 'path' and 'instructions' keys."
            )
        if not isinstance(file["path"], str) or not isinstance(
            file["instructions"], str
        ):
            raise ValueError("'path' and 'instructions' must be strings.")

    return files


def highlight_diff(diff_text: str, theme: str = "monokai") -> Syntax:
    return Syntax(diff_text, "diff", theme=theme, line_numbers=True)


def generate_diff(original: str, new: str, path: str) -> Syntax:
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )

    diff_text = "".join(diff)
    return highlight_diff(diff_text)


def validate_ai_response(response_text: str | list[dict[str, str]]) -> bool:
    if isinstance(response_text, list):
        # Extract 'text' from each dictionary in the list
        try:
            response_text = " ".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in response_text
            )
        except Exception as e:
            log.error("Error processing response_text list: %s", str(e))
            raise ValueError("Invalid format in response_text list.")
    elif not isinstance(response_text, str):
        log.debug(
            "validate_ai_response received type %s: %s",
            type(response_text),
            response_text,
        )
        raise ValueError(
            f"Invalid type for response_text: {type(response_text)}. Expected string."
        )

    # Log the processed response_text
    log.debug("Processed response_text for validation: %s", response_text)

    if not re.search(r"<SEARCH>.*?</SEARCH>", response_text, re.DOTALL):
        raise ValueError("AI response does not contain any <SEARCH> blocks")
    if not re.search(r"<REPLACE>.*?</REPLACE>", response_text, re.DOTALL):
        raise ValueError("AI response does not contain any <REPLACE> blocks")
    return True


def parse_search_replace_blocks(
    response_text: str, use_fuzzy: bool = True
) -> list[dict[str, str]]:
    """Parse the response text for SEARCH/REPLACE blocks.

    Args:
        response_text: The text containing SEARCH/REPLACE blocks.
        use_fuzzy: Whether to use fuzzy matching for search blocks.

    Returns:
        A list of dictionaries, each containing 'search', 'replace', and 'similarity' keys.
    """
    blocks = []
    pattern = r"<SEARCH>\s*(.*?)\s*</SEARCH>\s*<REPLACE>\s*(.*?)\s*</REPLACE>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        similarity = 1.0  # Default to exact match

        if use_fuzzy and search not in response_text:
            # Extract possible search targets from the response text
            possible_search_targets = re.findall(
                r"<SEARCH>\s*(.*?)\s*</SEARCH>", response_text, re.DOTALL
            )
            possible_search_targets = [
                target.strip() for target in possible_search_targets
            ]

            best_match = difflib.get_close_matches(
                search, possible_search_targets, n=1, cutoff=0.6
            )
            if best_match:
                similarity = difflib.SequenceMatcher(
                    None, search, best_match[0]
                ).ratio()
            else:
                similarity = 0.0

        blocks.append({"search": search, "replace": replace, "similarity": similarity})

    return blocks


async def apply_edits(
    console: Console,
    file_path,
    edit_instructions,
    original_content,
    use_fuzzy_search: bool = True,
):
    changes_made = False
    edited_content = original_content
    total_edits = len(edit_instructions)
    failed_edits = []
    console_output = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        edit_task = progress.add_task("[cyan]Applying edits...", total=total_edits)

        for i, edit in enumerate(edit_instructions, 1):
            search_content = edit["search"].strip()
            replace_content = edit["replace"].strip()
            similarity = edit["similarity"]

            # Use regex to find the content, ignoring leading/trailing whitespace
            pattern = re.compile(re.escape(search_content), re.DOTALL)
            match = pattern.search(edited_content)

            if match or (use_fuzzy_search and similarity >= 0.8):
                if not match:
                    # If using fuzzy search and no exact match, find the best match
                    best_match = difflib.get_close_matches(
                        search_content, [edited_content], n=1, cutoff=0.6
                    )
                    if best_match:
                        match = re.search(re.escape(best_match[0]), edited_content)

                if match:
                    # Replace the content using re.sub for more robust replacement
                    replace_content_cleaned = re.sub(
                        r"</?SEARCH>|</?REPLACE>", "", replace_content
                    )
                    edited_content = pattern.sub(
                        replace_content_cleaned, edited_content, count=1
                    )
                    changes_made = True

                    # Display the diff for this edit
                    diff_result = generate_diff(
                        search_content, replace_content, file_path
                    )
                    console.print(
                        Panel(
                            diff_result,
                            title=f"Changes in {file_path} ({i}/{total_edits}) - Similarity: {similarity:.2f}",
                            style="cyan",
                        )
                    )
                    console_output.append(
                        f"Edit {i}/{total_edits} applied successfully"
                    )
                else:
                    message = f"Edit {i}/{total_edits} not applied: content not found (Similarity: {similarity:.2f})"
                    console_output.append(message)
                    console.print(Panel(message, style="yellow"))
                    failed_edits.append(f"Edit {i}: {search_content}")
            else:
                message = f"Edit {i}/{total_edits} not applied: content not found (Similarity: {similarity:.2f})"
                console_output.append(message)
                console.print(Panel(message, style="yellow"))
                failed_edits.append(f"Edit {i}: {search_content}")

            progress.update(edit_task, advance=1)

    if not changes_made:
        message = "No changes were applied. The file content already matches the desired state."
        console_output.append(message)
        console.print(Panel(message, style="green"))
    else:
        # Write the changes to the file
        with open(file_path, "w") as file:
            file.write(edited_content)
        message = f"Changes have been written to {file_path}"
        console_output.append(message)
        console.print(Panel(message, style="green"))

    return edited_content, changes_made, failed_edits, "\n".join(console_output)


class Context:
    def __init__(
        self,
        anthropic_api_key: str,
        tavily_api_key: str,
        console: Console | None = None,
    ):
        self.config = Config()
        self.conversation_history = []
        self.console = console or Console()
        self.file_contents = {}
        self.code_editor_memory = []
        self.code_editor_files = set()
        self.automode = False
        self.use_tts = False
        self.tts_enabled = False
        self.voice_mode = False
        self.voice = None
        self.proc = ProcessHandler()
        self.token_stats = {
            "mainmodel": TokenStats(),
            "tool_checker": TokenStats(),
            "code_editor": TokenStats(),
            "code_execution": TokenStats(),
        }
        self.use_fuzzy_search = True
        self.client = Anthropic(api_key=anthropic_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)

    @classmethod
    def build_from_env(cls, console: Console | None = None):
        anthropic_api_key = get_env_checked("ANTHROPIC_API_KEY")
        tavily_api_key = get_env_checked("TAVILY_API_KEY")
        return cls(anthropic_api_key, tavily_api_key, console=console)

    def print(self, text: Any):
        self.console.print(text)

    def tavily_search(self, query: str) -> str:
        try:
            return self.tavily.qna_search(query=query, search_depth="advanced")
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def enter_voice_mode(self):
        if self.voice is None:
            log.info("Rebuilding voice mode handler")
            self.voice = VoiceMode.build_from_env()
        self.voice_mode = True

    def toggle_tts(self, enabled: bool):
        self.tts_mode = enabled

    def reset(self):
        self.code_editor_files = set()
        self.code_editor_memory = []
        self.conversation_history = []
        self.file_contents = {}
        self.token_stats = {
            "mainmodel": TokenStats(),
            "tool_checker": TokenStats(),
            "code_editor": TokenStats(),
            "code_execution": TokenStats(),
        }
        self.console.print(
            Panel(
                "Conversation history, token counts, file contents, code editor memory, and code editor files have been reset.",
                title="Reset",
                style="bold green",
            )
        )
        self.display_token_usage()

    def reset_code_editor_memory(self):
        self.code_editor_memory = []
        self.console.print(
            Panel(
                "Code editor memory has been reset.",
                title="Reset",
                style="bold green",
            )
        )

    def update_system_prompt(
        self,
        current_iteration: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> str:
        chain_of_thought_prompt = """
        Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

        Do not reflect on the quality of the returned search results in your response.

        IMPORTANT: Before using the read_multiple_files tool, always check if the files you need are already in your context (system prompt).
        If the file contents are already available to you, use that information directly instead of calling the read_multiple_files tool.
        Only use the read_multiple_files tool for files that are not already in your context.
        When instructing to read a file, always use the full file path.
        """

        files_in_context = "\n".join(self.file_contents.keys())
        file_contents_prompt = f"\n\nFiles already in your context:\n{files_in_context}\n\nFile Contents:\n"
        for path, content in self.file_contents.items():
            file_contents_prompt += f"\n--- {path} ---\n{content}\n"

        if self.automode:
            iteration_info = ""
            if current_iteration is not None and max_iterations is not None:
                iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
            return (
                BASE_SYSTEM_PROMPT
                + file_contents_prompt
                + "\n\n"
                + AUTOMODE_SYSTEM_PROMPT.format(iteration_info=iteration_info)
                + "\n\n"
                + chain_of_thought_prompt
            )
        else:
            return (
                BASE_SYSTEM_PROMPT
                + file_contents_prompt
                + "\n\n"
                + chain_of_thought_prompt
            )

    def create_files(self, files):
        results = []

        # Handle different input types
        if isinstance(files, str):
            # If a string is passed, assume it's a single file path
            files = [{"path": files, "content": ""}]
        elif isinstance(files, dict):
            # If a single dictionary is passed, wrap it in a list
            files = [files]
        elif not isinstance(files, list):
            return "Error: Invalid input type for create_files. Expected string, dict, or list."

        for file in files:
            try:
                if not isinstance(file, dict):
                    results.append(f"Error: Invalid file specification: {file}")
                    continue

                path = file.get("path")
                content = file.get("content", "")

                if path is None:
                    results.append("Error: Missing 'path' for file")
                    continue

                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                with open(path, "w") as f:
                    f.write(content)

                self.file_contents[path] = content
                results.append(f"File created and added to system prompt: {path}")
            except Exception as e:
                results.append(f"Error creating file: {str(e)}")

        return "\n".join(results)

    async def generate_edit_instructions(
        self, file_path, file_content, instructions, project_context, full_file_contents
    ):
        try:
            # Prepare memory context (maintains some context between calls)
            memory_context = "\n".join(
                [
                    f"Memory {i+1}:\n{mem}"
                    for i, mem in enumerate(self.code_editor_memory)
                ]
            )

            # Prepare full file contents context, excluding the file being edited if it's already in code_editor_files
            full_file_contents_context = "\n\n".join(
                [
                    f"--- {path} ---\n{content}"
                    for path, content in full_file_contents.items()
                    if path != file_path or path not in self.code_editor_files
                ]
            )

            system_prompt = f"""
            You are an expert coding assistant specializing in web development (CSS, JavaScript, React, Tailwind, Node.JS, Hugo/Markdown). Review the following information carefully:
        
            1. File Content:
            {file_content}
        
            2. Edit Instructions:
            {instructions}
        
            3. Project Context:
            {project_context}
        
            4. Previous Edit Memory:
            {memory_context}
        
            5. Full Project Files Context:
            {full_file_contents_context}
        
            Follow this process to generate edit instructions:
        
            1. <CODE_REVIEW>
            Analyze the existing code thoroughly. Describe how it works, identifying key components, 
            dependencies, and potential issues. Consider the broader project context and previous edits.
            </CODE_REVIEW>
        
            2. <PLANNING>
            Construct a plan to implement the requested changes. Consider:
            - How to avoid code duplication (DRY principle)
            - Balance between maintenance and flexibility
            - Relevant frameworks or libraries
            - Security implications
            - Performance impacts
            Outline discrete changes and suggest small tests for each stage.
            </PLANNING>
        
            3. Finally, generate SEARCH/REPLACE blocks for each necessary change:
            - Use enough context to uniquely identify the code to be changed
            - Maintain correct indentation and formatting
            - Focus on specific, targeted changes
            - Ensure consistency with project context and previous edits
        
            USE THIS FORMAT FOR CHANGES:
        
            <SEARCH>
            Code to be replaced (with sufficient context)
            </SEARCH>
            <REPLACE>
            New code to insert
            </REPLACE>
        
            IMPORTANT: ONLY RETURN CODE INSIDE THE <SEARCH> AND <REPLACE> TAGS. DO NOT INCLUDE ANY OTHER TEXT, COMMENTS, or Explanations. FOR EXAMPLE:
        
            <SEARCH>
            def old_function():
                pass
            </SEARCH>
            <REPLACE>
            def new_function():
                print("New Functionality")
            </REPLACE>
            """

            response: PromptCachingBetaMessage = self.client.beta.prompt_caching.messages.create(
                model=self.config.model.code_editor,
                max_tokens=8000,
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
                        "content": "Generate SEARCH/REPLACE blocks for the necessary changes.",
                    }
                ],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

            # Update token usage for code editor
            self.token_stats["code_editor"].update_from_response(response.usage)

            ai_response_text = response.content[0].text  # Extract the text

            # If ai_response_text is a list, handle it
            if isinstance(ai_response_text, list):
                ai_response_text = " ".join(
                    item["text"]
                    if isinstance(item, dict) and "text" in item
                    else str(item)
                    for item in ai_response_text
                )
            elif not isinstance(ai_response_text, str):
                ai_response_text = str(ai_response_text)

            # Validate AI response
            try:
                if not validate_ai_response(ai_response_text):
                    raise ValueError(
                        "AI response does not contain valid SEARCH/REPLACE blocks"
                    )
            except ValueError as ve:
                logging.error(f"Validation failed: {ve}")
                return []  # Return empty list to indicate failure

            # Parse the response to extract SEARCH/REPLACE blocks
            edit_instructions = parse_search_replace_blocks(ai_response_text)

            if not edit_instructions:
                raise ValueError("No valid edit instructions were generated")

            # Update code editor memory
            self.code_editor_memory.append(
                f"Edit Instructions for {file_path}:\n{ai_response_text}"
            )

            # Add the file to code_editor_files set
            self.code_editor_files.add(file_path)

            return edit_instructions

        except Exception as e:
            self.console.print(
                f"Error in generating edit instructions: {str(e)}", style="bold red"
            )
            log.error(
                "Error in generating edit instructions: %s", str(e), exc_info=True
            )
            return []  # Return empty list if any exception occurs

    async def edit_and_apply_multiple(
        self,
        files,
        project_context,
        is_automode: bool = False,
    ):
        results = []
        console_outputs = []

        logging.debug(f"edit_and_apply_multiple called with files: {files}")
        logging.debug(f"Project context: {project_context}")

        try:
            files = validate_files_structure(files)
        except ValueError as ve:
            logging.error(f"Validation error: {ve}")
            return [], f"Error: {ve}"

        logging.info(f"Starting edit_and_apply_multiple with {len(files)} file(s)")

        for file in files:
            path = file["path"]
            instructions = file["instructions"]
            logging.info(f"Processing file: {path}")
            try:
                original_content = self.file_contents.get(path, "")
                if not original_content:
                    logging.info(f"Reading content for file: {path}")
                    with open(path, "r") as f:
                        original_content = f.read()
                    self.file_contents[path] = original_content

                logging.info(f"Generating edit instructions for file: {path}")
                edit_instructions = await self.generate_edit_instructions(
                    path,
                    original_content,
                    instructions,
                    project_context,
                    self.file_contents,
                )

                logging.debug(f"AI response for {path}: {edit_instructions}")

                if not isinstance(edit_instructions, list) or not all(
                    isinstance(item, dict) for item in edit_instructions
                ):
                    raise ValueError(
                        "Invalid edit_instructions format. Expected a list of dictionaries."
                    )

                if edit_instructions:
                    self.console.print(
                        Panel(
                            f"File: {path}\nThe following SEARCH/REPLACE blocks have been generated:",
                            title="Edit Instructions",
                            style="cyan",
                        )
                    )
                    for i, block in enumerate(edit_instructions, 1):
                        self.console.print(f"Block {i}:")
                        self.console.print(
                            Panel(
                                f"SEARCH:\n{block['search']}\n\nREPLACE:\n{block['replace']}\nSimilarity: {block['similarity']:.2f}",
                                expand=False,
                            )
                        )

                    logging.info(f"Applying edits to file: {path}")
                    (
                        edited_content,
                        changes_made,
                        failed_edits,
                        console_output,
                    ) = await apply_edits(
                        self.console,
                        path,
                        edit_instructions,
                        original_content,
                        use_fuzzy_search=self.use_fuzzy_search,
                    )

                    console_outputs.append(console_output)

                    if changes_made:
                        self.file_contents[path] = edited_content
                        self.console.print(
                            Panel(
                                f"File contents updated in system prompt: {path}",
                                style="green",
                            )
                        )
                        logging.info(f"Changes applied to file: {path}")

                        if failed_edits:
                            logging.warning(f"Some edits failed for file: {path}")
                            logging.debug(f"Failed edits for {path}: {failed_edits}")
                            results.append(
                                {
                                    "path": path,
                                    "status": "partial_success",
                                    "message": f"Some changes applied to {path}, but some edits failed.",
                                    "failed_edits": failed_edits,
                                    "edited_content": edited_content,
                                }
                            )
                        else:
                            results.append(
                                {
                                    "path": path,
                                    "status": "success",
                                    "message": f"All changes successfully applied to {path}",
                                    "edited_content": edited_content,
                                }
                            )
                    else:
                        logging.warning(f"No changes applied to file: {path}")
                        results.append(
                            {
                                "path": path,
                                "status": "no_changes",
                                "message": f"No changes could be applied to {path}. Please review the edit instructions and try again.",
                            }
                        )
                else:
                    logging.warning(f"No edit instructions generated for file: {path}")
                    results.append(
                        {
                            "path": path,
                            "status": "no_instructions",
                            "message": f"No edit instructions generated for {path}",
                        }
                    )
            except Exception as e:
                logging.error(f"Error editing/applying to file {path}: {str(e)}")
                logging.exception("Full traceback:")
                error_message = f"Error editing/applying to file {path}: {str(e)}"
                results.append(
                    {"path": path, "status": "error", "message": error_message}
                )
                console_outputs.append(error_message)

        logging.info("Completed edit_and_apply_multiple")
        logging.debug(f"Results: {results}")
        return results, "\n".join(console_outputs)

    def read_multiple_files(self, paths, recursive=False):
        results = []

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            try:
                abs_path = os.path.abspath(path)
                if os.path.isdir(abs_path):
                    if recursive:
                        file_paths = glob.glob(
                            os.path.join(abs_path, "**", "*"), recursive=True
                        )
                    else:
                        file_paths = glob.glob(os.path.join(abs_path, "*"))
                    file_paths = [f for f in file_paths if os.path.isfile(f)]
                else:
                    file_paths = glob.glob(abs_path, recursive=recursive)

                for file_path in file_paths:
                    abs_file_path = os.path.abspath(file_path)
                    if os.path.isfile(abs_file_path):
                        if abs_file_path not in self.file_contents:
                            with open(abs_file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            self.file_contents[abs_file_path] = content
                            results.append(
                                f"File '{abs_file_path}' has been read and stored in the system prompt."
                            )
                        else:
                            results.append(
                                f"File '{abs_file_path}' is already in the system prompt. No need to read again."
                            )
                    else:
                        results.append(f"Skipped '{abs_file_path}': Not a file.")
            except Exception as e:
                results.append(f"Error reading path '{path}': {str(e)}")

        return "\n".join(results)

    def display_token_usage(self):
        from rich.box import ROUNDED
        from rich.table import Table

        table = Table(box=ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Input", style="magenta")
        table.add_column("Output", style="magenta")
        table.add_column("Cache Write", style="blue")
        table.add_column("Cache Read", style="blue")
        table.add_column("Total", style="green")
        table.add_column(
            f"% of Context ({self.config.max_context_tokens:,})", style="yellow"
        )
        table.add_column("Cost ($)", style="red")

        model_costs = {
            "Main Model": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
                "has_context": True,
            },
            "Tool Checker": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
                "has_context": False,
            },
            "Code Editor": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
                "has_context": True,
            },
            "Code Execution": {
                "input": 3.00,
                "output": 15.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
                "has_context": False,
            },
        }

        total_input = 0
        total_output = 0
        total_cache_write = 0
        total_cache_read = 0
        total_cost = 0
        total_context_tokens = 0

        for model, tokens in [
            ("Main Model", self.token_stats["mainmodel"]),
            ("Tool Checker", self.token_stats["tool_checker"]),
            ("Code Editor", self.token_stats["code_editor"]),
            ("Code Execution", self.token_stats["code_execution"]),
        ]:
            input_tokens = tokens.input
            output_tokens = tokens.output
            cache_write_tokens = tokens.cache_write
            cache_read_tokens = tokens.cache_read
            total_tokens = (
                input_tokens + output_tokens + cache_write_tokens + cache_read_tokens
            )

            total_input += input_tokens
            total_output += output_tokens
            total_cache_write += cache_write_tokens
            total_cache_read += cache_read_tokens

            input_cost = (input_tokens / 1_000_000) * model_costs[model]["input"]
            output_cost = (output_tokens / 1_000_000) * model_costs[model]["output"]
            cache_write_cost = (cache_write_tokens / 1_000_000) * model_costs[model][
                "cache_write"
            ]
            cache_read_cost = (cache_read_tokens / 1_000_000) * model_costs[model][
                "cache_read"
            ]
            model_cost = input_cost + output_cost + cache_write_cost + cache_read_cost
            total_cost += model_cost

            if model_costs[model]["has_context"]:
                total_context_tokens += total_tokens
                percentage = (total_tokens / self.config.max_context_tokens) * 100
            else:
                percentage = 0

            table.add_row(
                model,
                f"{input_tokens:,}",
                f"{output_tokens:,}",
                f"{cache_write_tokens:,}",
                f"{cache_read_tokens:,}",
                f"{total_tokens:,}",
                f"{percentage:.2f}%"
                if model_costs[model]["has_context"]
                else "Doesn't save context",
                f"${model_cost:.3f}",
            )

        grand_total = total_input + total_output + total_cache_write + total_cache_read
        total_percentage = (total_context_tokens / self.config.max_context_tokens) * 100

        table.add_row(
            "Total",
            f"{total_input:,}",
            f"{total_output:,}",
            f"{total_cache_write:,}",
            f"{total_cache_read:,}",
            f"{grand_total:,}",
            f"{total_percentage:.2f}%",
            f"${total_cost:.3f}",
            style="bold",
        )

        self.console.print(table)

    def reset_conversation(self):
        self.conversation_history = []
        self.token_stats = {
            "mainmodel": TokenStats(),
            "tool_checker": TokenStats(),
            "code_editor": TokenStats(),
            "code_execution": TokenStats(),
        }
        self.file_contents = {}
        self.code_editor_files = set()
        self.reset_code_editor_memory()
        self.print(
            Panel(
                "Conversation history, token counts, file contents, code editor memory, and code editor files have been reset.",
                title="Reset",
                style="bold green",
            )
        )
        self.display_token_usage()
