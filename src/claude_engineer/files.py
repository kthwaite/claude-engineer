import os
import logging

log = logging.getLogger(__name__)


class FileHandler:
    def __init__(self):
        self.file_contents = {}

    def reset(self):
        self.file_contents = {}

    def __getitem__(self, path: str) -> str:
        return self.file_contents.get(path, "")

    def __setitem__(self, path: str, content: str):
        self.file_contents[path] = content

    def __delitem__(self, path: str):
        del self.file_contents[path]

    def __contains__(self, path: str) -> bool:
        return path in self.file_contents

    def get(self, path: str, default: str = "") -> str:
        return self.file_contents.get(path, default)

    def get_files_in_context(self) -> str:
        return "\n".join(self.file_contents.keys())

    def get_file_contents(self) -> str:
        output = ""
        for path, content in self.file_contents.items():
            output += f"\n--- {path} ---\n{content}\n"
        return output

    def file_contents_prompt(self) -> str:
        file_contents_prompt = f"\n\nFiles already in your context:\n{self.get_files_in_context()}\n\nFile Contents:\n"
        file_contents_prompt += self.get_file_contents()
        return file_contents_prompt

    def create_files(self, files: str | dict | list) -> str:
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
