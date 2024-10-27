""" """

import asyncio
import os
import signal
import sys
import venv
import logging

log = logging.getLogger(__name__)


def setup_virtual_environment() -> tuple[str, str]:
    venv_name = "code_execution_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    try:
        if not os.path.exists(venv_path):
            venv.create(venv_path, with_pip=True)

        # Activate the virtual environment
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")

        return venv_path, activate_script
    except Exception as e:
        log.error("Error setting up virtual environment: %s", str(e), exc_info=True)
        raise


class ProcessHandler:
    """ """

    def __init__(self):
        self.running_processes = {}

    def reset(self):
        self.running_processes = {}

    async def execute_code(self, code: str, timeout: int | float = 10):
        venv_path, activate_script = setup_virtual_environment()
        log.info("Virtual environment created at: %s", venv_path)

        # Input validation
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        if not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a number")

        # Generate a unique identifier for this process
        process_id = f"process_{len(self.running_processes)}"

        # Write the code to a temporary file
        try:
            with open(f"{process_id}.py", "w") as f:
                f.write(code)
        except IOError as e:
            return process_id, f"Error writing code to file: {str(e)}"

        # Prepare the command to run the code
        if sys.platform == "win32":
            command = f'"{activate_script}" && python3 {process_id}.py'
        else:
            command = f'source "{activate_script}" && python3 {process_id}.py'

        try:
            # Create a process to run the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
                preexec_fn=None if sys.platform == "win32" else os.setsid,
            )

            # Store the process in our global dictionary
            self.running_processes[process_id] = process

            try:
                # Wait for initial output or timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                stdout = stdout.decode()
                stderr = stderr.decode()
                return_code = process.returncode
            except asyncio.TimeoutError:
                # If we timeout, it means the process is still running
                stdout = "Process started and running in the background."
                stderr = ""
                return_code = "Running"

            execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
            return process_id, execution_result
        except Exception as e:
            return process_id, f"Error executing code: {str(e)}"
        finally:
            # Cleanup: remove the temporary file
            try:
                os.remove(f"{process_id}.py")
            except OSError:
                pass  # Ignore errors in removing the file

    def stop_process(self, process_id):
        if process_id in self.running_processes:
            process = self.running_processes[process_id]
            if sys.platform == "win32":
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            del self.running_processes[process_id]
            return f"Process {process_id} has been stopped."
        else:
            return f"No running process found with ID {process_id}."
