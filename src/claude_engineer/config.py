""" """

from pydantic import BaseModel


class ModelConfig(BaseModel):
    main: str = "claude-3-5-sonnet-20241022"
    tool_checker: str = "claude-3-5-sonnet-20241022"
    code_editor: str = "claude-3-5-sonnet-20241022"
    code_execution: str = "claude-3-5-sonnet-20241022"

    @classmethod
    def new_set_all(cls, model: str) -> "ModelConfig":
        return cls(
            main=model,
            tool_checker=model,
            code_editor=model,
            code_execution=model,
        )


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    max_context_tokens: int = 200000
    max_continuation_iterations: int = 25
    continuation_exit_phrase: str = "AUTOMODE_COMPLETE"
