from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import (
    PromptCachingBetaMessage,
)
from anthropic.types.beta.prompt_caching.prompt_caching_beta_usage import (
    PromptCachingBetaUsage,
)
from anthropic.types.usage import Usage


class TokenStats:
    def __init__(self):
        self.input: int = 0
        self.output: int = 0
        self.cache_write: int = 0
        self.cache_read: int = 0

    def total(self):
        return self.input + self.output + self.cache_write + self.cache_read

    def reset(self):
        self.input = 0
        self.output = 0
        self.cache_write = 0
        self.cache_read = 0

    def update_from_response(self, usage: PromptCachingBetaUsage | Usage):
        self.input += usage.input_tokens
        self.output += usage.output_tokens
        if isinstance(usage, PromptCachingBetaUsage):
            self.cache_write += usage.cache_creation_input_tokens or 0
            self.cache_read += usage.cache_read_input_tokens or 0
