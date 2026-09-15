class ChatSettings:
    def __init__(self, *, max_tokens: int = None, show_tool_error_type: bool = True):
        """## DrekAI Chat Settings
        ### Parameters
        - max_tokens: The maximum input + output completion tokens allowed per response
        - show_tool_error_type: Shows the LLM the exception that occured during the tool's execution; if `False`, the LLM recieves "TOOL ERROR". If `True`, the LLM recieves the exception message (Not traceback)."""
        self.max_tokens = max_tokens
        self.show_tool_error_type = show_tool_error_type