import inspect
from inspect import _empty
from types import NoneType
from typing import Literal, Any

MISSING = object()

TOOL_PARAM_CLASSES = {
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
    dict: "object",
    list: "array",
    None: "null",
    NoneType: "null"
}


class Tool:
    def __init__(
            self, name: str = MISSING, *,
            callback = MISSING, sandbox_params: list[str] = []
                  ):
        '''## Create an AI Tool
        ### Parameters
        - name: The name of the tool; if it's missing, then the name of the callback function will be used.
        - callback: The callback of the tool; the description of the tool is decided by the callback's docs
        - sandbox_params: The parameters flagged to be hidden from the model and passed into the tool parameters.
        By using sandboxed parameters, it makes it impossible for the AI to manage data it isn't supposed to manage.
        Built-in sandbox parameters: 
          - `_tc_index`: The index of the tool call
        ### Example
        *Creating a tool that fetches a user's friends*      
        ```python
        from drekai import Chatbot
        from drekai.tools import Tool, ToolParameter

        assistant = Chatbot("Assistant", "You are a helpful assistant")
        chat = assistant.start_chat()

        async def get_friends(user, limit: int = 100) -> str:
            # 👇 The description passed to the LLM 
            """Fetch the user's friends list
            ## Parameters
            - limit: The limit of fetched users. This is used to reduce load on the API."""
            result: str | Any = user.get_friends(limit=limit)
            return "Friends list:" + result # The data returned to the LLM
        tools = [
            Tool(
                "get_friends",
                callback=get_friends,
                sandbox_params=["user"] # Accepted sandboxed parameters
                )
        ]

        user = ...
        async def main():
            response = await chat.generate_reply("Who is in my friend's list?", 
            tools=tools,
            sandbox_params={"user": user} # Impossible for the LLM to fetch another user's friends
            )
            print(response.choices[0].message.content)
        
        ```'''
        if callback is MISSING:
            raise ValueError("Callback is required for tools")
        self.name: str = name if name is not MISSING else callback.__name__
        self.callback = callback
        self.params: dict = inspect.signature(callback).parameters
        self.sandbox_params: list[str] = sandbox_params
        self.description: str = inspect.getdoc(self.callback)
        self.raw: dict = {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': {}
                },
                'required': []

            }
        }
        for pname, param in self.params.items():
            if pname in self.sandbox_params:
                continue

            ptype = param.annotation
            if ptype is _empty:
                raise ValueError(f"No parameter type present for parameter \"{pname}\".")
            if ptype not in TOOL_PARAM_CLASSES:
                raise ValueError(f"Unknown parameter type \"{ptype}\" for \"{pname}\"")
            
            required: bool = param.default is _empty

            if required:
                self.raw['function']['required'].append(pname)

            self.raw['function']['parameters']['properties'][param.name] = {'type': TOOL_PARAM_CLASSES[ptype]}



        

    async def callback(self, **kwargs) -> str:
        raise NotImplementedError(f"Tool callback for \"{self.name}\" not implemented")
