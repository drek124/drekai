import inspect
from types import NoneType
from typing import Literal, Any

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

class ToolParameter:
    def __init__(self, name: str, description: str, *, required: bool = True,
                  type: Any | Literal['string', 'number', 'integer', 'boolean',
                                'object',  'array', 'null'] = str):
        """## Tool Parameter
        ## Parameters
        - name: The name of the parameter; must match the present parameter in the callback.
        - description: The description of this parameter that will be given to the LLM.
        - type: The instance/object type of this parameter; Supported data types:
        ```python 
        str, int, float, bool, dict, list, None
        ```
        """
        self.name: str = name
        self.description: str = description
        self.required: bool = required
        if isinstance(type, str):
            self.type: str = type
        else:
            if type not in TOOL_PARAM_CLASSES:
                raise ValueError(f"Unknown tool parameter type \"{type}\"")
            self.type: str = TOOL_PARAM_CLASSES[type]

class Tool:
    def __init__(
            self, name: str, params: list[ToolParameter] = [], *,
            callback = None, sandbox_params: list[str] = []
                  ):
        """## Create an AI Tool
        ### Parameters
        - name: The name of the tool
        - params: The parameters of the tool
        - callback: The callback of the tool; the description of the tool is decided by the callback's docs
        - sandbox_params: The parameters hidden from the model and passed into the tool parameters.
        By using sandboxed parameters, it makes it impossible for the AI to manage data it isn't supposed to manage.
        Built-in sandbox parameters: 
          - `_tc_index`: The index of the tool call
        ### Example
        *Creating a tool that fetches a user's friends*      
        ```python
        from DrekAI import Chatbot
        from DrekAI.tools import Tool, ToolParameter

        assistant = Chatbot("Assistant", "You are a helpful assistant")
        chat = assistant.start_chat()

        async def get_friends(user, limit: int = 100) -> str:
            "Fetch the user's friends list" # The description passed to the LLM
            result: str | Any = user.get_friends(limit=limit)
            return "Friends list:" + result # The data returned to the LLM
        tools = [
            Tool(
                "get_friends", [ToolParameter("limit", "The limit of the fetched users", required=False, type=int)],
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
        
        ```"""
        self.name: str = name
        self.params: list[ToolParameter] = params
        self.sandbox_params: list[str] = sandbox_params
        if callback:
            self.callback = callback
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
                'required': [p.name for p in self.params if p.required]

            }
        }
        for tool in self.params:
            self.raw['function']['parameters']['properties'][tool.name] = {'type': tool.type, 'description': tool.description}

        

    async def callback(self, **kwargs) -> str:
        raise NotImplementedError(f"Tool callback for \"{self.name}\" not implemented")
