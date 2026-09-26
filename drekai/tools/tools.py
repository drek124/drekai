import inspect
from inspect import _empty
from types import NoneType
from typing import get_args, get_origin, Union

MISSING = object()

TOOL_PARAM_CLASSES = {
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
    dict: "object",
    list: "array",
    None: "null",
    NoneType: "null",
    _empty: "string" # Default datatype
}

class SandboxParam:
    """Flag a parameter as sandboxed"""
    pass # Useless, just meant to be a flag nothing more

class Tool:
    def __init__(
            self, name: str = MISSING, *, description: str = MISSING,
            callback = MISSING
                  ):
        '''## Create an AI Tool
        ### Parameters
        - name: The name of the tool; if it's missing, then the name of the callback function will be used.
        - callback: The callback of the tool; the description of the tool is decided by the callback's docs
        ## Sandbox Parameters
        Sandbox Parameters are parameters flagged to be hidden from the model while also passed into the tool parameters.
        By using sandboxed parameters, it makes it impossible for the AI to manage data it isn't supposed to manage.
        Built-in sandbox parameters: 
          - `_tc_index`: The index of the tool call
        ### How to flag Sandboxed parameters?
        ```python
        from drekai import Chatbot
        from drekai.tools import Tool, SandboxParam
    
                                              
        async def email_user( # 👇 Flagged as sandboxed parameter, will be hidden from LLM
            client_id: int | SandboxParam, 
            message: str, # Required text parameter
            include_signature: bool = True # Optional boolean parameter
        ):
            """
            Send an email to the user!
            ## Parameters
            - message: The message content you want to send
            - include_signature: Whether to include the signature in the message or not
            """ # Description fed to the LLM
            result = await send_email(target_id=client_id, message=message, signature=include_signature)
            return result

        
        tools = [
            Tool(callback=email_user) # Name, description, and parameter data directly read from the callback
        ]

        client = User(...) # whatever is set up
        chatbot = Chatbot(...)
        chat = chatbot.start_chat()
        response = await chat.generate_reply(
            "Hey! Send me an email!", 
            tools=tools,
            sandbox_params={'client_id': client.id} # Injected when tool is called.
        )
        print(response.choices[0].message.content)
        ```  
        '''
        
        if callback is MISSING:
            raise ValueError("Callback is required for tools")
        self.name: str = name if name is not MISSING else callback.__name__
        self.callback = callback
        self.params: dict = inspect.signature(callback).parameters
        self.sandbox_params: list[str] = []
        self.description: str = inspect.getdoc(self.callback) if description is MISSING else description
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

            origin = get_origin(ptype)
            if origin is Union or origin is __import__("types").UnionType:
                types_list = get_args(ptype)
                if SandboxParam in types_list:
                    self.sandbox_params.append(pname)
                    continue # Ignored sandboxed parameter
                ptype = types_list[0]


            if ptype not in TOOL_PARAM_CLASSES:
                raise ValueError(f"Unknown parameter type \"{ptype}\" for \"{pname}\"")
            
            required: bool = param.default is _empty

            if required:
                self.raw['function']['required'].append(pname)

            self.raw['function']['parameters']['properties'][param.name] = {'type': TOOL_PARAM_CLASSES[ptype]}


    async def callback(self, **kwargs) -> str:
        raise NotImplementedError(f"Tool callback for \"{self.name}\" not implemented")


