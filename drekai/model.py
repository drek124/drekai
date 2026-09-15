from typing import Literal
from . import Chat
from .settings import ChatSettings
from openai import AsyncOpenAI, HttpxBinaryResponseContent
MISSING = object()

class Model:
    def __init__(self, model_id: str, base_url: str, *, api_key = MISSING):
        """Create a new root model
        - id: The ID identified by the API endpoint, example: `drek-v6.7-pro`
        - base_url: The URL of the OpenAI-supported endpoint
        - api_key: The global API key for all child chatbots."""
        self.id: str = model_id
        self.base_url: str = base_url
        if 'gemini' in self.id:
            self.tool_compatability: str | None = 'gemini'
        else:
            self.tool_compatability: str = None
        if api_key is not MISSING:
            self.api_key = api_key  
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.api_key = None
            self.client: AsyncOpenAI = None


class Chatbot:
    def __init__(
            self, name: str, system_prompt: str, parent_model: Model, *,
            api_key = MISSING, temperature: float = 1.0, thinking: bool = True, reasoning_effort: Literal['low', 'medium', 'high', 'max'] = 'low',
            id = None
            ):
        """Create your new chatbot
        ## Parameters
        - name: The name of the chatbot; invisible to the LLM
        - system_prompt: The system prompt for the chatbot
        - parent_model: The root model that will be utilized
        - temperature: The temperature represents the creativity/charm level of the model.
        - api_key: Your authorized API key to the endpoint; if not present, `parent_model.api_key` will be utilized instead.
        - thinking: The ability for the LLM to think
        - reasoning_effort: When `thinking=True`, specify the effort the LLM would utilize to think
        - id: Use it when needed for your own backend development"""
        self.name: str = name
        self.id = id if id is not None else self.name.lower().strip().replace(' ', '-')
        self.parent_model: Model = parent_model
        self.system_prompt = system_prompt
        self.temperature: float = temperature
        self.reasoning_effort: str = reasoning_effort
        self.client = self.parent_model.client or AsyncOpenAI(api_key=api_key, base_url=parent_model.base_url)
        self.thinking: bool = thinking
    
    async def generate_text(self, prompt: str, role: str = 'user', **kwargs) -> HttpxBinaryResponseContent:
        """Generate a text response."""
        chat = self.start_chat()
        return await chat.generate_reply(prompt, role, **kwargs)

    def start_chat(self, *, options: ChatSettings = None) -> Chat:
        "Start a new chat with the chatbot"
        return Chat(self, options=options)
       