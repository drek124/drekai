import asyncio
import random
import string
import json
import traceback
import logging
from .tools import Tool
from .messaging import MessageItem
from .settings import ChatSettings
from openai import HttpxBinaryResponseContent
DEFAULT_OPTIONS = ChatSettings()

class Chat:
    def __init__(self, model, options):
        self.model = model
        self.messages: list[dict, dict] = [{
                    "role": "system",
                    "content": model.system_prompt
                }]
        self.total_tokens: int = 0
        self.options: ChatSettings = options or DEFAULT_OPTIONS
        self._stop_event: asyncio.Event | None = None
        self.id: str = (self.model.id + '-' + ''.join(random.choice(string.ascii_lowercase) for _ in range(15))).upper()



    def add_message_to_context(self, content: str, role: str = 'user') -> list[dict, dict]:
        """Adds a message to the messages context."""
        self.messages.append({
            "role": role,
            "content": str(content)
        })
        return self.messages

    async def generate_reply(
            self, prompt: str = None, prompt_role: str = 'user', *, thinking_callback = None, return_response_data: bool = True,
            tools: list[Tool] = None, sandbox_params = {}, stream_callback = None, options: ChatSettings = None,
            items: list[MessageItem] = [], 
    ) -> HttpxBinaryResponseContent:
        """## Response Generation
        ### Parameters
        - prompt: Add a prompt before generating
        - prompt_role: The role of the prompt; such as `'user'`, `'system'` & `'assistant'`
        - thinking_callback: The `async callback(thought: str)`; executed when a tool is called
        - return_response_data: Whether to directly return the AI's response or the full API response.
        - tools: The list of available tools
        - sandbox_params: The sandboxed parameters that are invisible to the model; if a tool call from this request requires a certain sandboxed parameter, it will be fetched from here. (e.g. `{'user_id': user.id}`)
        - stream_callback: `async callback(chunk: str)`; sets `stream` to `True`
        - items: The list of chat items to insert in the message (e.g. images)
        - options: The overwrite of the chat settings
        """
        
        contents = [] + [item.value for item in items]
        if prompt:
            contents.append({"type": "text", "text": str(prompt)})

        if contents:
            self.messages.append({
            "role": prompt_role,
            "content": contents
        })

        params = {}

        if not options:
            options = self.options
        
        if options.max_tokens:
            params['max_tokens'] = options.max_tokens
        if self.model.parent_model.tool_compatability == 'gemini':
           
            for tool in tools:
                if 'required' not in tool.raw.get('function', {}): 
                    continue
                preq = tool.raw['function']['required']
                tool.raw['function']['parameters']['required'] = preq
                tool.raw['function'].pop('required')


        if tools:
            params['tools'] = [tool.raw for tool in tools]

        if not self.model.thinking:
            params["extra_body"] = {
            "thinking": {"type": "disabled"}
        }

        stream = True if stream_callback else False
        

        r = await self.model.client.chat.completions.create(
        model=self.model.parent_model.id,
        messages=self.messages,
        stream=stream,
        reasoning_effort=self.model.reasoning_effort, 
        temperature=self.model.temperature, 
        **params
    )

        if stream_callback:
            full_content = ""
            reasoning_content = ""
            tool_calls_dict = {}
            self._stop_event = asyncio.Event()
            async for chunk in r:
                if self._stop_event.is_set():
                    await r.close()
                    full_content += '\n[GENERATION_STOPPED]'
                    assistant_msg = {"role": "assistant", "content": full_content}
                    self.messages.append(assistant_msg)
                    return assistant_msg if return_response_data else full_content
                # Track token usage if included in stream
                if chunk.usage:
                    self.total_tokens += chunk.usage.total_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 1. Accumulate text content & stream it
                if delta.content:
                    full_content += delta.content
                    await stream_callback(delta.content)

                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    if thinking_callback:
                        await thinking_callback(delta.reasoning_content)

                # 3. Assemble streamed tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = tc
                        else:
                            # Concatenate incoming argument chunks
                            if tc.function and tc.function.arguments:
                                tool_calls_dict[idx].function.arguments += tc.function.arguments

            # Convert assembled tool calls to list
            tool_calls = list(tool_calls_dict.values()) if tool_calls_dict else None
            text_response = full_content

            # Append assembled message to history
            assistant_msg = {"role": "assistant", "content": full_content or None}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

        else:
            # Non-streaming path
            self.messages.append(r.choices[0].message)
            self.total_tokens += r.usage.total_tokens
            text_response = r.choices[0].message.content
            tool_calls = r.choices[0].message.tool_calls
            reasoning_content = getattr(r.choices[0].message, 'reasoning_content', None)


        

        # Tool Execution Phase
        if tool_calls:
            if getattr(self.model.parent_model, 'tool_compatability', None) == 'gemini':
                # Convert Pydantic SDK object to dict if non-streaming
                if not isinstance(self.messages[-1], dict) and hasattr(self.messages[-1], "model_dump"):
                    self.messages[-1] = self.messages[-1].model_dump()

                last_msg = self.messages[-1]

                if isinstance(last_msg, dict) and "tool_calls" in last_msg:
                    for tc in last_msg["tool_calls"]:
                        # 1. Attach directly to tool call dict
                        tc["thought_signature"] = "skip_thought_signature_validator"
                        
                        # 2. Attach to provider_specific_fields / extra_content (OpenAI proxy target)
                        tc["extra_content"] = {"google": {"thought_signature": "skip_thought_signature_validator"}}
            for ii, tc in enumerate(tool_calls, start=1):
                tool = None
                sandbox_params['_tc_index'] = ii # Add tool call index incase a tool needs it
                for t in tools:
                    if t.name == tc.function.name:
                        tool = t
                        break

                args = json.loads(tc.function.arguments)
                for sp in tool.sandbox_params:
                    args[sp] = sandbox_params[sp]

                if reasoning_content and thinking_callback:
                    await thinking_callback(reasoning_content)
                try:
                    tool_response = await tool.callback(**args)
                except Exception as e:
                    tb = traceback.format_exc()
                    tool_response = f"ERROR: {e}" if options.show_tool_error_type else "TOOL ERROR"
                    logging.error(tb)
                last_tool = (ii == len(tool_calls))
                response = await self.tool_response(
                    tc.id, 
                    tool_response, 
                    stream_callback=stream_callback,
                    generate_response=last_tool, 
                    sandbox_params=sandbox_params, 
                    tools=tools
                )

                if not last_tool:
                    continue

            if return_response_data:
                return response
            
            return text_response

        return r if return_response_data else text_response


    async def stop_live_generation(self):
        "Stops the live generation"
        self._stop_event.set()
    
    def clear(self):
        """Wipes the chat; the system prompt doesn't get removed."""
        self.messages = []
        if self.model.system_prompt:
                self.messages.append({
                    "role": "system",
                    "content": self.model.system_prompt
                })

    def delete_first_message(self, role: str):
        """Deletes the first message in the context with the targeted role. System prompt cannot be deleted."""
        for msg in self.messages:
            if msg == self.messages[0]: continue
            if msg['role'] != role: continue
            self.messages.remove(msg)
            break

    async def tool_response(self, call_id: str, content, *, stream_callback, generate_response: bool = True, sandbox_params: dict = {}, tools = []):
        self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': str(content)})
        if not generate_response: return
        response = await self.generate_reply(return_response_data=True, tools=tools, sandbox_params=sandbox_params, stream_callback=stream_callback)
        return response
