import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

from dotenv import load_dotenv

load_dotenv() 


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model_id = os.getenv("MODEL_ID", "openai/gpt-4o-mini")
        print(f"\nUsing model ID: {self.model_id}")
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("API_KEY"),
        )
        print(f"\nUsing base URL: {self.client.base_url}")
        print(f"\nUsing API key: {self.client.api_key}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py)
        """
        is_python = server_script_path.endswith('.py')
        if not (is_python):
            raise ValueError("Server script must be a .py file")
        
        # Get the script path without the file name
        script_path = server_script_path.rsplit('/', 1)[0]

        # Get the file name
        file_name = server_script_path.rsplit('/', 1)[-1]
            
        command = "uv"
        server_params = StdioServerParameters(
            command=command,
            args=[
                "--directory",
                script_path,
                "run",
                "python",
                file_name
            ],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        tools = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema,
                ),
            )
            for tool in (await self.session.list_tools()).tools
        ]

        # Initial Claude API call
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=1000,
            messages=messages,
            tools=tools,
        ).choices[0].message

        # Process response and handle tool calls
        final_text = []

        if response.content:
            final_text.append(response.content)
        else:
            for tool_call in response.tool_calls:
                messages.append(response)

                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                # print(f"\nCalling tool {tool_name} with args {tool_args}")

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)

                # print(f"\nTool {tool_name} result: {result}")

                # Continue conversation with tool results
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps([content.model_dump() for content in result.content]),
                })

                # print(f"\nMessages after tool call: {messages}")

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=1000,
                    messages=messages,
                    tools=result.content[0].text,
                ).choices[0].message

                # print(f"\nResponse after tool call: {response}")

                final_text.append(response.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():

    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
