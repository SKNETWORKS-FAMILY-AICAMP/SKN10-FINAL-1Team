from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv()
async def main():
    client = MultiServerMCPClient(
        {
            "context7": {
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"],
                "env": {
                    "DEFAULT_MINIMUM_TOKENS": "6000"
                }
            }
        }
    )
    tools = await client.get_tools()
    
    agent = create_react_agent(
        "anthropic:claude-3-7-sonnet-latest",
        tools
    )
    
    # Add whatever you want to do with the agent here
    return agent

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
    h_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "w"}]}
)
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)

print(math_response)
print(weather_response)