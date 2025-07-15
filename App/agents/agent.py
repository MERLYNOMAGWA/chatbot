from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from App.langchain.tools import MenuSearchTool
from App.llm.loader import load_llm

def create_agent():
    print("Loading LLM.")
    llm = load_llm()

    print("nitializing memory.")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=True
    )

    print("Loading tool.")
    tool_instance = MenuSearchTool()
    tools = [
        Tool(
            name=tool_instance.name,
            func=tool_instance._run, 
            description=tool_instance.description
        )
    ]

    print("Initializing agent...")
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("Agent loaded successfully.")
    return agent