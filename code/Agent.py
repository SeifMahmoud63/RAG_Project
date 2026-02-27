from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from config import get_llm
from tools import create_tools
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

def build_agent(vector, chunks):
    llm = get_llm()
    tools = create_tools(vector, chunks)
    llm_with_bind = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    memory = MemorySaver()
    def call_model(state: MessagesState):
        messages = state["messages"]
        
        
        sys_prompt = SystemMessage(content="""- You are a helpful assistant with access to tools.
- First, briefly reason about the user's request.
- Use the necessary tools to gather information, but avoid redundant calls.
- If you have enough information or the tools don't provide new data, provide your final answer.
- Efficiency is key: aim to solve the request in as few steps as possible.
- If a tool fails or returns no results, explain that to the user instead of looping.""")

        if not any(isinstance(m, SystemMessage) for m in messages):
            current_messages = [sys_prompt] + messages
        else:
            current_messages = messages

        response = llm_with_bind.invoke(current_messages)
        
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent") 

    return workflow.compile(checkpointer=memory)