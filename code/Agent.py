from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from config import get_llm
from tools import create_tools
from langchain_core.messages import SystemMessage, ToolMessage

def build_agent(vector, chunks):
    llm = get_llm()
    tools = create_tools(vector, chunks)
    llm_with_bind = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        
        sys_prompt = SystemMessage(content="""You are a professional AI Assistant.
        - To answer technical questions, use 'Search_Local_Documents'.
        - For general info, use 'Tavily_Tool'.
        - IMPORTANT: Once you receive data from a tool, analyze it and give the FINAL ANSWER immediately. 
        - DO NOT call the same tool twice with the same query.""")

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

    return workflow.compile()