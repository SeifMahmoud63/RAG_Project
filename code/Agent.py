from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from config import get_llm
from config import get_tools
from retriever import advanced_retrieve 

def build_agent(vector, chunks):
    llm = get_llm()
    tools = get_tools()
    llm_with_bind = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        last_input = messages[-1].content

        results = advanced_retrieve(vector, chunks, last_input)
        if results:
            context_text = "\n\n".join([doc.page_content for doc in results])
            response = llm.invoke(f"Context:\n{context_text}\n\nQuestion: {last_input}")
        else:
            response = llm_with_bind.invoke(messages)

        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()