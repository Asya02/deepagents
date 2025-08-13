from typing import Annotated, NotRequired, Sequence, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage
from langchain_gigachat import GigaChat
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]


def create_custom_react_agent(model, prompt, tools, state_schema=AgentState, additional_fields=None):
    """Create a custom react agent."""

    graph_builder = StateGraph(
        state_schema=state_schema or AgentState
    )

    llm_with_tools = model.bind_tools(tools, additional_fields=additional_fields)
    tool_node = ToolNode(tools=tools)

    def chatbot(state):
        """Chatbot that uses tools."""
        return {"messages": [llm_with_tools.invoke([("system", prompt)]+state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    agent = graph_builder.compile()

    return agent


if __name__ == "__main__":
    model = GigaChat(model="GigaChat-2-Max", verify_ssl_certs=False, profanity_check=False)
    prompt = "ты полезный ассистент"

    class think(BaseModel):
        """Use it for thinking."""
        thought: str = Field(..., description="Your thoughts")

    tools = [think]

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    state_schema = State

    agent = create_custom_react_agent(model, prompt, tools, state_schema)
    inputs = {"messages": [("user", "Сколько будет 39204872685291*6943520757")]}
    print(agent.invoke(inputs))
