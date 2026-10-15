"""LangGraph wiring: the ask box, and the timed proposal pass."""

import logging
from typing import Annotated

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .schema import TickOutput
from .tools import OPERATOR_TOOLS

log = logging.getLogger(__name__)


class AskState(TypedDict):
    messages: Annotated[list, add_messages]


class ProposeState(TypedDict):
    # One message: the table the proposer builds.
    messages: Annotated[list, add_messages]
    result: TickOutput | None


SYSTEM_PROMPT = """You answer questions from a truck dispatcher about their \
fleet, using the tools provided.

The operator runs a delivery fleet. Deliveries move through pending, \
assigned, in_transit, and end as completed, cancelled or failed. Trucks are \
available, on_delivery, maintenance or offline.

Answer from tool results only. If the data does not support an answer, say \
so rather than estimating. There are no ETAs or promised times in this \
system — if asked which deliveries are late, say that the data does not \
record a promised time, and offer what you do have: how long each has been \
waiting or in transit.

Trucks carry a driver_id, not a name. Resolve it with list_drivers rather \
than showing the dispatcher a uuid. A null driver_id only means the truck has \
no usual driver: drivers are assigned per delivery, so an available truck \
without one can still be dispatched. Don't call it a problem.

Be brief. A dispatcher is reading this mid-shift. Answer the question and \
stop — do not offer to do more, because each question is answered on its \
own and you will not remember this one."""


PROPOSE_PROMPT = """You suggest truck assignments to a dispatcher, who reviews \
each one before anything happens.

The message lists pending deliveries that have an available truck within \
2 km of the pickup, with a position under 15 min old. Distances are \
straight-line from the truck's last known position.

Rules:
- Use only references and plates from the table.
- At most one proposal per truck and one per delivery. Each must stand alone: \
the dispatcher may approve any of them and reject the rest. When two \
deliveries want the same truck, give it to the one that has waited longer.
- Every listed truck qualifies. "No driver" is fine: drivers are assigned per \
delivery.
- why: one sentence with the numbers from the table, e.g. "0.1 km from the \
Apapa Wharf pickup, waiting 2 h 14 min, position 1 min old." """


def build_ask_graph(llm):
    """Wire the ask-box graph. `llm` is a chat model; tools are bound here."""
    model = llm.bind_tools(OPERATOR_TOOLS)

    async def agent(state: AskState) -> dict:
        # The system prompt is prepended each turn rather than stored in
        # state, so it can't be pushed out by a long tool conversation.
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        reply = await model.ainvoke(messages)
        return {"messages": [reply]}

    def needs_tools(state: AskState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AskState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(OPERATOR_TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", needs_tools, {"tools": "tools", END: END})
    # Unconditional: after tools run, the model always gets to see the result.
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_proposal_graph(llm):
    """The timed pass: one model call, no tools. The proposer hands it a table
    with the distances already worked out, so there is nothing to fetch and
    little to reason about. With tool rounds and open-ended reasoning a tick
    took 9-43 s.
    """
    # include_raw: the token counts are logged per tick, so cost stays visible.
    decider = llm.with_structured_output(TickOutput, include_raw=True)

    async def decide(state: ProposeState) -> dict:
        out = await decider.ainvoke([SystemMessage(PROPOSE_PROMPT)] + state["messages"])
        u = out["raw"].usage_metadata or {}
        log.info(
            "tick: model call in=%s out=%s reasoning=%s",
            u.get("input_tokens"),
            u.get("output_tokens"),
            (u.get("output_token_details") or {}).get("reasoning"),
        )
        if out["parsing_error"]:
            raise out["parsing_error"]
        return {"result": out["parsed"]}

    graph = StateGraph(ProposeState)
    graph.add_node("decide", decide)
    graph.add_edge(START, "decide")
    graph.add_edge("decide", END)
    return graph.compile()
