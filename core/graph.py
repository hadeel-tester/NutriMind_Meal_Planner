"""LangGraph meal planning agent.

Graph flow:
    START → load_profile → agent → tools (if tool call) → agent → ...
                                 → END  (if no tool call)

Nodes:
    load_profile  — reads user_id from state, injects profile before the ReAct loop
    agent         — ReAct reasoning step; calls LLM with bound tools
    tools         — executes whichever tool the agent selected (ToolNode)
"""

import os
import sys

# Ensure project root is on sys.path when this file is run directly
# (e.g. `python core/graph.py`). Has no effect when imported as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from core.state import AgentState
from core.memory import init_db, load_profile as db_load_profile
from prompts.system_prompts import MEAL_PLANNER_SYSTEM_PROMPT

load_dotenv()
init_db()

# ---------------------------------------------------------------------------
# Dummy tool — replaced by real tools in Prompt 3–6
# ---------------------------------------------------------------------------

@tool
def get_nutrition(food_name: str) -> str:
    """Get nutrition macros for a food item. Returns calories, protein, carbs, and fat per 100g.

    Use this tool whenever you need nutrition data for a specific food before
    including it in a meal plan. Never guess nutrition values.

    Args:
        food_name: Name of the food item (e.g. 'chicken breast', 'brown rice').

    Returns:
        A string with calorie and macro information per 100g.
    """
    return (
        f"Nutrition for '{food_name}' (placeholder data): "
        "200 kcal, 10 g protein, 25 g carbs, 5 g fat per 100 g. "
        "This will be replaced by real CIQUAL lookup in Prompt 3."
    )


_TOOLS = [get_nutrition]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_llm_with_tools = _llm.bind_tools(_TOOLS)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = {
    "health_goals": "balanced nutrition",
    "dietary_restrictions": [],
    "calorie_target": 2000,
    "allergies": [],
}


def load_profile(state: AgentState) -> dict:
    """Entry node: load user profile from SQLite and inject into state.

    If a profile is found, a HumanMessage summarising it is prepended to
    state["messages"] so the agent has explicit profile context from turn 1.
    If no profile is found, a sensible default is used.
    """
    user_id = state.get("user_id", "default")
    profile = db_load_profile(user_id)
    profile_found = profile is not None
    if not profile_found:
        profile = DEFAULT_PROFILE.copy()

    updates: dict = {"user_profile": profile, "current_step": "agent"}

    if profile_found:
        restrictions = profile.get("dietary_restrictions") or []
        allergies = profile.get("allergies") or []
        summary = (
            f"[User profile loaded]\n"
            f"Name: {profile.get('name') or 'not set'}\n"
            f"Age: {profile.get('age') or 'not set'}\n"
            f"Weight: {profile.get('weight_kg') or 'not set'} kg  "
            f"Height: {profile.get('height_cm') or 'not set'} cm\n"
            f"Health goals: {profile.get('health_goals') or 'balanced nutrition'}\n"
            f"Dietary restrictions: {', '.join(restrictions) if restrictions else 'none'}\n"
            f"Allergies: {', '.join(allergies) if allergies else 'none'}\n"
            f"Calorie target: {profile.get('calorie_target') or 2000} kcal/day"
        )
        updates["messages"] = [HumanMessage(content=summary)]

    return updates


def agent(state: AgentState) -> dict:
    """ReAct reasoning node: calls the LLM with the current message history.

    Prepends a system message that includes the user's health profile so the
    LLM can personalise its recommendations on every turn.
    """
    profile = state.get("user_profile") or DEFAULT_PROFILE
    profile_text = (
        f"Health goals: {profile.get('health_goals', 'balanced nutrition')}\n"
        f"Dietary restrictions: {', '.join(profile.get('dietary_restrictions', [])) or 'none'}\n"
        f"Calorie target: {profile.get('calorie_target', 2000)} kcal/day\n"
        f"Allergies: {', '.join(profile.get('allergies', [])) or 'none'}"
    )
    system_msg = SystemMessage(
        content=f"{MEAL_PLANNER_SYSTEM_PROMPT}\n\n## User health profile\n{profile_text}"
    )
    messages = [system_msg] + list(state["messages"])
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Route to tools if the last message has tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("load_profile", load_profile)
_builder.add_node("agent", agent)
_builder.add_node("tools", ToolNode(_TOOLS))

_builder.add_edge(START, "load_profile")
_builder.add_edge("load_profile", "agent")
_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
_builder.add_edge("tools", "agent")

meal_agent = _builder.compile()

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke-test invocation...\n")
    result = meal_agent.invoke(
        {
            "messages": [HumanMessage(content="Create a simple meal plan for today")],
            "user_id": "test_user_001",
            "user_profile": {},
            "meal_plan": {},
            "shopping_list": [],
            "current_step": "start",
            "error": None,
        }
    )

    for msg in result["messages"]:
        label = type(msg).__name__
        content = msg.content if msg.content else f"[tool_calls: {msg.tool_calls}]"
        print(f"{label}: {content}\n")
