"""ReAct Agent - Reason + Act loop for warehouse action queries.

Architecture:
  - LLM decides which tool to call (or declares done) each iteration
  - interrupt() pauses for human approval BEFORE each tool execution
  - After approval → execute tool → update react_steps → loop or finish
  - Max MAX_REACT_STEPS iterations to prevent runaway loops

HITL flow:
  1. LLM selects tool + args + reasoning
  2. interrupt() → user sees {tool_name, args, reasoning}
  3. User calls POST /api/action/approve {session_id, approved}
  4. If approved → execute tool, store result
  5. LLM sees tool result → decides next step or "done"
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from src.core import get_settings
from src.core.state import AgentState
from src.core.prompts import get_prompt
from src.core.tracing import trace_agent_node
from src.core.resilience import resilient_call, llm_circuit, llm_rate_limiter
from src.agents.action_tools import execute_tool, get_tools_prompt
import structlog

log = structlog.get_logger()
settings = get_settings()

MAX_REACT_STEPS = 5

_REACT_SYSTEM_FALLBACK = """{tools_prompt}

You are a warehouse management action agent. Your job is to execute actions requested by the user.

REACT LOOP RULES:
1. Analyze the user request and any previous tool results.
2. Decide: should you call a tool OR are you done?
3. Reply with ONLY valid JSON - no markdown, no explanation outside JSON.

If you need to call a tool, reply:
{{
  "action": "call_tool",
  "tool_name": "<tool name>",
  "tool_args": {{<args object>}},
  "reasoning": "<1-2 sentence explanation of WHY you are calling this tool>"
}}

If you are done (no more tools needed), reply:
{{
  "action": "done",
  "summary": "<concise summary of what was accomplished, mentioning all tool results>"
}}

IMPORTANT:
- Only call tools that exist in the list above.
- Be precise with argument types (int, list, str).
- If a previous tool failed, explain the failure in reasoning or declare done.
- Never call more than {max_steps} tools total.
- Extract IDs from the user query or previous results.
"""


def _get_llm():
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )


def _build_react_prompt(query: str, react_steps: list[dict]) -> str:
    """Build the message history for the ReAct loop."""
    parts = [f"User request: {query}"]
    if react_steps:
        parts.append(f"\nPrevious steps completed ({len(react_steps)}/{MAX_REACT_STEPS} max):")
        for step in react_steps:
            parts.append(
                f"  Step {step['step']}: called {step['tool']}({json.dumps(step['args'])}) "
                f"→ {'SUCCESS' if step.get('success') else 'FAILED'}: {step.get('message', '')[:200]}"
            )
    return "\n".join(parts)


def _parse_llm_decision(content: str) -> dict:
    """Extract JSON decision from LLM response, handling markdown fences."""
    content = content.strip()
    # Strip markdown fences
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    # Find first JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(content)


@trace_agent_node("react_agent", prompt_key="react_system")
def react_agent_node(state: AgentState) -> dict:
    """Single iteration of the ReAct loop.

    Each call: Think → Interrupt(approve) → Execute → return updated state.
    The supervisor loops back here until status != 'processing'.
    """
    query = state.get("original_query", "")
    react_steps: list[dict] = list(state.get("react_steps") or [])
    messages = state.get("messages", [])

    # Guard: max steps exceeded
    if len(react_steps) >= MAX_REACT_STEPS:
        log.warning("react_max_steps_reached", steps=len(react_steps))
        return {
            "messages": messages,
            "react_steps": react_steps,
            "react_result": f"Reached maximum of {MAX_REACT_STEPS} action steps. Completed: " +
                            "; ".join(s.get("message", "")[:80] for s in react_steps),
            "status": "completed",
        }

    # ── THINK: LLM decides next action ──────────────────────────────────────
    try:
        react_system_template = get_prompt("react_system")
    except Exception as e:
        log.error("prompt_load_failed", prompt="react_system", error=str(e))
        return {
            "messages": state.get("messages", []),
            "status": "failed",
            "error": f"Prompt 'react_system' unavailable: {e}",
            "react_steps": react_steps,
        }

    system_content = react_system_template.format(
        tools_prompt=get_tools_prompt(),
        max_steps=MAX_REACT_STEPS,
    )
    user_content = _build_react_prompt(query, react_steps)

    try:
        llm = _get_llm()
        response = resilient_call(
            llm.invoke,
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ],
            circuit=llm_circuit,
            rate_limiter=llm_rate_limiter,
        )
        decision = _parse_llm_decision(response.content)
    except Exception as e:
        log.error("react_llm_error", error=str(e))
        return {
            "messages": messages,
            "react_steps": react_steps,
            "react_result": f"ReAct agent error: {e}",
            "status": "failed",
            "error": str(e),
        }

    action = decision.get("action", "done")

    # ── DONE branch ──────────────────────────────────────────────────────────
    if action == "done":
        summary = decision.get("summary", "Actions completed.")
        log.info("react_done", steps=len(react_steps), summary=summary[:100])
        return {
            "messages": messages,
            "react_steps": react_steps,
            "react_result": summary,
            "status": "completed",
            "pending_tool_call": None,
        }

    # ── CALL_TOOL branch ─────────────────────────────────────────────────────
    tool_name = decision.get("tool_name", "")
    tool_args = decision.get("tool_args", {})
    reasoning = decision.get("reasoning", "")
    step_num = len(react_steps) + 1

    log.info("react_tool_selected", tool=tool_name, step=step_num)

    # ── HITL: Interrupt for human approval ───────────────────────────────────
    approval_response = interrupt({
        "type": "tool_approval",
        "step": step_num,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "reasoning": reasoning,
        "steps_so_far": len(react_steps),
        "max_steps": MAX_REACT_STEPS,
    })

    approved = approval_response.get("approved", False)
    feedback = approval_response.get("feedback", "")

    if not approved:
        rejection_msg = feedback or "User rejected this tool call."
        log.info("react_tool_rejected", tool=tool_name, step=step_num)
        new_step = {
            "step": step_num,
            "tool": tool_name,
            "args": tool_args,
            "reasoning": reasoning,
            "approved": False,
            "success": False,
            "message": f"Rejected: {rejection_msg}",
            "result": {},
        }
        return {
            "messages": messages,
            "react_steps": react_steps + [new_step],
            "react_result": f"Action stopped at step {step_num}: {rejection_msg}",
            "status": "action_rejected",
            "pending_tool_call": None,
        }

    # ── EXECUTE the approved tool ─────────────────────────────────────────────
    log.info("react_executing_tool", tool=tool_name, args=tool_args, step=step_num)
    result = execute_tool(tool_name, tool_args)

    new_step = {
        "step": step_num,
        "tool": tool_name,
        "args": tool_args,
        "reasoning": reasoning,
        "approved": True,
        "success": result.get("success", False),
        "message": result.get("message", ""),
        "result": result.get("data", {}),
    }
    updated_steps = react_steps + [new_step]

    log.info(
        "react_step_done",
        tool=tool_name,
        success=result.get("success"),
        step=step_num,
    )

    # Return "processing" → supervisor will loop back to react_agent
    return {
        "messages": messages,
        "react_steps": updated_steps,
        "status": "processing",
        "pending_tool_call": None,
    }
