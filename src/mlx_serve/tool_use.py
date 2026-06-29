"""
tool_use.py — Agent loop for server-side tool execution.

When a model supports tool calling and the user sets `"tool_use": "server"`
in the request body (or on the model config), mlx-serve runs an agent loop:

  1. Send messages + tools to mlx_lm
  2. If the model returns tool calls, execute them
  3. Feed results back and repeat
  4. Return final response to client

This module handles the orchestration. The actual HTTP proxying is done by
router.py's chat_completions endpoint.
"""

import json
import logging
from typing import Any

from . import config, events

logger = logging.getLogger("mlx-serve.tool_use")


class ToolExecutionError(Exception):
    """Raised when a tool call fails during execution."""


async def run_agent_loop(
    messages: list[dict],
    tools: list[dict] | None,
    model_name: str,
    hf_path: str,
    request_headers,
    *,
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict:
    """
    Run the agent loop for a chat completion request with tool use.

    Args:
        messages: Chat messages (will be modified in-place to add tool results).
        tools: Tool definitions from the request.
        model_name: Friendly model name.
        hf_path: HuggingFace path for the model.
        request_headers: Original request headers.
        max_iterations: Max agent loop iterations (default from config).
        timeout_seconds: Timeout per iteration in seconds.
        stream: Whether to stream the response.
        **kwargs: Additional parameters to pass to mlx_lm (temperature, etc.).

    Returns:
        Final response dict from mlx_lm.
    """
    max_iters = max_iterations or config.TOOL_CONFIG.max_iterations
    timeout = timeout_seconds or config.TOOL_CONFIG.timeout_seconds

    # Track iteration
    iteration = 0
    total_tool_calls = 0
    response_data: dict = {}

    # Build the request body for mlx_lm
    body = {
        "model": hf_path,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }
    if tools:
        body["tools"] = tools

    logger.info(
        f"Agent loop starting for {model_name}: "
        f"iterations={max_iters}, timeout={timeout}s, tools={len(tools) if tools else 0}"
    )

    # Import here to avoid circular imports
    from .router import _HTTP_CLIENT

    target = f"http://127.0.0.1:{config.MLX_PORT}/v1/chat/completions"

    while iteration < max_iters:
        iteration += 1
        logger.info(f"Agent loop iteration {iteration}/{max_iters}")

        # Send request to mlx_lm
        resp = await _HTTP_CLIENT.post(
            target,
            json=body,
            headers={k: v for k, v in request_headers.items() if k.lower() in {"content-type", "authorization"}},
            timeout=timeout,
        )

        if resp.status_code != 200:
            raise ToolExecutionError(
                f"mlx_lm returned {resp.status_code}: {resp.text[:500]}"
            )

        response_data = resp.json()
        choices = response_data.get("choices", [])

        if not choices:
            raise ToolExecutionError("Empty response from mlx_lm")

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason")

        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total_tool_calls += len(tool_calls)
            logger.info(
                f"Iteration {iteration}: model made {len(tool_calls)} tool call(s)"
            )
            events.emit(
                events.EventType.TOOL_CALL,
                model=model_name,
                detail={"tool_count": len(tool_calls), "iteration": iteration},
            )

            # Append the assistant message (with tool_calls) to the conversation
            # BEFORE the tool results — mlx_lm requires this ordering.
            messages.append(message)

            # Extract tool call details and execute each tool
            tool_results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                try:
                    args_str = func.get("arguments", "{}")
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                    logger.warning(f"Failed to parse arguments for tool {tool_name}")

                # Execute the tool
                result = await _execute_tool(tool_name, args, model_name)
                events.emit(
                    events.EventType.TOOL_RESULT,
                    model=model_name,
                    detail={"tool": tool_name},
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": tool_name,
                        "content": str(result),
                    }
                )

            # Add tool results to messages
            messages.extend(tool_results)

            # Rebuild body for next iteration
            body["messages"] = messages
            if tools:
                body["tools"] = tools

            # Continue loop
            continue

        # No tool calls or finish_reason indicates completion
        if finish_reason in ("stop", "length"):
            logger.info(
                f"Agent loop completed in {iteration} iteration(s), "
                f"{total_tool_calls} total tool call(s)"
            )
            return response_data

        # Unexpected finish reason
        logger.warning(f"Unexpected finish_reason: {finish_reason}")
        return response_data

    # Max iterations reached
    logger.warning(
        f"Agent loop reached max iterations ({max_iters}) for {model_name}"
    )
    events.emit(
        events.EventType.REQUEST_COLD_START,
        model=model_name,
        detail={"agent_loop_max_iterations": max_iters},
    )
    return response_data


async def _execute_tool(
    tool_name: str, args: dict, model_name: str
) -> Any:
    """
    Execute a tool by name with the given arguments.

    Currently supports a limited set of built-in tools. In a full implementation,
    this would support:
    - Built-in tools (search, calculate, etc.)
    - User-defined tools from the request
    - MCP tools from the model pool

    Args:
        tool_name: Name of the tool to execute.
        args: Arguments to pass to the tool.
        model_name: Model name for logging.

    Returns:
        Tool execution result.
    """
    logger.info(f"Executing tool {tool_name} with args: {args}")

    # Built-in tools
    if tool_name == "calculate":
        # Safe arithmetic evaluation — only allows numbers and + - * / ** ( )
        import ast
        import operator as op

        _SAFE_OPS = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.Mod: op.mod,
            ast.FloorDiv: op.floordiv,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
        }

        def _safe_eval(node):
            if isinstance(node, ast.Constant):  # numbers
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value!r}")
            elif isinstance(node, ast.BinOp):
                left = _safe_eval(node.left)
                right = _safe_eval(node.right)
                if type(node.op) not in _SAFE_OPS:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return _SAFE_OPS[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _safe_eval(node.operand)
                if type(node.op) not in _SAFE_OPS:
                    raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
                return _SAFE_OPS[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported expression: {type(node).__name__}")

        try:
            expression = str(args.get("expression", ""))
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "search":
        query = args.get("query", "")
        # Placeholder: in a full implementation, this would call an external search API
        return f"Search results for: {query} (placeholder)"

    elif tool_name == "get_weather":
        location = args.get("location", "unknown")
        # Placeholder
        return f"Weather in {location}: 22°C, sunny (placeholder)"

    else:
        # Unknown tool
        logger.warning(f"Unknown tool requested: {tool_name}")
        return f"Tool '{tool_name}' is not implemented"


def format_tool_definitions(tools: list[dict]) -> list[dict]:
    """
    Format tool definitions for mlx_lm's apply_chat_template.

    mlx_lm expects tools in OpenAI format:
    [
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]

    Args:
        tools: Tool definitions from the request.

    Returns:
        Formatted tool definitions.
    """
    formatted = []
    for tool in tools:
        if tool.get("type") == "function" or "function" in tool:
            formatted.append(tool)
        else:
            # Assume function type if not specified
            formatted.append({"type": "function", "function": tool})
    return formatted
