# tools.py
import json
from datetime import datetime, timezone

# --- Tool schemas: the CONTRACT the model sees. It never sees the implementation. ---
# The description is load-bearing: it's how the model decides WHETHER to call this.
TOOLS = [
  {
    "name": "get_current_time",
    "description": "Get the current UTC date and time. Use whenever the user asks about "
                   "'now', 'today', or anything time-relative.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
  },
  {
    "name": "lookup_order",
    "description": "Look up a customer order by its ID. Returns status, total and carrier. "
                   "Use when the user mentions an order number.",
    "input_schema": {
      "type": "object",
      "properties": {
        "order_id": {"type": "string", "description": "Order identifier, e.g. 'A-1042'"},
      },
      "required": ["order_id"],
    },
  },
]

# Fake DB — swap for a real repository later
_ORDERS = {
  "A-1042": {"status": "shipped", "total_usd": 149.99, "carrier": "DHL"},
  "A-1043": {"status": "payment_failed", "total_usd": 89.00, "carrier": None},
}


def get_current_time() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lookup_order(order_id: str) -> str:
  order = _ORDERS.get(order_id)
  if order is None:
    # Raise here; the loop converts it into an is_error tool_result the model can recover from
    raise KeyError(f"order {order_id} not found")
  return json.dumps(order)


# name -> callable dispatch table (your handler mapping)
REGISTRY = {
  "get_current_time": get_current_time,
  "lookup_order": lookup_order,
}


def execute_tool(name: str, tool_input: dict) -> str:
  fn = REGISTRY.get(name)
  if fn is None:
    raise KeyError(f"unknown tool: {name}")
  # WARNING: tool_input comes from the MODEL. Treat it as untrusted input.
  return fn(**tool_input)