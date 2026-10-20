import uuid

from langchain_core.tools import tool

from .client import APIError, DispatchAPI

# One client for the process. httpx pools connections.
api = DispatchAPI()

DELIVERY_STATUSES = {"pending", "assigned", "in_transit", "completed", "cancelled", "failed"}


@tool
async def list_trucks() -> str:
    """List every truck with its status, last known position and driver id.

    Statuses are available, on_delivery, maintenance, offline.
    A truck with null lng/lat has never reported a position.
    """
    try:
        trucks = await api.list_trucks()
    except APIError as e:
        return f"Could not list trucks: {e.message}"
    return _compact(trucks)


@tool
async def list_deliveries(status: str = "") -> str:
    """List deliveries, optionally filtered by status.

    Valid statuses: pending, assigned, in_transit, completed, cancelled, failed.
    Pass an empty string for all of them. The list is thin — use
    get_delivery for stops and the event timeline.
    """
    status = status.strip().lower()
    if status and status not in DELIVERY_STATUSES:
        return f"Unknown status {status!r}. Use one of: {', '.join(sorted(DELIVERY_STATUSES))}, or an empty string."
    try:
        deliveries = await api.list_deliveries(status or None)
    except APIError as e:
        return f"Could not list deliveries: {e.message}"
    return _compact(deliveries)

@tool
async def list_drivers() -> str:
    """List every driver with their id, name and phone.

    Trucks carry a driver_id, not a name — use this to turn one into the
    other. There is no lookup by id; fetch the list and match.
    """
    try:
        drivers = await api.list_drivers()
    except APIError as e:
        return f"Could not list drivers: {e.message}"
    return _compact(drivers)





@tool
async def get_delivery(delivery_id: str) -> str:
    """Full detail for one delivery: its stops, its events, and the position
    of the truck carrying it. delivery_id is a uuid from list_deliveries.
    """
    try:
        uuid.UUID(delivery_id.strip())
    except ValueError:
        # The model sometimes passes the reference (DEL-2026-10001) instead.
        return "delivery_id must be the uuid `id` from list_deliveries, not the reference."
    try:
        detail = await api.get_delivery(delivery_id.strip())
    except APIError as e:
        return f"Could not load delivery: {e.message}"
    return _compact(detail)


def _compact(value) -> str:
    """Tool results go into the prompt, so they pay for themselves in tokens.
    Pretty-printing is for logs, not for the model."""
    import json

    return json.dumps(value, separators=(",", ":"), default=str)


OPERATOR_TOOLS = [list_trucks, list_drivers, list_deliveries, get_delivery]