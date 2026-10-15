import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from .client import APIError
from .tools import api

log = logging.getLogger(__name__)

# Which trucks count as candidates. Filtering here, not in the prompt, means
# the model only sees real options, and a tick with none makes no model call.
# This model judged the raw table badly without long reasoning.
RADIUS_KM = 2.0
FRESH_MIN = 15
# How many trucks the model sees per delivery, and how many deliveries per
# tick. The table is the whole prompt, so these bound its size.
NEAREST = 3
MAX_DELIVERIES = 20


@dataclass
class Pick:
    """A model proposal mapped back to ids."""

    delivery_id: str
    truck_id: str
    reference: str
    plate: str
    why: str

    @property
    def title(self) -> str:
        return f"Assign {self.plate} to {self.reference}"


async def run_tick(graph) -> dict:
    """One pass: read state, build the table, one model call, validate, write.

    Returns a summary rather than raising, because this runs on a timer and
    one bad tick must not stop the next.
    """
    try:
        pending = await api.list_deliveries("pending")
        trucks = await api.list_trucks()
        existing = await api.list_proposals("pending")
        rejected = await api.list_proposals("rejected")
    except APIError as e:
        log.warning("tick: could not read state: %s", e.message)
        return {"status": "error", "error": e.message}

    # Anything already waiting for the dispatcher is off limits, so it isn't
    # shown to the model at all.
    claimed_deliveries = {p["delivery_id"] for p in existing if p["delivery_id"]}
    claimed_trucks = {p["truck_id"] for p in existing if p["truck_id"]}
    open_deliveries = [d for d in pending if d["id"] not in claimed_deliveries]
    available = [t for t in trucks if t["status"] == "available"]
    free = [t for t in available if t["id"] not in claimed_trucks]
    # A truck that has never reported a position can't be judged near anything.
    placed = [t for t in free if t["lat"] is not None and t["lng"] is not None]

    # Cheap guards before spending a model call. Most ticks end here.
    if not pending:
        return {"status": "idle", "reason": "nothing pending"}
    if not available:
        return {"status": "idle", "reason": "no available trucks"}
    if not open_deliveries or not free:
        return {"status": "idle", "reason": "everything open already has a proposal"}
    if not placed:
        return {"status": "idle", "reason": "no available truck has a position"}

    # Oldest first: they're the ones worth raising.
    open_deliveries.sort(key=lambda d: d["created_at"])
    open_deliveries = open_deliveries[:MAX_DELIVERIES]
    pickups = await asyncio.gather(*(_pickup(d["id"]) for d in open_deliveries))
    rows = [(d, p) for d, p in zip(open_deliveries, pickups) if p is not None]
    if not rows:
        return {"status": "idle", "reason": "no pickup positions"}

    now = datetime.now(timezone.utc)
    # A pair the dispatcher already rejected would otherwise come back every tick.
    declined = {(p["delivery_id"], p["truck_id"]) for p in rejected}
    rows = [
        (d, p, near)
        for d, p in rows
        if (near := _candidates([t for t in placed if (d["id"], t["id"]) not in declined], p, now))
    ]
    if not rows:
        return {"status": "idle", "reason": f"no fresh truck within {RADIUS_KM:g} km of a pickup"}

    table = _table(rows, now)
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(table)], "result": None})
    except Exception as e:
        # The model timed out, or its answer didn't parse into TickOutput.
        # Nothing was written; the next tick starts clean.
        log.warning("tick: model pass failed: %r", e)
        return {"status": "error", "error": f"model pass failed: {type(e).__name__}"}
    proposals = result["result"].proposals if result["result"] else []

    picks, dropped = _resolve(proposals, rows, placed)
    kept, invalid = _validate(picks, pending, available, existing)
    dropped += invalid
    for reason in dropped:
        log.info("tick: dropped a proposal: %s", reason)

    written = []
    for p in kept:
        try:
            created = await api.create_proposal(_to_api_shape(p))
            written.append(created["id"])
        except APIError as e:
            # Another process may have taken the truck between validation
            # and this write. Not worth failing the tick over.
            log.info("tick: proposal refused: %s", e.message)

    return {
        "status": "ok",
        "proposed": len(written),
        "dropped": len(dropped),
        "ids": written,
    }


async def _pickup(delivery_id: str) -> dict | None:
    """The first stop. Only the detail endpoint has stops."""
    try:
        detail = await api.get_delivery(delivery_id)
    except APIError as e:
        log.info("tick: no detail for %s: %s", delivery_id, e.message)
        return None
    stop = next((s for s in detail["stops"] if s["stop_type"] == "pickup"), None)
    if not stop or stop["lat"] is None or stop["lng"] is None:
        return None
    return stop


def _candidates(trucks, pickup: dict, now: datetime) -> list[dict]:
    """Free trucks near enough to a pickup, with a recent enough position, nearest first."""
    near = [
        t for t in trucks
        if _km(t, pickup) <= RADIUS_KM and _minutes_since(t["last_seen_at"], now) <= FRESH_MIN
    ]
    return sorted(near, key=lambda t: _km(t, pickup))[:NEAREST]


def _table(rows, now: datetime) -> str:
    """The whole prompt: each delivery that has candidates, with them.

    Distances are worked out here, not by the model. It's cheaper, and the
    model doesn't have to reason its way to a number.
    """
    lines = [
        f"Pending deliveries, oldest first, that have an available truck within "
        f"{RADIUS_KM:g} km with a position under {FRESH_MIN} min old:"
    ]
    for d, pickup, near in rows:
        options = "; ".join(
            f"{t['plate_number']} {_km(t, pickup):.1f} km, position {_ago(t['last_seen_at'], now)} old"
            + ("" if t["driver_id"] else ", no driver")
            for t in near
        )
        lines.append(
            f"- {d['reference']} | waiting {_ago(d['created_at'], now)} | "
            f"pickup {pickup['address']} | {options}"
        )
    return "\n".join(lines)


def _resolve(proposals, rows, trucks) -> tuple[list[Pick], list[str]]:
    """References and plates back to ids. Anything not in the table is dropped."""
    by_ref = {d["reference"]: d["id"] for d, _, _ in rows}
    by_plate = {t["plate_number"]: t["id"] for t in trucks}
    picks, dropped = [], []
    for p in proposals:
        d, t = by_ref.get(p.delivery), by_plate.get(p.truck)
        if not d or not t:
            dropped.append(f"Assign {p.truck} to {p.delivery}: not in the table")
            continue
        picks.append(Pick(d, t, p.delivery, p.truck, p.why))
    return picks, dropped


def _validate(proposals, pending, available, existing):
    """Three layers. The model is told these rules; this enforces them.

    A model that breaks a rule occasionally is normal. A pane full of
    conflicting proposals is not, so the rules are checked rather than
    trusted.
    """
    pending_ids = {d["id"] for d in pending}
    available_ids = {t["id"] for t in available}
    # Anything already waiting for a decision is off limits — proposing it
    # again just fills the pane with duplicates.
    claimed_deliveries = {p["delivery_id"] for p in existing if p["delivery_id"]}
    claimed_trucks = {p["truck_id"] for p in existing if p["truck_id"]}

    kept, dropped = [], []
    seen_deliveries, seen_trucks = set(), set()

    for p in proposals:
        if p.delivery_id not in pending_ids:
            dropped.append(f"{p.title}: delivery is not pending")
            continue
        if p.truck_id not in available_ids:
            dropped.append(f"{p.title}: truck is not available")
            continue
        if p.delivery_id in claimed_deliveries:
            dropped.append(f"{p.title}: delivery already has a pending proposal")
            continue
        if p.truck_id in claimed_trucks:
            dropped.append(f"{p.title}: truck already has a pending proposal")
            continue
        # Within this batch: two proposals for one truck can't both be
        # approved, and the second would 409 in front of the dispatcher.
        if p.delivery_id in seen_deliveries:
            dropped.append(f"{p.title}: a second proposal for the same delivery")
            continue
        if p.truck_id in seen_trucks:
            dropped.append(f"{p.title}: a second proposal for the same truck")
            continue

        seen_deliveries.add(p.delivery_id)
        seen_trucks.add(p.truck_id)
        kept.append(p)

    return kept, dropped


def _km(truck: dict, stop: dict) -> float:
    """Straight-line (haversine) distance, as the console shows it."""
    lat1, lng1, lat2, lng2 = map(math.radians, (truck["lat"], truck["lng"], stop["lat"], stop["lng"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _minutes_since(iso: str | None, now: datetime) -> float:
    if not iso:
        return math.inf
    return max(0.0, (now - datetime.fromisoformat(iso)).total_seconds() / 60)


def _ago(iso: str | None, now: datetime) -> str:
    if not iso:
        return "unknown"
    mins = int(_minutes_since(iso, now))
    return f"{mins} min" if mins < 60 else f"{mins // 60} h {mins % 60} min"


def _to_api_shape(p: Pick) -> dict:
    """The API wants kind, action and changes; the model doesn't produce
    them. Filling them here keeps the ids written once, not twice."""
    return {
        "kind": "change",
        "title": p.title,
        "why": p.why,
        "delivery_id": p.delivery_id,
        "truck_id": p.truck_id,
        "action": {
            "type": "assign",
            "delivery_id": p.delivery_id,
            "truck_id": p.truck_id,
        },
        "changes": [
            {"k": "Truck", "from": "unassigned", "to": p.plate},
            {"k": "Status", "from": "pending", "to": "assigned"},
        ],
    }
