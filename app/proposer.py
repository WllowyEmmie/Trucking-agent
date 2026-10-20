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
# Failed deliveries: a wider net (it's a recovery, and the alternative is the
# goods sitting on a broken truck), and only recent failures: an old one has
# usually been dealt with outside the system.
FAILED_RADIUS_KM = 10.0
FAILED_LOOKBACK_H = 24

# How a failed delivery goes out again (the API's reopen modes), chosen here
# from the failure reason and whether the goods were collected, the same way
# the console suggests it (lib/reopen.ts). The model only picks the truck.
PLANS = {
    "reassign": ("resume", "Reassign", "nothing was collected, so any truck can do it from the pickup"),
    "try_again": ("resume", "Try again", "the same truck still has the goods and does the drop-offs left"),
    "handover": ("handover", "Hand over", "another truck collects the goods from where the failed truck stopped, then does the drop-offs left"),
    "restart": ("restart", "Start over", "fresh goods from the pickup, every stop again"),
}


@dataclass
class Pick:
    """A model proposal mapped back to ids."""

    delivery_id: str
    truck_id: str
    reference: str
    plate: str
    why: str
    # Set for a failed delivery going out again.
    plan: str | None = None
    old_plate: str | None = None

    @property
    def title(self) -> str:
        if self.plan == "try_again":
            return f"Send {self.plate} again on {self.reference}"
        if self.plan == "handover":
            return f"Hand {self.reference} over from {self.old_plate} to {self.plate}"
        if self.plan == "restart":
            return f"Start {self.reference} over on {self.plate}"
        if self.plan == "reassign":
            return f"Reassign {self.reference} to {self.plate}"
        return f"Assign {self.plate} to {self.reference}"


@dataclass
class FailedRow:
    """A failed delivery, the plan for it, and the trucks that could do it."""

    d: dict
    reason: str | None
    failed_at: str
    collected: bool
    plan: str
    old: dict | None
    # Where the new truck goes first, and what to call it in the table.
    point: dict | None
    point_label: str
    near: list


async def run_tick(graph) -> dict:
    """One pass: read state, build the table, one model call, validate, write.

    Returns a summary rather than raising, because this runs on a timer and
    one bad tick must not stop the next.
    """
    try:
        pending = await api.list_deliveries("pending")
        failed = await api.list_deliveries("failed")
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
    available = [t for t in trucks if t["status"] == "available"]
    free = [t for t in available if t["id"] not in claimed_trucks]
    # A truck that has never reported a position can't be judged near anything.
    placed = [t for t in free if t["lat"] is not None and t["lng"] is not None]
    # A pair the dispatcher already rejected would otherwise come back every tick.
    declined = {(p["delivery_id"], p["truck_id"]) for p in rejected}
    now = datetime.now(timezone.utc)

    # Cheap guards before spending a model call. Most ticks end here.
    if not pending and not failed:
        return {"status": "idle", "reason": "nothing pending or failed"}
    if not free:
        return {"status": "idle", "reason": "no available trucks without a proposal"}

    rows, idle = await _pending_rows(pending, claimed_deliveries, placed, declined, now)
    failed_rows = await _failed_rows(failed, claimed_deliveries, trucks, free, placed, declined, now)
    if not rows and not failed_rows:
        return {"status": "idle", "reason": idle or "no failed delivery has a truck to send"}

    table = "\n\n".join(x for x in (_table(rows, now) if rows else "", _failed_table(failed_rows, now)) if x)
    try:
        result = await graph.ainvoke({"messages": [HumanMessage(table)], "result": None})
    except Exception as e:
        # The model timed out, or its answer didn't parse into TickOutput.
        # Nothing was written; the next tick starts clean.
        log.warning("tick: model pass failed: %r", e)
        return {"status": "error", "error": f"model pass failed: {type(e).__name__}"}
    proposals = result["result"].proposals if result["result"] else []

    picks, dropped = _resolve(proposals, rows, failed_rows, free)
    kept, invalid = _validate(picks, pending, failed, available, existing)
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


async def _pending_rows(pending, claimed, placed, declined, now):
    """Pending deliveries with a near, fresh truck. Also why there are none."""
    open_deliveries = [d for d in pending if d["id"] not in claimed]
    if not pending:
        return [], "nothing pending"
    if not open_deliveries:
        return [], "everything pending already has a proposal"
    if not placed:
        return [], "no available truck has a position"
    # Oldest first: they're the ones worth raising.
    open_deliveries.sort(key=lambda d: d["created_at"])
    open_deliveries = open_deliveries[:MAX_DELIVERIES]
    pickups = await asyncio.gather(*(_pickup(d["id"]) for d in open_deliveries))
    rows = [
        (d, p, near)
        for d, p in zip(open_deliveries, pickups)
        if p is not None
        and (near := _candidates([t for t in placed if (d["id"], t["id"]) not in declined], p, now))
    ]
    return rows, ("" if rows else f"no fresh truck within {RADIUS_KM:g} km of a pickup")


async def _failed_rows(failed, claimed, trucks, free, placed, declined, now) -> list[FailedRow]:
    """Recent failed deliveries with a plan and at least one truck for it."""
    todo = [d for d in failed if d["id"] not in claimed][:MAX_DELIVERIES]
    if not todo:
        return []
    details = await asyncio.gather(*(_detail(d["id"]) for d in todo))
    by_id = {t["id"]: t for t in trucks}
    free_ids = {t["id"] for t in free}
    out = []
    for d, detail in zip(todo, details):
        if not detail:
            continue
        ev = next((e for e in reversed(detail["events"]) if e["event_type"] == "failed"), None)
        if not ev or _minutes_since(ev["created_at"], now) > FAILED_LOOKBACK_H * 60:
            continue
        reason = ev.get("note")
        stops = detail["stops"]
        collected = goods_on_truck(stops)
        old = by_id.get(detail["delivery"].get("truck_id") or "")
        plan = choose_plan(reason, collected, old, free_ids)
        if plan is None:
            continue
        if plan == "try_again":
            near = [old] if (d["id"], old["id"]) not in declined else []
            point, label = None, ""
        else:
            if plan == "handover":
                point, label = old, f"{old['plate_number']}'s last position"
                pool = [t for t in placed if t["id"] != old["id"]]
            else:
                # Start over goes back to the first stop; reassign to the first one left.
                todo_stops = stops if plan == "restart" else [s for s in stops if s["status"] in ("pending", "arrived")]
                first = todo_stops[0] if todo_stops else None
                if not first or first["lat"] is None:
                    continue
                point, label = first, f"the pickup at {first['address']}"
                pool = placed
            near = _candidates([t for t in pool if (d["id"], t["id"]) not in declined], point, now, FAILED_RADIUS_KM)
        if near:
            out.append(FailedRow(d, reason, ev["created_at"], collected, plan, old, point, label, near))
    return out


def goods_on_truck(stops) -> bool:
    """A pickup was collected and a stop is still to do: the goods are on the
    truck that failed. The API decides the same way (reopen.go)."""
    collected = any(s["stop_type"] == "pickup" and s["status"] == "completed" for s in stops)
    open_ = any(s["status"] in ("pending", "arrived") for s in stops)
    return collected and open_


def choose_plan(reason, collected: bool, old: dict | None, free_ids: set) -> str | None:
    """Which way out, from why it failed and where the goods are."""
    if not collected:
        return "restart" if reason == "goods_damaged" else "reassign"
    if reason == "goods_damaged":
        return "restart"
    has_position = bool(old) and old.get("lat") is not None and old.get("lng") is not None
    if reason not in ("breakdown", "accident") and old and old["id"] in free_ids and old.get("driver_id"):
        return "try_again"
    # Broke down, or the truck that has the goods can't go again yet.
    return "handover" if has_position else "restart"


async def _detail(delivery_id: str) -> dict | None:
    try:
        return await api.get_delivery(delivery_id)
    except APIError as e:
        log.info("tick: no detail for %s: %s", delivery_id, e.message)
        return None


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


def _candidates(trucks, pickup: dict, now: datetime, radius_km: float = RADIUS_KM) -> list[dict]:
    """Free trucks with a driver, near enough to a pickup, with a recent enough
    position, nearest first. The API refuses a truck with no driver."""
    near = [
        t for t in trucks
        if t.get("driver_id")
        # Off shift means the driver isn't working; the API also refuses
        # their positions, so a fresh one already implies on shift.
        and t.get("driver_on_shift", True)
        and _km(t, pickup) <= radius_km and _minutes_since(t["last_seen_at"], now) <= FRESH_MIN
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
            for t in near
        )
        lines.append(
            f"- {d['reference']} | waiting {_ago(d['created_at'], now)} | "
            f"pickup {pickup['address']} | {options}"
        )
    return "\n".join(lines)


REASON_TEXT = {
    "breakdown": "breakdown",
    "accident": "accident",
    "customer_unavailable": "customer unavailable",
    "goods_damaged": "goods damaged",
    "other": "other",
}


def _failed_table(rows: list[FailedRow], now: datetime) -> str:
    """Failed deliveries: the plan is fixed, the model picks the truck."""
    if not rows:
        return ""
    lines = [
        "Failed deliveries to send out again. The plan for each is already "
        "chosen from the failure reason and where the goods are; pick one "
        "truck from its row:"
    ]
    for f in rows:
        _mode, title, what = PLANS[f.plan]
        where = "after collecting" if f.collected else "before collecting"
        if f.plan == "try_again":
            options = f"{f.old['plate_number']} (same truck, has the goods)"
        else:
            options = "; ".join(
                f"{t['plate_number']} {_km(t, f.point):.1f} km from {f.point_label}, "
                f"position {_ago(t['last_seen_at'], now)} old"
                for t in f.near
            )
        lines.append(
            f"- {f.d['reference']} | failed {_ago(f.failed_at, now)} ago: "
            f"{REASON_TEXT.get(f.reason or '', 'no reason given')}, {where} | "
            f"plan: {title} ({what}) | {options}"
        )
    return "\n".join(lines)


def _resolve(proposals, rows, failed_rows, trucks) -> tuple[list[Pick], list[str]]:
    """References and plates back to ids. Anything not in the table is
    dropped, and so is a truck that isn't one of that row's options."""
    by_ref = {d["reference"]: (d["id"], {t["plate_number"]: t["id"] for t in near}, None) for d, _, near in rows}
    for f in failed_rows:
        by_ref[f.d["reference"]] = (f.d["id"], {t["plate_number"]: t["id"] for t in f.near}, f)
    picks, dropped = [], []
    for p in proposals:
        row = by_ref.get(p.delivery)
        t = row[1].get(p.truck) if row else None
        if not row or not t:
            dropped.append(f"{p.truck} for {p.delivery}: not in the table")
            continue
        f = row[2]
        picks.append(Pick(row[0], t, p.delivery, p.truck, p.why,
                          plan=f.plan if f else None,
                          old_plate=f.old["plate_number"] if f and f.old else None))
    return picks, dropped


def _validate(proposals, pending, failed, available, existing):
    """Three layers. The model is told these rules; this enforces them.

    A model that breaks a rule occasionally is normal. A pane full of
    conflicting proposals is not, so the rules are checked rather than
    trusted.
    """
    pending_ids = {d["id"] for d in pending}
    failed_ids = {d["id"] for d in failed}
    available_ids = {t["id"] for t in available}
    # Anything already waiting for a decision is off limits — proposing it
    # again just fills the pane with duplicates.
    claimed_deliveries = {p["delivery_id"] for p in existing if p["delivery_id"]}
    claimed_trucks = {p["truck_id"] for p in existing if p["truck_id"]}

    kept, dropped = [], []
    seen_deliveries, seen_trucks = set(), set()

    for p in proposals:
        if p.delivery_id not in (failed_ids if p.plan else pending_ids):
            dropped.append(f"{p.title}: delivery is not {'failed' if p.plan else 'pending'}")
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
    if p.plan:
        mode, title, _what = PLANS[p.plan]
        return {
            "kind": "change",
            "title": p.title,
            "why": p.why,
            "delivery_id": p.delivery_id,
            "truck_id": p.truck_id,
            "action": {
                "type": "reopen",
                "mode": mode,
                "delivery_id": p.delivery_id,
                "truck_id": p.truck_id,
            },
            "changes": [
                {"k": "Plan", "from": "failed", "to": title},
                {"k": "Truck", "from": p.old_plate or "none", "to": p.plate},
                {"k": "Status", "from": "failed", "to": "assigned"},
            ],
        }
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
