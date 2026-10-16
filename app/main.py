import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import customer
from .client import APIError
from .config import settings
from .graph import build_ask_graph, build_proposal_graph
from .proposer import run_tick
from .tools import api

logging.basicConfig(level=logging.INFO)
# httpx logs every request at INFO; at four or five per tick that buries the
# one line per tick that says what happened.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("app.scheduler")

llm = ChatOpenAI(
    model=settings.model,
    base_url=settings.model_base_url,
    api_key=settings.openrouter_api_key,
)


ask_graph = build_ask_graph(llm)
propose_graph = build_proposal_graph(llm)

# One tick at a time, scheduled or manual. A tick that finds this held is
# skipped, never queued: a queued tick would reason about a fleet that has
# moved on since it was due.
tick_lock = asyncio.Lock()


def _outcome(result: dict) -> str:
    status = result.get("status")
    if status == "ok":
        return f"proposed {result['proposed']}, dropped {result['dropped']}"
    if status == "idle":
        return f"idle: {result['reason']}"
    return f"{status}: {result.get('error') or result.get('reason')}"


async def locked_tick(source: str) -> dict:
    # No await between the check and the acquire, so nothing can slip in.
    if tick_lock.locked():
        log.info("tick (%s): skipped, the previous tick is still running", source)
        return {"status": "skipped", "reason": "a tick is already running"}
    async with tick_lock:
        result = await run_tick(propose_graph)
    log.info("tick (%s): %s", source, _outcome(result))
    return result


async def _scheduled_tick() -> None:
    # run_tick catches what it expects; this catches the rest. An exception
    # escaping a task is swallowed by asyncio, and the agent would just go
    # quiet with nothing to say why.
    try:
        await locked_tick("scheduled")
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("tick (scheduled): crashed; the next one runs as usual")


async def scheduler() -> None:
    """A tick every `tick_seconds`, on a fixed cadence. Each runs as its own
    task, so a slow tick doesn't push the schedule back: the next one comes
    due on time and is skipped if the slow one is still going."""
    log.info("scheduler: a tick every %gs", settings.tick_seconds)
    running: set[asyncio.Task] = set()
    try:
        while True:
            task = asyncio.create_task(_scheduled_tick())
            running.add(task)
            task.add_done_callback(running.discard)
            await asyncio.sleep(settings.tick_seconds)
    finally:
        # Shutdown: stop an in-flight tick and wait for it, rather than
        # leaving it pending when the loop closes.
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ticker = asyncio.create_task(scheduler())
    yield
    ticker.cancel()
    with suppress(asyncio.CancelledError):
        await ticker
    await api.aclose()


app = FastAPI(title="Dispatch agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3471"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    """Up, and able to reach the dispatch API."""
    try:
        await api.list_trucks()
    except APIError as e:
        return {"status": "degraded", "dispatch_api": e.message}
    except Exception as e:
        return {"status": "degraded", "dispatch_api": str(e)}
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        result = await ask_graph.ainvoke({"messages": [HumanMessage(req.question)]})
    except Exception as e:
        # Out of credits, a provider outage, a timeout. Say which, briefly,
        # instead of a bare 500 the console can only call "failed".
        logging.getLogger("app.ask").exception("ask failed")
        raise HTTPException(502, detail=f"The model call failed: {type(e).__name__}: {str(e)[:160]}")
    return AskResponse(answer=result["messages"][-1].content)


class CustomerAsk(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    # The viewer's IANA zone, so times read as their local time. Falls back to the operator's.
    timezone: str | None = Field(default=None, max_length=64)


customer_limit = customer.RateLimit(settings.customer_questions_per_link, settings.customer_questions_per_minute)
# Units and the operator's zone change only with a restart of the Go API.
_operator: dict = {}


@app.post("/track/{token}/ask", response_model=AskResponse)
async def customer_ask(token: str, req: CustomerAsk) -> AskResponse:
    """A customer's question about the one delivery their link is for. No tools:
    the answer comes only from GET /track/{token} (see customer.py)."""
    question = req.question.strip()
    if len(token) != 16:
        raise HTTPException(404, detail="delivery not found")
    if not question:
        raise HTTPException(422, detail="Type a question.")
    if not customer_limit.allow(token):
        raise HTTPException(429, detail=customer.more_help_note())
    try:
        track = await api.track(token)
        if not _operator:
            _operator.update(await api.get_config())
    except APIError as e:
        if e.status == 404:
            raise HTTPException(404, detail="delivery not found")
        raise HTTPException(503, detail="Delivery details aren't available right now.")
    tz = customer.zone_for(req.timezone, _operator.get("timezone", "UTC"))
    try:
        text = await customer.answer(llm, track, question, tz, _operator.get("units", "metric"))
    except Exception as e:
        logging.getLogger("app.customer").exception("customer ask failed")
        raise HTTPException(502, detail=f"Questions aren't available right now ({type(e).__name__}).")
    return AskResponse(answer=text)


@app.post("/tick")
async def tick():
    """Run one proposal pass now. Shares the scheduler's lock, so it's
    skipped rather than run alongside a scheduled tick."""
    return await locked_tick("manual")
