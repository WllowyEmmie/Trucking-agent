from typing import Any
from urllib.parse import quote

import httpx

from .config import settings


class APIError(Exception):
    """The Go API refused. `message` is its own wording, worth surfacing."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"{status}: {message}")


class DispatchAPI:
    def __init__(self, base_url: str | None = None):
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.api_url,
            timeout=settings.api_timeout_s,
            headers={
                "Content-Type": "application/json",
                # The API lets the agent read the fleet and write proposals,
                # nothing else.
                "Authorization": f"Bearer {settings.agent_api_key}",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, json: Any = None, params: dict | None = None) -> Any:
        try:
            res = await self._client.request(method, path, json=json, params=params)
        except httpx.TransportError as e:
            # Connection refused, timeout, DNS: the API is down or unreachable.
            # Status 0, as the console's api.ts does, so callers handle it like
            # any other refusal instead of a crash.
            raise APIError(0, f"dispatch API unreachable ({type(e).__name__})") from e
        if res.status_code >= 400:
            try:
                message = res.json().get("error", res.reason_phrase)
            except (ValueError, AttributeError):
                message = res.reason_phrase
            raise APIError(res.status_code, message)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    # ---- reads ----

    async def list_trucks(self) -> list[dict]:
        return await self._request("GET", "/trucks")

    async def list_drivers(self) -> list[dict]:
        return await self._request("GET", "/drivers")

    async def list_deliveries(self, status: str | None = None) -> list[dict]:
        # params, not string building: the status can come from the model.
        return await self._request("GET", "/deliveries", params={"status": status} if status else None)

    async def get_delivery(self, delivery_id: str) -> dict:
        return await self._request("GET", f"/deliveries/{_seg(delivery_id)}")

    async def whoami(self, cookie: str | None, authorization: str | None) -> dict | None:
        """Who a browser request is from, by asking the API with that
        request's own cookie or bearer token (never the agent's key).
        None when it isn't signed in."""
        headers = {"Authorization": authorization or ""}
        if cookie:
            headers["Cookie"] = cookie
        try:
            res = await self._client.get("/auth/me", headers=headers)
        except httpx.TransportError as e:
            raise APIError(0, f"dispatch API unreachable ({type(e).__name__})") from e
        if res.status_code == 401:
            return None
        if res.status_code >= 400:
            raise APIError(res.status_code, res.reason_phrase)
        return res.json()

    async def get_config(self) -> dict:
        return await self._request("GET", "/config")

    async def track(self, token: str) -> dict:
        """Public tracking view. The only read the customer bot gets."""
        return await self._request("GET", f"/track/{_seg(token)}")

    # ---- writes ----
    # Not exposed as tools yet. Auto mode will use assign; nothing else
    # should be reachable by the model.

    async def assign_truck(self, delivery_id: str, truck_id: str) -> dict:
        return await self._request(
            "POST", f"/deliveries/{_seg(delivery_id)}/assign", {"truck_id": truck_id}
        )

    async def list_proposals(self, status: str = "pending") -> list[dict]:
        return await self._request("GET", "/proposals", params={"status": status})

    async def create_proposal(self, proposal: dict) -> dict:
        return await self._request("POST", "/proposals", proposal)


def _seg(value: str) -> str:
    """One path segment. A value with a slash or "../" in it (a model's
    argument, a customer's link) can't reach a different endpoint."""
    return quote(str(value), safe="")