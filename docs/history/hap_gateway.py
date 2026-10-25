"""hap_gateway.py - FastAPI gateway for the Hayekian Agent Protocol (v2).

Thin HTTP layer over hap_core: all validation lives in the engine, so the
gateway cannot drift out of sync with the protocol rules.

Requires: fastapi, pydantic v2, cryptography.
Run:      uvicorn hap_gateway:app --reload

State is in memory and single-process. Use a database and a lock (or a
serialised queue) if you deploy more than one worker.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from hap_core import (
    AgentRegistry,
    AuctionOpen,
    Bid,
    BidRejected,
    Task,
    VickreyAuctionEngine,
)

app = FastAPI(title="HAP Gateway API", version="2.0.0")

registry = AgentRegistry()
engine = VickreyAuctionEngine(registry)

# Map engine rejection codes to HTTP statuses.
_REJECTION_STATUS = {
    "unknown_task": status.HTTP_404_NOT_FOUND,
    "unknown_agent": status.HTTP_401_UNAUTHORIZED,
    "key_mismatch": status.HTTP_401_UNAUTHORIZED,
    "bad_signature": status.HTTP_401_UNAUTHORIZED,
    "closed": status.HTTP_410_GONE,
    "over_reserve": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "duplicate": status.HTTP_409_CONFLICT,
}


class AgentRegistration(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_pubkey: str = Field(..., pattern=r"^(0x)?[a-fA-F0-9]{64}$")


class TaskSubmission(BaseModel):
    description: str
    required_capability: str
    reserve_tokens: int = Field(..., gt=0, description="Maximum payout; bids above it are rejected")
    bid_window_seconds: float = Field(30.0, gt=0, le=3600)


class BidSubmission(BaseModel):
    task_id: str
    agent_id: str
    agent_pubkey: str
    bid_amount_tokens: int = Field(..., gt=0)
    nonce: str = Field(..., min_length=1)
    signature: str


@app.post("/v1/agents/register", status_code=status.HTTP_201_CREATED)
async def register_agent(reg: AgentRegistration):
    try:
        ok = registry.register(reg.agent_id, reg.agent_pubkey)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=409, detail="agent_id is already bound to a different key")
    return {"agent_id": reg.agent_id, "status": "REGISTERED"}


@app.post("/v1/tasks/publish", status_code=status.HTTP_201_CREATED)
async def publish_task(sub: TaskSubmission):
    task = Task(
        task_id=str(uuid.uuid4()),
        description=sub.description,
        required_capability=sub.required_capability,
        reserve_tokens=sub.reserve_tokens,
        bid_deadline=time.time() + sub.bid_window_seconds,
    )
    engine.publish(task)
    return {"task_id": task.task_id, "bid_deadline": task.bid_deadline, "status": "PUBLISHED"}


@app.post("/v1/auctions/bid", status_code=status.HTTP_202_ACCEPTED)
async def submit_bid(sub: BidSubmission):
    try:
        engine.submit_bid(Bid(**sub.model_dump()))
    except BidRejected as exc:
        raise HTTPException(
            status_code=_REJECTION_STATUS.get(exc.reason, status.HTTP_400_BAD_REQUEST),
            detail={"reason": exc.reason, "message": str(exc)},
        )
    return {"status": "BID_ACCEPTED"}


@app.post("/v1/auctions/{task_id}/settle")
async def settle_auction(task_id: str):
    try:
        result = engine.settle(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    except AuctionOpen:
        raise HTTPException(status_code=409, detail="Bid deadline has not passed")
    if result is None:
        return {"task_id": task_id, "status": "NO_VALID_BIDS"}
    return {
        "task_id": result.task_id,
        "status": "SETTLED",
        "winner_id": result.winner_id,
        "payout_tokens": result.payout_tokens,
        "n_valid_bids": result.n_valid_bids,
    }
