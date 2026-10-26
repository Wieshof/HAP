# Historical prototype (September 2026), kept for reference. See docs/history/README.md.
"""hap_core.py - Hayekian Agent Protocol (HAP) reference core, v2.

A sealed-bid, second-price (Vickrey) procurement auction with:
  * identity binding (agent_id <-> Ed25519 public key via a registry)
  * one sealed bid per agent per task
  * bid deadlines and a reserve price (the task poster's maximum payout)
  * canonical, delimiter-safe signing payloads
  * a cost model from which truthful bids are derived
  * a swarm entropy monitor based on real cluster-membership entropy

Scope note: this code binds identities to keys but does NOT provide Sybil
resistance. The Vickrey property assumes one bidder cannot control several
identities. In practice that requires an external mechanism (posted stake,
verified identity, or rate-limited registration).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519


# ---------------------------------------------------------------------------
# Data types and signing
# ---------------------------------------------------------------------------

def _strip_0x(h: str) -> str:
    return h[2:] if h.startswith("0x") else h


def signing_digest(task_id: str, agent_id: str, bid_amount_tokens: int, nonce: str) -> bytes:
    """SHA-256 digest of the canonical JSON payload (matches the JSON schema).

    JSON canonicalisation avoids the ambiguity of ':'-joined strings, where an
    agent_id containing ':' could collide with a different field split.
    """
    body = json.dumps(
        {
            "task_id": task_id,
            "agent_id": agent_id,
            "bid_amount_tokens": bid_amount_tokens,
            "nonce": nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).digest()


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    description: str
    required_capability: str
    reserve_tokens: int      # maximum the poster will pay; bids above it are rejected
    bid_deadline: float      # epoch seconds; sealed bids close at this time


@dataclasses.dataclass(frozen=True)
class Bid:
    task_id: str
    agent_id: str
    agent_pubkey: str
    bid_amount_tokens: int
    nonce: str
    signature: str


@dataclasses.dataclass(frozen=True)
class Settlement:
    task_id: str
    winner_id: str
    payout_tokens: int
    n_valid_bids: int


# ---------------------------------------------------------------------------
# Identity registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """First-write-wins binding of agent_id to an Ed25519 public key."""

    def __init__(self) -> None:
        self._keys: Dict[str, str] = {}

    def register(self, agent_id: str, pubkey_hex: str) -> bool:
        """Returns True if newly registered or identical to the existing binding,
        False if agent_id is already bound to a different key."""
        key = _strip_0x(pubkey_hex).lower()
        try:
            raw = bytes.fromhex(key)
        except ValueError as exc:
            raise ValueError("public key must be hexadecimal") from exc
        if len(raw) != 32:
            raise ValueError("Ed25519 public key must be 32 bytes")
        return self._keys.setdefault(agent_id, key) == key

    def key_for(self, agent_id: str) -> Optional[str]:
        return self._keys.get(agent_id)


# ---------------------------------------------------------------------------
# Auction engine
# ---------------------------------------------------------------------------

class BidRejected(Exception):
    """Raised when a bid fails validation. `reason` is a stable machine code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class AuctionOpen(Exception):
    """Raised when settlement is requested before the bid deadline."""


class VickreyAuctionEngine:
    def __init__(self, registry: AgentRegistry, clock: Callable[[], float] = time.time) -> None:
        self._registry = registry
        self._clock = clock
        self._tasks: Dict[str, Task] = {}
        self._bids: Dict[str, Dict[str, Bid]] = {}
        self._settled: Dict[str, Optional[Settlement]] = {}

    def publish(self, task: Task) -> None:
        if task.task_id in self._tasks:
            raise ValueError("task already published")
        self._tasks[task.task_id] = task
        self._bids[task.task_id] = {}

    @staticmethod
    def _signature_valid(bid: Bid) -> bool:
        try:
            pubkey = ed25519.Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(_strip_0x(bid.agent_pubkey))
            )
            pubkey.verify(
                bytes.fromhex(_strip_0x(bid.signature)),
                signing_digest(bid.task_id, bid.agent_id, bid.bid_amount_tokens, bid.nonce),
            )
            return True
        except (InvalidSignature, ValueError):
            return False

    def submit_bid(self, bid: Bid) -> None:
        task = self._tasks.get(bid.task_id)
        if task is None:
            raise BidRejected("unknown_task", "task not found")
        if bid.task_id in self._settled or self._clock() >= task.bid_deadline:
            raise BidRejected("closed", "bidding is closed")

        registered = self._registry.key_for(bid.agent_id)
        if registered is None:
            raise BidRejected("unknown_agent", "agent is not registered")
        if registered != _strip_0x(bid.agent_pubkey).lower():
            raise BidRejected("key_mismatch", "public key does not match registered identity")
        if not self._signature_valid(bid):
            raise BidRejected("bad_signature", "invalid Ed25519 signature")
        if not 0 < bid.bid_amount_tokens <= task.reserve_tokens:
            raise BidRejected("over_reserve", "bid must be positive and within the reserve")
        if bid.agent_id in self._bids[bid.task_id]:
            # Sealed-bid: one bid per agent, no revisions. This also makes nonce
            # replay harmless, since a replayed bid is just a duplicate.
            raise BidRejected("duplicate", "agent has already bid on this task")

        self._bids[bid.task_id][bid.agent_id] = bid

    def settle(self, task_id: str, *, allow_early: bool = False) -> Optional[Settlement]:
        """Close the auction. Winner = lowest bid; payout = second-lowest bid,
        or the reserve if there is only one bid. Ties break on agent_id so the
        outcome is deterministic. `allow_early` exists for tests and demos."""
        task = self._tasks[task_id]
        if task_id in self._settled:
            return self._settled[task_id]
        if not allow_early and self._clock() < task.bid_deadline:
            raise AuctionOpen("bid deadline has not passed")

        bids = sorted(
            self._bids[task_id].values(),
            key=lambda b: (b.bid_amount_tokens, b.agent_id),
        )
        result: Optional[Settlement] = None
        if bids:
            payout = task.reserve_tokens if len(bids) == 1 else bids[1].bid_amount_tokens
            result = Settlement(task_id, bids[0].agent_id, payout, len(bids))
        self._settled[task_id] = result
        return result


# ---------------------------------------------------------------------------
# Agents and the cost model
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CostModel:
    """C_i(T) = c_token * T_hat + c_compute * L   (both terms in token units)."""
    c_token: float = 1.0
    c_compute: float = 2.0   # tokens per second of latency

    def expected_cost(self, est_tokens: float, est_latency_s: float) -> float:
        return self.c_token * est_tokens + self.c_compute * est_latency_s


class HAPAgent:
    """Bids its expected cost, optionally shaded upward by `margin`.

    With margin=0 the agent bids truthfully, which is a weakly dominant
    strategy in a second-price auction under private values and risk
    neutrality. A positive margin is a crude hedge against the winner's curse
    that arises when cost estimates share common-value components.
    """

    def __init__(
        self,
        agent_id: str,
        capability: str,
        token_budget: int,
        cost_model: CostModel = CostModel(),
        margin: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.agent_id = agent_id
        self.capability = capability
        self.token_budget = token_budget
        self.cost_model = cost_model
        self.margin = margin
        self._rng = rng or random.Random()
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self.pubkey_hex = self._private_key.public_key().public_bytes_raw().hex()

    def _estimate(self, task: Task) -> tuple[float, float]:
        # Stand-in for the agent's private model of the task. A real agent would
        # derive these from the task description and its own history.
        return self._rng.uniform(100, 300), self._rng.uniform(1, 10)

    def generate_bid(self, task: Task) -> Optional[Bid]:
        if task.required_capability != self.capability:
            return None
        est_tokens, est_latency = self._estimate(task)
        if est_tokens > self.token_budget:
            return None
        bid_amount = math.ceil(
            self.cost_model.expected_cost(est_tokens, est_latency) * (1.0 + self.margin)
        )
        if bid_amount > task.reserve_tokens:
            return None

        nonce = uuid.uuid4().hex
        digest = signing_digest(task.task_id, self.agent_id, bid_amount, nonce)
        return Bid(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_pubkey=self.pubkey_hex,
            bid_amount_tokens=bid_amount,
            nonce=nonce,
            signature=self._private_key.sign(digest).hex(),
        )


# ---------------------------------------------------------------------------
# Swarm entropy monitor
# ---------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class SwarmEntropyMonitor:
    """Shannon entropy of cluster membership over candidate-output embeddings.

    Outputs are grouped by greedy leader clustering: each embedding joins the
    most similar existing cluster if cosine similarity to that cluster's leader
    is >= `similarity_cutoff`, otherwise it starts a new cluster. Then

        H = -sum_k P(v_k) log2 P(v_k),   P(v_k) = |cluster k| / n,

    normalised by log2(n) so H lies in [0, 1]. H = 0 means every agent produced
    (semantically) the same output; H = 1 means every output is distinct.

    Limitations:
      * Greedy clustering is order-dependent; use k-means or agglomerative
        clustering if that matters.
      * Low entropy is only a warning sign when diversity is expected. For tasks
        with one correct answer, convergence is legitimate, so pair the breaker
        with external verification (tests, compilers) rather than acting on
        entropy alone.
      * `threshold` is an empirical policy parameter, not a constant. Calibrate
        it on baseline runs for each model family.
    """

    def __init__(self, threshold: float = 0.35, similarity_cutoff: float = 0.9) -> None:
        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff

    def cluster(self, embeddings: Sequence[Sequence[float]]) -> List[int]:
        leaders: List[Sequence[float]] = []
        labels: List[int] = []
        for emb in embeddings:
            best, best_sim = -1, -1.0
            for idx, leader in enumerate(leaders):
                sim = _cosine(emb, leader)
                if sim > best_sim:
                    best, best_sim = idx, sim
            if best >= 0 and best_sim >= self.similarity_cutoff:
                labels.append(best)
            else:
                leaders.append(emb)
                labels.append(len(leaders) - 1)
        return labels

    def compute_entropy(self, embeddings: Sequence[Sequence[float]]) -> float:
        """Normalised cluster entropy in [0, 1]. With fewer than two outputs
        there is nothing to compare, so 1.0 (not evaluable) is returned."""
        n = len(embeddings)
        if n < 2:
            return 1.0
        counts: Dict[int, int] = {}
        for label in self.cluster(embeddings):
            counts[label] = counts.get(label, 0) + 1
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        return max(0.0, h / math.log2(n))

    def is_healthy(self, embeddings: Sequence[Sequence[float]]) -> bool:
        return self.compute_entropy(embeddings) >= self.threshold


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    registry = AgentRegistry()
    engine = VickreyAuctionEngine(registry)
    monitor = SwarmEntropyMonitor(threshold=0.35)

    agents = [
        HAPAgent("agent_coder_01", "python_dev", 5000),
        HAPAgent("agent_coder_02", "python_dev", 4000),
        HAPAgent("agent_coder_03", "python_dev", 4500, margin=0.10),
    ]
    for a in agents:
        registry.register(a.agent_id, a.pubkey_hex)

    task = Task(
        task_id=str(uuid.uuid4()),
        description="Refactor connection pool",
        required_capability="python_dev",
        reserve_tokens=800,
        bid_deadline=time.time() + 60,
    )
    engine.publish(task)

    for a in agents:
        bid = a.generate_bid(task)
        if bid:
            engine.submit_bid(bid)

    result = engine.settle(task.task_id, allow_early=True)
    if result:
        print(
            f"Winner: {result.winner_id} | Payout: {result.payout_tokens} tokens "
            f"| Valid bids: {result.n_valid_bids}"
        )

    # Entropy checks: a diverse swarm, then a full informational cascade.
    diverse = [[0.1, 0.8], [0.9, 0.2], [0.5, 0.5], [-0.7, 0.3]]
    cascade = [[0.30, 0.70], [0.31, 0.69], [0.29, 0.71], [0.30, 0.70]]
    print(f"Diverse swarm  H={monitor.compute_entropy(diverse):.2f} "
          f"healthy={monitor.is_healthy(diverse)}")
    print(f"Cascade swarm  H={monitor.compute_entropy(cascade):.2f} "
          f"healthy={monitor.is_healthy(cascade)}")


if __name__ == "__main__":
    main()
