"""Durable per-task spending ledger; reserve before any paid request.

An unresolved request retains its full reservation across crashes. All requests
using this ledger run under one process lock. The cap never resets between runs.
Credentials and account-wide balances belong outside source control.
"""

import fcntl
import json
import os
import time
from decimal import Decimal
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path, cap="0.50"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.path.with_suffix(".lock")).open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if Decimal(self.data["cap_usd"]) != Decimal(cap):
                raise ValueError("existing budget cap cannot be silently changed")
        else:
            self.data = {"cap_usd": str(cap), "entries": []}
            self.save()

    def save(self):
        temp = self.path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @property
    def committed(self):
        return sum(
            (Decimal(e.get("cost_usd", e["reserved_usd"])) for e in self.data["entries"]),
            Decimal(0),
        )

    def reserve(self, request_id, amount):
        amount = Decimal(str(amount))
        if not amount.is_finite() or amount <= 0:
            raise ValueError("reservation must be finite and positive")
        if any(e["request_id"] == request_id for e in self.data["entries"]):
            raise ValueError("request already recorded; reuse saved response, do not rebill")
        if self.committed + amount > Decimal(self.data["cap_usd"]):
            raise BudgetExceeded("request would exceed the cumulative task budget")
        self.data["entries"].append(
            {
                "request_id": request_id,
                "reserved_usd": str(amount),
                "status": "reserved",
                "time": time.time(),
            }
        )
        self.save()

    def settle(self, request_id, cost, **metadata):
        cost = Decimal(str(cost))
        if not cost.is_finite() or cost < 0:
            raise ValueError("invalid reported cost")
        entry = next(e for e in self.data["entries"] if e["request_id"] == request_id)
        if entry["status"] != "reserved":
            raise ValueError("request already settled")
        if cost > Decimal(entry["reserved_usd"]):
            raise BudgetExceeded("provider charge exceeds reserved bound; stop and audit")
        entry.update(status="settled", cost_usd=str(cost), **metadata)
        self.save()

    def summary(self):
        actual = sum(
            (Decimal(e["cost_usd"]) for e in self.data["entries"] if "cost_usd" in e), Decimal(0)
        )
        return {
            "cap_usd": self.data["cap_usd"],
            "reported_cost_usd": str(actual),
            "committed_including_unresolved_usd": str(self.committed),
            "requests": len(self.data["entries"]),
            "unresolved": sum(e["status"] == "reserved" for e in self.data["entries"]),
        }

    def close(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
