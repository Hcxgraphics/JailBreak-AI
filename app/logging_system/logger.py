
import json
import os
from datetime import datetime

import os as _os
LOG_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "attack_logs.json")


# ── Core log/read helpers ─────────────────────────────────────────────────────

def _load_logs() -> list[dict]:

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_logs(logs: list[dict]) -> None:
    """Write the full log list back to disk."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


# ── Public API ────────────────────────────────────────────────────────────────

def log_attack(data: dict, attack_type: str = "unknown") -> None:
    """
    Append a single attack result to the log file.

    Args:
        data:        Dict of attack result fields (from any attack module).
        attack_type: Short label identifying the attack, e.g. "dr_attack".
                     Defaults to data["attack_type"] if already present,
                     otherwise uses the attack_type argument.
    """
    entry = dict(data)  # don't mutate the caller's dict
    entry["timestamp"] = str(datetime.now())

    # Allow the caller to embed attack_type inside data, or pass it explicitly
    if "attack_type" not in entry:
        entry["attack_type"] = attack_type

    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)


def get_all_logs() -> list[dict]:
    """Return every log entry ever saved."""
    return _load_logs()


def get_logs_by_type(attack_type: str) -> list[dict]:
    """
    Filter log entries by attack type.
    Useful when multiple attacks share the same log file.

    Example:
        dr_attack_logs = get_logs_by_type("dr_attack")
    """
    return [l for l in _load_logs() if l.get("attack_type") == attack_type]


def clear_logs() -> None:
    """Wipe all log entries (resets the file to an empty list)."""
    _save_logs([])


def get_summary_stats() -> dict:
    """
    Quick summary across ALL attack types.

    Returns:
        total, successes, overall ASR%, per-attack-type breakdown, latest timestamp.
    """
    logs = _load_logs()
    if not logs:
        return {"total": 0}

    total = len(logs)
    successes = sum(1 for l in logs if l.get("success"))

    # Break down by attack_type
    types: dict[str, dict] = {}
    for l in logs:
        t = l.get("attack_type", "unknown")
        if t not in types:
            types[t] = {"total": 0, "successes": 0}
        types[t]["total"] += 1
        if l.get("success"):
            types[t]["successes"] += 1

    for t, counts in types.items():
        n = counts["total"]
        counts["asr_percent"] = round(counts["successes"] / n * 100, 2) if n else 0.0

    return {
        "total": total,
        "successes": successes,
        "asr_percent": round(successes / total * 100, 2),
        "by_attack_type": types,
        "latest": logs[-1].get("timestamp", ""),
    }