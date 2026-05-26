

import json
import os
from datetime import datetime


LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_logs.json")



def _load_logs() -> list[dict]:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_logs(logs: list[dict]) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)



def log_attack(data: dict, attack_type: str = "unknown") -> None:
    
    entry = dict(data)  # don't mutate the caller's dict
    entry["timestamp"] = str(datetime.now())

    if "attack_type" not in entry:
        entry["attack_type"] = attack_type

    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)


def get_all_logs() -> list[dict]:
    return _load_logs()


def get_logs_by_type(attack_type: str) -> list[dict]:
    return [l for l in _load_logs() if l.get("attack_type") == attack_type]


def clear_logs() -> None:
    _save_logs([])


def get_summary_stats() -> dict:
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