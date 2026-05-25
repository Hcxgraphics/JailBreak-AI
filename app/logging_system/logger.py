import json
from datetime import datetime

LOG_FILE = "app/logging_system/attack_logs.json"

def log_attack(data):

    data["timestamp"] = str(datetime.now())

    try:

        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

    except:

        logs = []

    logs.append(data)

    with open(LOG_FILE, "w") as f:

        json.dump(logs, f, indent=4)