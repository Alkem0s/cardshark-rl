import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = r"C:\Users\efe20\.gemini\antigravity-ide\brain\6d7cdc79-c081-429e-9d37-906a22322292\.system_generated\logs\transcript.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        step = data.get("step_index")
        if step is not None and step >= 7055:
            t = data.get("type")
            src = data.get("source")
            content = str(data.get("content", ""))
            if t == "USER_INPUT":
                print(f"\n=== USER INPUT (step {step}) ===\n{content}")
            elif t == "PLANNER_RESPONSE":
                content_text = data.get("content") or ""
                if content_text:
                    print(f"\n--- PLANNER RESPONSE (step {step}) ---\n{content_text}")
                calls = data.get("tool_calls", [])
                if calls:
                    print(f"Calls: {[(c.get('name'), c.get('args')) for c in calls]}")


