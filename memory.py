# memory.py
import json
import os
from config import DATA_DIR

class SocialMemory:
    def __init__(self, agent_name):
        self.file_path = os.path.join(DATA_DIR, f"memory_{agent_name}.json")
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"relationships": {}}

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def update_relationship(self, target_name, affinity_change, trust_change):
        rel = self.data["relationships"].get(target_name, {"affinity": 50, "trust": 50})
        # 限制在 0-100 之间
        rel["affinity"] = max(0, min(100, rel["affinity"] + affinity_change))
        rel["trust"] = max(0, min(100, rel["trust"] + trust_change))
        self.data["relationships"][target_name] = rel