# scenario_manager.py  ── 硬核升级版
"""
升级：
  1. 使用 WorldEngine（FSM）替换 WorldEnvironment
  2. load_era 后并发初始化所有 Agent 的人格指纹
  3. save_state / load_state 使用 WorldEngine.to_dict/from_dict
  4. 向 Agent 注入 era 字符串（供 PersonaEngine 用）
"""
import asyncio
import json
import os
import re

from infra.llm_client import llm_chat

from config import get_settings
from agent import SocialAgent
from world_engine import WorldEngine
from prompt_templates import SCENARIO_GENERATOR

_settings = get_settings()


class ScenarioManager:
    def __init__(self):
        self.scenarios_base_dir = os.path.join(_settings.DATA_DIR, "scenarios")
        os.makedirs(self.scenarios_base_dir, exist_ok=True)

        self.agents: list = []
        self.world_env: WorldEngine = None
        self.scene_desc = ""
        self.current_task = ""
        self.shared_workspace = ""
        self.current_dialogue = ""
        self._era_name = ""

    # ── 世界生成 ──────────────────────────────────────────────

    async def generate_dynamic_scenario(self, theme: str, genre: str, session_id: str):
        print(f"\n🌌 [世界架构师] 开辟平行时空「{theme}」({genre})")
        prompt = SCENARIO_GENERATOR.substitute(theme=theme, genre=genre)
        try:
            raw = await llm_chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2000,
                timeout=60.0,
                retries=3,
                fallback_text="",
            )
            if not raw:
                raise ValueError("LLM 返回为空")
            raw = _strip_json(raw.strip())
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                raise ValueError("未能提取有效 JSON")

            world_data = json.loads(match.group(0))
            target_dir = os.path.join(self.scenarios_base_dir, session_id)
            os.makedirs(target_dir, exist_ok=True)

            with open(os.path.join(target_dir, "scene.json"), "w", encoding="utf-8") as f:
                json.dump(world_data["scene"], f, ensure_ascii=False, indent=4)
            for agent in world_data["agents"]:
                with open(os.path.join(target_dir, f"{agent['name']}.json"), "w", encoding="utf-8") as f:
                    json.dump(agent, f, ensure_ascii=False, indent=4)

            print(f"✅ 创世完成 → {target_dir}")
            return session_id
        except Exception as e:
            print(f"❌ 场景生成失败: {e}")
            return None

    # ── 世界加载（同步，供 asyncio.to_thread 调用）────────────

    def load_era(self, era_folder_name: str) -> list:
        era_path = os.path.join(self.scenarios_base_dir, era_folder_name)
        if not os.path.exists(era_path):
            raise FileNotFoundError(f"找不到时代文件夹: {era_path}")

        self._era_name = era_folder_name

        with open(os.path.join(era_path, "scene.json"), "r", encoding="utf-8") as f:
            scene_data = json.load(f)

        era_str = scene_data.get("era", "")
        self.scene_desc = (
            f"【所处时代】{era_str}\n"
            f"【地点】{scene_data.get('location')}\n"
            f"【环境】{scene_data.get('scene_desc')}"
        )

        # ── 加载或初始化世界状态 ─────────────────────────────
        state_file = os.path.join(era_path, "state.json")
        if os.path.exists(state_file):
            print(f"🔄 发现存档，唤醒 [{era_folder_name}]…")
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            self.current_task      = state_data.get("current_task",      scene_data.get("current_task",""))
            self.shared_workspace  = state_data.get("shared_workspace",  scene_data.get("initial_workspace",""))
            self.current_dialogue  = state_data.get("current_dialogue",  scene_data.get("initial_dialogue",""))
            world_dict = state_data.get("world_engine")
            if world_dict:
                self.world_env = WorldEngine.from_dict(world_dict)
            else:
                self.world_env = WorldEngine(state_data.get("env_variables", scene_data.get("env_variables",{})))
        else:
            print(f"🌱 初始化全新世界…")
            self.current_task     = scene_data.get("current_task", "")
            self.shared_workspace = scene_data.get("initial_workspace", "")
            self.current_dialogue = scene_data.get("initial_dialogue", "")
            self.world_env = WorldEngine(scene_data.get("env_variables", {}))

        # ── 加载 Agents ───────────────────────────────────────
        self.agents = []
        required_keys = ["name","identity","personality","initial_metrics","task_role"]
        for fn in os.listdir(era_path):
            if fn.endswith(".json") and fn not in ["scene.json","state.json"]:
                try:
                    with open(os.path.join(era_path, fn), "r", encoding="utf-8") as f:
                        ad = json.load(f)
                    if not all(k in ad for k in required_keys):
                        continue
                    agent = SocialAgent(**{k: ad[k] for k in required_keys})
                    agent.set_era(era_str)
                    self.agents.append(agent)
                except Exception as e:
                    print(f"⚠️ 跳过 {fn}: {e}")

        # ── 挂载 RAG ─────────────────────────────────────────
        try:
            from rag_engine import KnowledgeRetriever
            kr = KnowledgeRetriever(era_folder_name)
            for agent in self.agents:
                agent.mount_knowledge(kr)
        except Exception:
            print("ℹ️  无离线知识库，依赖模型基础能力。")

        print(f"✅ [{era_str}] 唤醒了 {len(self.agents)} 位数字生命")
        return self.agents

    async def initialize_agents(self):
        """
        并发为所有 Agent 初始化人格指纹。
        在 server.py 的 create_world / stream_next 首次调用后执行。
        """
        await asyncio.gather(*(a.initialize() for a in self.agents))

    # ── 存档 ──────────────────────────────────────────────────

    def save_state(self, session_id: str):
        target_dir = os.path.join(self.scenarios_base_dir, session_id)
        if not os.path.exists(target_dir):
            return
        state = {
            "current_task":     self.current_task,
            "shared_workspace": self.shared_workspace,
            "current_dialogue": self.current_dialogue[-_settings.DIALOGUE_CONTEXT_WINDOW:],
            "world_engine":     self.world_env.to_dict() if self.world_env else {},
        }
        with open(os.path.join(target_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"):   text = text[3:]
    if text.endswith("```"):       text = text[:-3]
    return text.strip()
