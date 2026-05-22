# 息壤 · 架构升级说明 v2.0

## 一、升级前 vs 升级后 对比

| 维度 | v1.0（原版） | v2.0（升级版） |
|---|---|---|
| **并发安全** | 裸全局字典，多请求竞态 | `SessionManager` 加锁 + TTL 过期 |
| **事件循环** | 同步 LLM 调用阻塞 async 路由 | 全 `AsyncOpenAI`，`to_thread` 包裹同步库 |
| **提示词管理** | 散落在各 .py 文件中 | 集中于 `prompt_templates.py` |
| **用户层** | 无用户概念 | `user_profile.py` 跨会话成长档案 |
| **叙事结构** | 线性自动推进 | `narrative_engine.py` 分支选项 + 弧线阶段 |
| **人文启发** | 被动"时空回响" | `reflection_engine.py` 主动触发人文回响 |
| **配置管理** | 硬编码 config.py | Pydantic Settings + `.env` 文件 |
| **API 端点** | 3 个 | 8 个（+健康/选项/选择/反思/档案） |

---

## 二、新文件清单

```
config.py              ← Pydantic Settings，所有配置从 .env 读取
prompt_templates.py    ← 全部 LLM Prompt 集中管理
session_manager.py     ← 异步安全的会话存储，TTL 自动过期
user_profile.py        ← 用户跨会话成长档案系统
narrative_engine.py    ← 玩家选项生成 + 故事弧线管理
reflection_engine.py   ← 人文反思触发与生成
.env.example           ← 环境变量配置模板
```

---

## 三、API 端点全览

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 服务状态检查 |
| POST | `/api/create_world` | 创建新的历史沙盒世界 |
| POST | `/api/intervene` | 注入自由文本干预指令 |
| GET | `/api/choices/{session_id}` | 获取当前回合的叙事选项 |
| POST | `/api/choose` | 提交玩家选择 |
| GET | `/api/stream_next/{session_id}` | SSE 流式推演（核心） |
| POST | `/api/reflect` | 主动触发人文反思 |
| GET | `/api/profile/{user_id}` | 读取用户成长档案 |

---

## 四、SSE 事件类型

前端监听 `/api/stream_next/{session_id}` 时，会收到以下类型的事件：

| `type` 字段 | 含义 | 关键字段 |
|---|---|---|
| `narrator` | 旁白 / 突发事件 | `content` |
| `historical_echo` | 时空回响（文化底蕴揭示） | `content` |
| `thinking` | Agent 正在思考 | `name` |
| `stream_token` | LLM 逐字流输出 | `name`, `content` |
| `agent_action` | 本回合 Agent 完整行动 | `name`, `action`, `dialogue`, `show_image`, `workspace` |
| `choices_ready` | 本回合有选项可拉取 | `round` |
| `reflection` | 人文反思触发 | `insight`, `reflection_question`, `era_fact` |
| `error` | 出错 | `content` |

---

## 五、数据流架构图

```
用户浏览器
    │
    ├─ POST /api/create_world  ──→  ScenarioManager.generate_dynamic_scenario()
    │                                      │ asyncio.to_thread
    │                                      ↓
    │                               SessionManager.create()
    │                               UserProfileStore.save()
    │
    ├─ GET  /api/choices       ──→  NarrativeEngine.generate_choices()
    ├─ POST /api/choose        ──→  NarrativeEngine.choice_to_intervention()
    │                                      │ → SessionManager.set_intervention()
    │
    ├─ GET  /api/stream_next   ──→  SpatiotemporalDirector.direct_next_scene() [async]
    │   (SSE)                           │
    │                                   ↓
    │                          SocialAgent.generate_response_stream() [async gen]
    │                               ├─ KnowledgeRetriever.retrieve() [to_thread]
    │                               ├─ SocialMemory.retrieve_episodic_memory() [to_thread]
    │                               └─ AsyncOpenAI.chat.completions.create() [native async]
    │                                   │
    │                          ReflectionEngine.generate() [async, every N rounds]
    │                          UserProfileStore.save() [to_thread]
    │                          ScenarioManager.save_state() [to_thread]
    │
    └─ POST /api/reflect       ──→  ReflectionEngine.generate() [async]
                                    UserProfileStore.save() [to_thread]
```

---

## 六、下一步建议

**近期（P0）**
- [ ] 前端 `index.html` 接入 `/api/choices` 和 `/api/profile` 端点
- [ ] 前端渲染 `reflection` 事件（弹出人文感悟卡片）
- [ ] 补充 `pydantic-settings` 到 requirements.txt

**中期（P1）**
- [ ] 引入 Redis 替换内存 SessionManager（多进程部署）
- [ ] 为 `user_profile.py` 引入轻量 SQLite 存储替换 JSON 文件
- [ ] 多语言 Prompt 支持（英文版）

**长期（P2）**
- [ ] 语音 TTS 集成（历史人物开口说话）
- [ ] 知识图谱可视化面板
- [ ] 多人协作模式（多个玩家同时干预一个世界）
