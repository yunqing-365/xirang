-- 息壤 PostgreSQL 初始化脚本
-- 用于存储用户档案（替换 JSON 文件存储）

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id          TEXT PRIMARY KEY,
    data             JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 知识掌握度热力图（反范式化，便于按朝代查询）
CREATE TABLE IF NOT EXISTS knowledge_mastery (
    user_id          TEXT NOT NULL,
    knowledge_key    TEXT NOT NULL,
    era              TEXT,
    label            TEXT,
    score            INT  DEFAULT 50,
    attempts         INT  DEFAULT 0,
    last_seen        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, knowledge_key)
);

-- 小测验历史
CREATE TABLE IF NOT EXISTS quiz_history (
    id               BIGSERIAL PRIMARY KEY,
    user_id          TEXT NOT NULL,
    session_id       TEXT,
    era              TEXT,
    questions_total  INT,
    correct          INT,
    score_pct        INT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 班级房间（持久化）
CREATE TABLE IF NOT EXISTS classrooms (
    room_id          TEXT PRIMARY KEY,
    room_name        TEXT,
    teacher_id       TEXT,
    session_id       TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at        TIMESTAMPTZ
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_knowledge_mastery_user  ON knowledge_mastery(user_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_mastery_era   ON knowledge_mastery(era);
CREATE INDEX IF NOT EXISTS idx_quiz_history_user       ON quiz_history(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated   ON user_profiles(updated_at);

-- 自动更新 updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_user_profiles_updated
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- 管理员用户（演示）
INSERT INTO user_profiles (user_id, data)
VALUES ('admin', '{"roles": ["admin"], "total_sessions": 0}')
ON CONFLICT DO NOTHING;
