-- 息壤完整数据库初始化脚本
-- 包含：基础表 + P0 变现功能表
-- 用法：psql -U xirang -d xirang -f init_full.sql
-- 特性：全部使用 IF NOT EXISTS，支持幂等重复执行

-- ══════════════════════════════════════════════
-- 基础功能表（原 init.sql）
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id          TEXT PRIMARY KEY,
    data             JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS quiz_history (
    id               BIGSERIAL PRIMARY KEY,
    user_id          TEXT NOT NULL,
    session_id       TEXT,
    era              TEXT,
    questions_total  INT,
    correct          INT,
    score_delta      INT,
    taken_at         TIMESTAMPTZ DEFAULT NOW()
);

-- 班级基础表（P0 前就存在，schema_v2_p0.sql 中 ALTER 它）
CREATE TABLE IF NOT EXISTS classrooms (
    id               BIGSERIAL PRIMARY KEY,
    teacher_id       TEXT NOT NULL,
    class_code       TEXT UNIQUE,
    class_name       TEXT,
    era              TEXT DEFAULT '北宋·熙宁变法',
    expires_at       TIMESTAMPTZ,
    is_active        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classrooms_teacher ON classrooms(teacher_id);
CREATE INDEX IF NOT EXISTS idx_classrooms_code    ON classrooms(class_code);

-- ══════════════════════════════════════════════
-- P0 变现功能表
-- ══════════════════════════════════════════════

-- 手机号用户表
CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT PRIMARY KEY,
    phone            TEXT UNIQUE,
    wechat_openid    TEXT UNIQUE,
    display_name     TEXT NOT NULL DEFAULT '',
    roles            JSONB NOT NULL DEFAULT '["user"]',
    plan             TEXT NOT NULL DEFAULT 'free',
    plan_expires_at  TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    last_login_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_phone  ON users(phone);
CREATE INDEX IF NOT EXISTS idx_users_openid ON users(wechat_openid);
CREATE INDEX IF NOT EXISTS idx_users_plan   ON users(plan);

-- 订阅配额追踪表
CREATE TABLE IF NOT EXISTS user_quota (
    user_id           TEXT PRIMARY KEY REFERENCES users(user_id),
    plan              TEXT NOT NULL DEFAULT 'free',
    sessions_used     INT  NOT NULL DEFAULT 0,
    messages_used     INT  NOT NULL DEFAULT 0,
    cycle_start       TIMESTAMPTZ NOT NULL DEFAULT date_trunc('month', NOW()),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- 月度配额重置函数
CREATE OR REPLACE FUNCTION reset_monthly_quota()
RETURNS void AS $$
UPDATE user_quota
SET sessions_used = 0,
    messages_used = 0,
    cycle_start   = date_trunc('month', NOW()),
    updated_at    = NOW()
WHERE cycle_start < date_trunc('month', NOW());
$$ LANGUAGE sql;

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    order_id         TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id),
    plan             TEXT NOT NULL,
    duration_days    INT  NOT NULL,
    amount           INT  NOT NULL,
    channel          TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    trade_no         TEXT,
    qr_code_url      TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    paid_at          TIMESTAMPTZ,
    refunded_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_orders_user   ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_trade  ON orders(trade_no);

-- 邀请码表
CREATE TABLE IF NOT EXISTS invite_codes (
    code             TEXT PRIMARY KEY,
    plan             TEXT NOT NULL DEFAULT 'teacher_pro',
    duration_days    INT  NOT NULL DEFAULT 365,
    created_by       TEXT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    expires_at       TIMESTAMPTZ,
    used_by          TEXT REFERENCES users(user_id),
    used_at          TIMESTAMPTZ
);

-- 班级成员表
CREATE TABLE IF NOT EXISTS classroom_members (
    id               BIGSERIAL PRIMARY KEY,
    class_code       TEXT NOT NULL,
    user_id          TEXT NOT NULL REFERENCES users(user_id),
    display_name     TEXT,
    role             TEXT DEFAULT 'student',
    joined_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (class_code, user_id)
);

CREATE INDEX IF NOT EXISTS idx_cm_class ON classroom_members(class_code);
CREATE INDEX IF NOT EXISTS idx_cm_user  ON classroom_members(user_id);

-- 短信验证码表
CREATE TABLE IF NOT EXISTS sms_codes (
    phone            TEXT PRIMARY KEY,
    code             TEXT NOT NULL,
    attempts         INT  NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    expires_at       TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '5 minutes'),
    verified         BOOLEAN DEFAULT FALSE
);

-- 过期验证码清理函数
CREATE OR REPLACE FUNCTION clean_expired_sms()
RETURNS void AS $$
DELETE FROM sms_codes WHERE expires_at < NOW();
$$ LANGUAGE sql;

-- ══════════════════════════════════════════════
-- 视图
-- ══════════════════════════════════════════════

CREATE OR REPLACE VIEW v_teacher_classroom_summary AS
SELECT
    c.teacher_id,
    c.class_code,
    c.class_name,
    c.era,
    c.is_active,
    c.created_at,
    COUNT(cm.user_id) AS member_count
FROM classrooms c
LEFT JOIN classroom_members cm ON cm.class_code = c.class_code
GROUP BY c.teacher_id, c.class_code, c.class_name, c.era, c.is_active, c.created_at;

-- ══════════════════════════════════════════════
-- 注释
-- ══════════════════════════════════════════════

COMMENT ON TABLE users             IS '息壤用户主表，支持手机号和微信双登录';
COMMENT ON TABLE orders            IS '支付订单，对接微信支付和支付宝';
COMMENT ON TABLE invite_codes      IS '邀请码，管理员生成，教师兑换激活专业版';
COMMENT ON TABLE classroom_members IS '班级成员，学生用班级码加入';

-- 完成
SELECT '✅ 息壤数据库初始化完成' AS status;
