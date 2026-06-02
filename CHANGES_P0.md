# P0 变现功能升级说明

> 版本：v2.2.0-p0  
> 日期：2025-05  
> 依据：息壤产品化与变现策略（xirang_commercialization_strategy）

---

## 新增文件一览

| 文件 | 说明 |
|------|------|
| `infra/sms_auth.py` | 手机号验证码登录/注册 |
| `infra/quota.py` | 使用配额系统（免费3次/月） |
| `infra/invite.py` | 班级码 + 邀请码系统 |
| `infra/payment.py` | 支付宝/微信支付骨架 |
| `deploy/sql/schema_v2_p0.sql` | 新增数据库表结构 |

## 修改文件

| 文件 | 改动 |
|------|------|
| `server.py` | 注册5个新路由；`create_world` 加配额门控；PDF导出加权限门控 |
| `config.py` | 新增 `SMS_SANDBOX`、`PAYMENT_SANDBOX`、支付商配置占位 |

---

## 新增 API 端点

### 🔐 手机号登录（`infra/sms_auth.py`）
```
POST /api/auth/send_sms       获取验证码（沙箱返回明文code）
POST /api/auth/verify_sms     验证+登录（自动注册新用户）
POST /api/auth/wechat_login   微信登录（预留，待接入）
```

### 📊 配额查询（`infra/quota.py`）
```
GET  /api/quota/status        查我的配额和套餐状态
GET  /api/quota/plans         返回所有套餐详情
```

### 🏫 班级码系统（`infra/invite.py`）
```
POST /api/classroom/create         教师创建班级，返回6位班级码
POST /api/classroom/join           学生用班级码加入
GET  /api/classroom/my_classes     教师查看自己的班级列表
GET  /api/classroom/info/{code}    查询班级信息
POST /api/classroom/close/{code}   教师关闭班级

POST /api/invite/generate          管理员批量生成邀请码
POST /api/invite/redeem            用户兑换邀请码激活套餐
GET  /api/invite/list              管理员查看所有邀请码
```

### 💳 支付（`infra/payment.py`）
```
GET  /api/payment/plans            查看可购买套餐及价格
POST /api/payment/create_order     发起支付，获取二维码
GET  /api/payment/order/{id}       查询订单状态（前端轮询）
POST /api/payment/notify/{channel} 支付回调（微信/支付宝）
POST /api/payment/mock_notify      沙箱模拟支付成功 ⬅ 测试用
GET  /api/payment/my_orders        查看历史订单
```

---

## 套餐与配额规则

| 套餐 | 会话次数/月 | 价格 | PDF导出 | 教师大屏 |
|------|-------------|------|---------|---------|
| 免费体验版 | 3次 | ¥0 | ❌ | ❌ |
| 学生版 | 无限 | ¥39/月 · ¥299/年 | ❌ | ❌ |
| 教师专业版 | 无限 | ¥199/月 · ¥1599/年 | ✅ | ✅ |
| 学校版 | 无限 | ¥8000/年 | ✅ | ✅ |

- 超出免费配额时：`POST /api/create_world` 返回 **402** + `upgrade_url`
- 无PDF导出权限时：`POST /api/export/student_report` 返回 **403**
- 教师/管理员角色不受配额限制

---

## 快速测试流程（沙箱模式）

```bash
# 1. 获取验证码（沙箱直接返回code）
curl -X POST http://localhost:8000/api/auth/send_sms \
  -H "Content-Type: application/json" \
  -d '{"phone": "13800138000"}'

# 2. 登录，拿到 JWT token
curl -X POST http://localhost:8000/api/auth/verify_sms \
  -d '{"phone": "13800138000", "code": "上一步返回的code", "role": "teacher"}'

# 3. 教师创建班级码（用上一步的 access_token）
curl -X POST http://localhost:8000/api/classroom/create \
  -H "Authorization: Bearer <token>" \
  -d '{"class_name": "高二3班", "era": "北宋·熙宁变法"}'

# 4. 学生加入班级
curl -X POST http://localhost:8000/api/classroom/join \
  -d '{"class_code": "ABC123", "display_name": "张同学"}'

# 5. 查看配额
curl http://localhost:8000/api/quota/status

# 6. 创建支付订单
curl -X POST http://localhost:8000/api/payment/create_order \
  -d '{"plan": "teacher_pro", "duration_days": 365, "channel": "wechat"}'

# 7. 沙箱模拟支付成功
curl -X POST http://localhost:8000/api/payment/mock_notify \
  -d '{"order_id": "上一步返回的order_id"}'
```

---

## 生产上线 Checklist

- [ ] 环境变量 `XIRANG_SMS_SANDBOX=false`，填写 `ALIYUN_ACCESS_KEY_ID/SECRET`
- [ ] 环境变量 `XIRANG_PAYMENT_SANDBOX=false`，填写微信/支付宝商户参数
- [ ] 运行 `deploy/sql/schema_v2_p0.sql` 建表
- [ ] `XIRANG_AUTH_ENABLED=true` 开启鉴权
- [ ] 配置支付回调域名（需HTTPS）：`https://yourdomain.com/api/payment/notify/wechat`
- [ ] 微信登录：完成开放平台认证后实现 `sms_auth.py` 中的 `wechat_login` TODO
- [ ] 将内存存储（`_users`, `_quota_store`, `_classrooms`, `_orders`）迁移至 PostgreSQL
