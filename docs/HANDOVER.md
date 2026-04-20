# EchoChat Demo — Handover

本项目是给 Nature 审稿人用的心超 AI demo，已完整部署在 `eez194.ece.ust.hk`，通过 `https://echochat.micro-heart.com` 对外访问。本文档覆盖接手所需的一切。

---

## 1. 服务器与账号

| | 值 |
|---|---|
| Host | `eez194.ece.ust.hk` |
| User（日常） | `jguoaz` — 读写代码、起/停服务、跑 git |
| Admin | `xmli` — 只用于 `sudo`（nginx、certbot、apt） |
| 项目根目录 | `/nfs/usrhome2/EchoChatDemo/`（NFS，多机共享） |
| 模型权重 | `/home/jguoaz/usdemo/echochatv1.5/`（本地盘，16 GB；**不要移到 NFS**，NFS 慢 10x） |
| GitHub | <https://github.com/JR-Guo/EchoChatDemo> |
| 公网域名 | <https://echochat.micro-heart.com> |
| 登录共享密码 | `echochat` |

GPU 分配：cuda:1 → view classifier，cuda:2 → echochat 主模型。

---

## 2. 一键启动（正常场景）

SSH 到 `jguoaz@eez194.ece.ust.hk`，两条命令：

```bash
# 2.1 视图分类器（port 8996, cuda:1）
cd /nfs/usrhome2/jguoaz/PAH_platform/AI_service
tmux new-session -dA -s view-classifier \
  'export PYTHONPATH=$(pwd); export CUDA_VISIBLE_DEVICES=1; \
   exec /home/jguoaz/anaconda3/envs/platformpah/bin/python api_view_openai.py \
     --host 0.0.0.0 --port 8996 --device cuda:0 2>&1 | tee /tmp/vc.log'

# 2.2 echochat 主服务（port 12345, cuda:2）
cd /nfs/usrhome2/EchoChatDemo
bash scripts/run_prod.sh
```

等 ~100 秒模型加载完，两端健康检查：

```bash
curl -s http://127.0.0.1:8996/health    # {"status":"healthy","model_loaded":true,...}
curl -s http://127.0.0.1:12345/healthz  # {"status":"ok","model_loaded":true,...}
```

两个都返回 `model_loaded: true` 后，浏览器打开 <https://echochat.micro-heart.com/login>，密码 `echochat`。

---

## 3. 停止服务

```bash
tmux kill-session -t echochat-demo
tmux kill-session -t view-classifier

# tmux kill 有时不彻底，补一刀：
pkill -9 -f 'gunicorn.*12345'
pkill -9 -f 'api_view_openai'
```

---

## 4. 日志

```bash
# echochat 主服务
tail -f /nfs/usrhome2/EchoChatDemo/logs/service.log
tmux attach -t echochat-demo     # Ctrl+B, D 退出（不要 Ctrl+C）

# view classifier
tail -f /tmp/vc.log
tmux attach -t view-classifier
```

---

## 5. 烟测（任何改动后先跑一遍）

```bash
cd /nfs/usrhome2/EchoChatDemo
bash scripts/smoke.sh /nfs/usrhome2/echo/6F6O8CO0_anon
```

预期输出 9 个步骤全过，最后一行 `saved /tmp/report.pdf (~10KB)`。

如果想看完整 UI 流程（measurement / disease / vqa），参考 `docs/superpowers/plans/2026-04-18-echochat-demo.md` §6 的 API 契约，或直接浏览器演示。

---

## 6. 目录结构

```
/nfs/usrhome2/EchoChatDemo/
├── app/                        FastAPI 应用层
│   ├── main.py                 入口；lifespan context 加载模型
│   ├── auth.py                 共享密码登录 + Secure cookie
│   ├── config.py               pydantic-settings，读 .env
│   ├── storage.py              filesystem 存储（study/<id>/ 布局）
│   ├── routers/
│   │   ├── pages.py            HTML 页面
│   │   ├── meta.py             /api/constants /healthz
│   │   ├── upload.py           /api/study /process SSE
│   │   ├── study.py            /api/study/<id> GET/PATCH/DELETE
│   │   ├── tasks.py            /api/study/<id>/task/* + SSE
│   │   └── report_io.py        PATCH section + export PDF/DOCX
│   ├── services/
│   │   ├── dicom_pipeline.py   pydicom + opencv → mp4/png
│   │   ├── view_classifier.py  httpx 客户端（OpenAI-compat API）
│   │   ├── echochat_engine.py  ms-swift PtEngine wrapper + asyncio.Lock
│   │   ├── export.py           weasyprint PDF + python-docx DOCX
│   │   ├── progress.py         per-task 内存 pub/sub（SSE）
│   │   └── tasks/{report,measurement,disease,vqa}.py
│   └── models/                 Pydantic schemas
├── constants/                  28 diseases / 22 measurements / 38 views / presets / prompts
├── templates/                  Jinja2 HTML（base + login + home + upload + workspace）
├── static/                     CSS (Tailwind tokens) + JS (vanilla)
├── scripts/
│   ├── run_prod.sh             主服务启动脚本
│   ├── run_dev.sh              本地热重载
│   ├── deploy.sh               rsync（现在直接在服务器改，不常用）
│   └── smoke.sh                端到端测试
├── tests/                      71 tests（pytest -m 'not slow'）
├── docs/
│   ├── target.md               产品需求
│   ├── disease(1).py / measurement(1).py / report(1).py   授权能力列表
│   └── superpowers/
│       ├── specs/2026-04-18-echochat-demo-design.md
│       └── plans/2026-04-18-echochat-demo.md
├── logs/service.log            运行时日志
├── data/sessions/<study_id>/   运行时数据（每 study 一目录）
└── .env                        本地配置（不入 git）
```

---

## 7. `.env` 配置

必备字段：

```bash
ECHOCHAT_HOST=0.0.0.0
ECHOCHAT_PORT=12345
ECHOCHAT_MODEL_PATH=/home/jguoaz/usdemo/echochatv1.5
ECHOCHAT_DATA_DIR=/nfs/usrhome2/EchoChatDemo/data
VIEW_CLASSIFIER_URL=http://127.0.0.1:8996
VIEW_CLASSIFIER_API_KEY=pah-guNsdzQV6r7tOZiN1-G431DAWE57T8Z1HCcj9BLaKDY
SHARED_PASSWORD=echochat
SESSION_SECRET=<随机 64 字符>
CUDA_VISIBLE_DEVICES=2
VIDEO_MAX_PIXELS=20971520
VIDEO_MIN_PIXELS=13107
FPS=1
FPS_MAX_FRAMES=16
```

`.env` 不进 git。换服务器/账号时自己填。

---

## 8. Conda 环境 / 依赖

用的是 **`/home/jguoaz/anaconda3/envs/platformpah`**（Python 3.11.14, torch 2.9.0+cu128）。

相关 Python 依赖分三处：

1. conda env 自带：torch, opencv, httpx, jinja2（省了装）
2. **ms-swift 从 NFS editable install**：`/nfs/usrhome2/jguoaz/EchoTTA_workspace/ms-swift/`
3. 其他依赖 `pip install --user` 到 `~/.local/lib/python3.11/site-packages/`（fastapi, pydantic-settings, sse-starlette, weasyprint, python-docx, pydicom<3, pytest, pytest-asyncio, respx, qwen-vl-utils<0.0.12, msgspec 等）

**接手人要重建环境**（新账号 / 迁机器）：

```bash
# 1. ms-swift
cd /nfs/usrhome2/jguoaz/EchoTTA_workspace/ms-swift
/home/jguoaz/anaconda3/envs/platformpah/bin/pip install --user -e .

# 2. 其他 Python 依赖
/home/jguoaz/anaconda3/envs/platformpah/bin/pip install --user \
  'fastapi>=0.125' 'pydantic-settings>=2.2' 'sse-starlette>=2.1' 'pydicom<3.0' \
  'weasyprint>=60.2' 'python-docx>=1.1' 'gunicorn>=21.2' 'respx>=0.21' \
  'python-multipart' 'jinja2>=3.1' 'itsdangerous>=2.2' 'qwen-vl-utils<0.0.12' \
  'av' 'decord' 'opencv-python-headless>=4.9' 'pytest>=8.0' 'pytest-asyncio>=0.23' \
  'msgspec'
```

**坑**：`qwen-vl-utils` 必须 `<0.0.12`，新版跟 ms-swift 内部 `require_version` 冲突。`flash-attn` **不要装**，代码已改用 torch 自带 `sdpa`，装 flash-attn 反而编译 20 分钟还容易失败。

---

## 9. Nginx / HTTPS

- 配置文件：`/etc/nginx/sites-available/echochat.micro-heart.com`（要 `sudo` 看）
- SSL：Let's Encrypt，certbot 自动续期（systemd timer），证书路径 `/etc/letsencrypt/live/echochat.micro-heart.com/`
- 安全头：HSTS / X-Content-Type-Options / X-Frame-Options 都已配
- 反代：`localhost:12345` → `echochat.micro-heart.com`
- SSE 相关设置：`proxy_buffering off; proxy_read_timeout 900s;` 已在配置里

改完 nginx 配置后：

```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## 10. 开发工作流

本地目录 `/root/sync_workspace/jiarong/eez195/echo_ui/` 在**不用了**（原本用于 eez195，后来换到 eez194 后代码直接改在服务器）。现在流程：

```bash
# 在服务器改代码
ssh jguoaz@eez194.ece.ust.hk
cd /nfs/usrhome2/EchoChatDemo

# 改完跑测试
/home/jguoaz/anaconda3/envs/platformpah/bin/pytest -m 'not slow'

# 改涉及模型启动逻辑 → 重启服务
tmux kill-session -t echochat-demo
pkill -9 -f 'gunicorn.*12345'
bash scripts/run_prod.sh

# push
git add -u
git commit -m "..."
git push origin main
```

改纯前端（templates/ static/）不需要重启 —— uvicorn 用 Jinja 即时渲染。不过 gunicorn 没开 `--reload`，为了保险重启一下。

---

## 11. 常见故障

| 症状 | 原因 | 解决 |
|---|---|---|
| 启动时 `Address already in use` | tmux kill 没杀干净 gunicorn master | `pkill -9 -f 'gunicorn.*12345'` 再起 |
| healthz `model_loaded: false` 不变 | 模型加载卡死 | 确认 `ECHOCHAT_MODEL_PATH=/home/jguoaz/usdemo/echochatv1.5`（本地盘，不是 NFS）|
| 上传后 clip 切面显示 "Unknown" | view classifier 没起 | 跑 §2.1 |
| 生成报告 10 段全空 | 模型输出没被 parser 匹配 | `logs/service.log` 找 "raw output" 看原始响应，可能需要调 `app/services/tasks/report.py` 的 `_split_sections` |
| SSE 在浏览器里断开 | nginx 默认 buffer SSE | 配置 `proxy_buffering off`（已配）|
| PDF 导出失败 | weasyprint 系统依赖缺（libpango） | `apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libcairo2` |
| 浏览器弹"证书过期" | Let's Encrypt 90 天没续上 | `sudo certbot renew --force-renewal`；然后 `sudo systemctl reload nginx` |

---

## 12. 已知限制 & 待办（低优）

按影响从高到低：

1. **VQA 示例题库是空的** — `constants/presets.py` 里 `VQA_EXAMPLES = []`。领域专家填几条（"What view is this?"、"Describe LV wall motion" 等）就好，前端下拉菜单会自动显示。
2. **Sessions 目录无 TTL** — `data/sessions/` 只增不减，定期手动清 / 上 cron。
3. **日志 rotation** — `logs/service.log` 用 `tee -a`，无限增长。可以加 logrotate。
4. **多医生并发** — 全局共享密码 `echochat`，所有人能互相访问 study（URL 里 UUID 不泄漏就相对安全）。硬隔离需要加账号系统。
5. **sdpa vs flash-attn** — 目前 sdpa，单任务推理慢 ~20%（30s → 36s）。并发很多时可考虑装 flash-attn（麻烦）。
6. **Trash 目录不自动清** — 删 clip 会移到 `sessions/<id>/trash/` 而不是真删（早期防 `rm` 误删的设计）。
7. **Cookie Secure flag 只在 HTTPS 下设** — 代码已用 `X-Forwarded-Proto` 探测，不用手动改。

---

## 13. 架构速览

```
浏览器
   │  HTTPS
   ▼
nginx (eez194:443)  ─ HSTS, Secure cookie, HTTP→HTTPS 301
   │ proxy_pass http://127.0.0.1:12345
   ▼
gunicorn + uvicorn worker (single worker, model loaded once)
   │  tmux "echochat-demo" / cuda:2
   │
   ├─── app/services/echochat_engine.py  (ms-swift PtEngine, sdpa attention)
   │        └─> echochatv1.5 on cuda:2   (16 GB VRAM)
   │
   ├─── app/services/view_classifier.py  (httpx client)
   │        └─> http://127.0.0.1:8996/v1/chat/completions  (OpenAI-compat)
   │              │  tmux "view-classifier" / cuda:1
   │              └─> EchoViewCLIP (38-class, 1.3 GB)
   │
   └─── app/services/dicom_pipeline.py   (pydicom + opencv → mp4/png)
                └─> /nfs/usrhome2/EchoChatDemo/data/sessions/<id>/converted/
```

每个 study 在 `data/sessions/<id>/` 下一个目录：`raw/`（原 DICOM）、`converted/`（转出的 mp4/png + 缩略图）、`meta.json`、`results/*.json`、`exports/*.{pdf,docx}`。

---

## 14. 联系 / 参考

- 需求文档：`docs/target.md`
- 设计文档：`docs/superpowers/specs/2026-04-18-echochat-demo-design.md`
- 实施计划（含 API 契约 + 代码片段）：`docs/superpowers/plans/2026-04-18-echochat-demo.md`
- GitHub：<https://github.com/JR-Guo/EchoChatDemo>

有问题先看 `docs/superpowers/specs/` 里的设计文档，最详细。
