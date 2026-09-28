from __future__ import annotations

import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST, PORT = "127.0.0.1", 8765


def load_local_env() -> None:
    """Load local secrets without adding a third-party dependency."""
    path = ROOT / ".env.local"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
SERVER_CONFIG = {
    "apiBase": os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    "apiKey": os.environ.get("DEEPSEEK_API_KEY", ""),
    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
}

with (ROOT / "data" / "knowledge.json").open("r", encoding="utf-8") as file:
    KNOWLEDGE = json.load(file)
with (ROOT / "data" / "deep_knowledge.json").open("r", encoding="utf-8") as file:
    DEEP_KNOWLEDGE = json.load(file)
with (ROOT / "data" / "paper_knowledge.json").open("r", encoding="utf-8") as file:
    PAPER_KNOWLEDGE = json.load(file)
with (ROOT / "data" / "job_profile.json").open("r", encoding="utf-8") as file:
    JOB_PROFILE = json.load(file)

ALL_TOPICS = [*JOB_PROFILE["topics"], *DEEP_KNOWLEDGE["topics"], *PAPER_KNOWLEDGE["topics"]]

INTENT_RULES = {
    "自我介绍": ["自我介绍", "介绍一下自己", "简单介绍", "个人情况"],
    "项目深挖": ["项目", "实习", "负责", "经历", "挑战", "困难", "成果", "怎么做", "君同", "滨江"],
    "技术原理": ["区别", "原理", "是什么", "解释", "如何理解", "为什么", "python", "fastapi", "postgresql", "redis", "接口", "后端", "缓存", "事务", "索引", "数据抽取", "agent", "rag", "模型", "算法", "数据库", "docker", "部署", "上下文", "记忆", "harness", "mcp", "蒸馏", "量化"],
    "方案设计": ["如何设计", "怎么实现", "系统设计", "方案", "落地", "优化", "改进", "架构", "如果让你"],
    "行为经历": ["优点", "缺点", "失败", "冲突", "压力", "合作", "学习", "领导", "沟通", "复盘"],
    "求职动机": ["职业规划", "为什么选择", "为什么应聘", "离职", "转行", "未来五年", "岗位理解"]
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify(question: str) -> str:
    q = normalize(question)
    if any(word in q for word in INTENT_RULES["自我介绍"]):
        return "自我介绍"
    if any(word in q for word in ["如果让你", "如何设计", "系统设计", "你会怎么实现"]):
        return "方案设计"
    if any(word in q for word in ["你在", "你负责", "你做过", "项目中", "实习中", "遇到的困难", "最大困难", "最大挑战"]):
        return "项目深挖"
    scores = {name: sum(3 if word in q else 0 for word in words) for name, words in INTENT_RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "通用问答"


def token_set(text: str) -> set[str]:
    text = normalize(text)
    tokens = set(re.findall(r"[a-z0-9+#.-]{2,}", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for size in (2, 3, 4):
            tokens.update(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    return tokens


def _score(question: str, keywords: list[str], content: str) -> float:
    q = normalize(question)
    # Longer exact phrases carry more intent than generic words such as “模型” or “上下文”.
    exact = sum(max(8, len(normalize(keyword)) * 3) for keyword in keywords if normalize(keyword) in q)
    overlap = len(token_set(question) & token_set(content))
    return exact + min(overlap, 10)


def retrieve_facts(question: str, limit: int = 5) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in KNOWLEDGE["facts"]:
        content = " ".join([item["title"], *item["keywords"], *item["facts"]])
        score = _score(question, item["keywords"], content)
        if score >= 3:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def retrieve_topics(question: str, limit: int = 5) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in ALL_TOPICS:
        content = " ".join([item["title"], *item["keywords"], item["short_answer"], *item["key_points"], *item.get("deep_dive", [])])
        score = _score(question, item["keywords"], content)
        if score >= 3:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


retrieve = retrieve_facts
retrieve_deep = retrieve_topics


def select_story(question: str) -> dict[str, Any] | None:
    q = normalize(question)
    for story in KNOWLEDGE["story_bank"]:
        if any(trait.lower() in q for trait in story["traits"]):
            return story
    if any(word in q for word in ["困难", "失败", "稳定", "质量", "问题"]):
        return KNOWLEDGE["story_bank"][1]
    if any(word in q for word in ["学习", "转行", "专业", "成长"]):
        return KNOWLEDGE["story_bank"][0]
    return None


def local_outline(question: str, category: str, facts: list[dict[str, Any]], topics: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline fallback: useful evidence outline, explicitly not presented as a model answer."""
    story = select_story(question)
    if category == "自我介绍":
        answer = "我是浙大城市学院电子信息专业的2027届硕士。我的实践以Python、数据处理和AI应用交付为主：在研究院参与大模型安全、RAG与工具调用研究，在君同未来完成评测数据处理、模型接口调用和安全攻防平台原型开发，经历过从需求拆解、新模块实现到测试验证的完整流程。虽然我不是传统后端专业出身，但这些经历与内部业务系统、数据链路和Python服务开发具有较强迁移性，我也在重点补强FastAPI、PostgreSQL和Redis。"
        followups = ["你的经历为什么适合Python后端？", "你做过哪个从零到一的模块？", "你准备如何补齐后端技术差距？"]
    elif story and category in ("行为经历", "项目深挖"):
        answer = f"这段经历中最关键的问题是：{story['situation']} 我的处理是：{story['action']} 最终，{story['result']} 复盘来看，{story['reflection']}"
        followups = ["当时还有其他方案吗？", "你个人承担了哪一部分？", "如果重新做会怎么改进？"]
    elif category == "方案设计":
        framework = KNOWLEDGE["solution_frameworks"][0]
        related = "；".join(topic["short_answer"] for topic in topics[:2])
        answer = "参考提纲：" + "；".join(framework["points"]) + (f"。相关知识：{related}" if related else "")
        followups = topics[0]["followups"][:3] if topics else ["如何定义验收指标？", "异常情况如何处理？", "如何控制成本和风险？"]
    elif topics:
        topic = topics[0]
        answer = f"参考提纲：{topic['short_answer']} " + "；".join(topic["key_points"][:4])
        followups = topic["followups"][:3]
    elif facts:
        fact = facts[0]
        answer = f"参考提纲：可以结合“{fact['title']}”回答。" + "；".join(fact["facts"][:4])
        followups = ["你个人具体负责什么？", "最大的困难是什么？", "最终如何验证效果？"]
    else:
        answer = "当前本地知识库没有足够直接的材料。建议接入模型后，让模型先识别问题意图，再结合通用知识生成针对性回答；如果涉及个人经历，应明确说明没有直接经验并给出分析思路。"
        followups = ["你会如何拆解这个问题？", "你会如何验证方案？"]
    return {
        "mode": "outline", "category": category, "answer": answer, "followups": followups[:3],
        "risk": facts[0]["safe_claim"] if facts else topics[0]["boundary"] if topics else "没有资料支撑的经历不要编造。",
        "evidence_ids": [item["id"] for item in facts], "topic_ids": [item["id"] for item in topics]
    }


def build_messages(question: str, category: str, facts: list[dict[str, Any]], topics: list[dict[str, Any]], history: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    length = "100到160字" if mode == "short" else "160到260字"
    system = f"""你是一名通用AI面试回答助手。候选人是浙大城市学院电子信息专业2027届硕士，求职方向包括AI Agent应用、大模型应用与评测、Python后端和AI安全治理。当前正在针对一个岗位进行定向准备，但工具仍应能够回答其他通用面试问题。

你的第一目标是准确回答面试官当前的问题，而不是复述简历或知识卡片。请先在内部判断问题意图，再选择相关资料组织回答。

回答规则：
1. 开头第一句直接回答问题，不能绕题。
2. 个人经历只能使用“个人事实”中的内容；“技术知识”不能伪装成亲身经历。
3. 没有直接经验时使用“我的理解是”“如果让我设计”或“我会”这类表达。
4. 回答适合口述，长度{length}，优先形成一个连贯段落。
5. 行为题使用情境、行动、结果、反思的自然结构；技术题先讲结论，再讲原理和项目连接；方案题讲目标、方案、验证和风险。
6. 不编造数字、团队规模、商业化结果、发表或授权状态，也不夸大AI辅助生成代码的独立开发程度。
7. 生成2到3个紧贴本题的后续追问，不要重复当前问题。
8. 当前问题与目标岗位相关时，优先使用岗位匹配点组织回答；对FastAPI、PostgreSQL、Redis等缺少直接项目事实的技术，只能按技术理解和迁移方案回答。

只返回JSON，不要Markdown：
{{"category":"问题类型","answer":"可直接口述的完整回答","followups":["追问1","追问2"],"risk":"仅供备战模式查看的事实提醒","evidence_ids":["事实ID"],"topic_ids":["知识ID"]}}

候选人概况：{json.dumps(KNOWLEDGE['candidate'], ensure_ascii=False)}
当前目标岗位：{json.dumps(JOB_PROFILE['active_profile'], ensure_ascii=False)}
回答原则：{json.dumps(KNOWLEDGE['scope']['answer_principles'], ensure_ascii=False)}
全局事实边界：{json.dumps(KNOWLEDGE['global_boundaries'], ensure_ascii=False)}
本题个人事实：{json.dumps(facts, ensure_ascii=False)}
本题技术知识（含论文提炼资料）：{json.dumps(topics, ensure_ascii=False)}

论文回答约束：如果资料带有source或relationship字段，回答必须区分“论文提出/实验显示”和候选人个人职责；不得把共同作者经历表述为独立完成，也不得把模拟实验表述为生产部署。"""
    recent = history[-3:] if history else []
    context = "\n".join(f"上一问：{item.get('question', '')}\n回答摘要：{item.get('summary', '')}" for item in recent)
    user = f"问题类型：{category}\n面试官当前问题：{question}\n最近对话：{context or '无'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_model(payload: dict[str, Any], question: str, category: str, facts: list[dict[str, Any]], topics: list[dict[str, Any]]) -> dict[str, Any]:
    browser_config = payload.get("config") or {}
    config = {
        "apiBase": browser_config.get("apiBase") or SERVER_CONFIG["apiBase"],
        "apiKey": browser_config.get("apiKey") or SERVER_CONFIG["apiKey"],
        "model": browser_config.get("model") or SERVER_CONFIG["model"]
    }
    api_base = str(config.get("apiBase") or "").rstrip("/")
    api_key = str(config.get("apiKey") or "")
    model = str(config.get("model") or "")
    if not api_base or not api_key or not model:
        fallback = local_outline(question, category, facts, topics)
        fallback["warning"] = "当前未连接模型，显示的是知识库参考提纲；接入模型后会针对原问题组织完整回答。"
        return fallback
    parsed = urlparse(api_base)
    if parsed.scheme not in ("http", "https") or (parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost")):
        fallback = local_outline(question, category, facts, topics)
        fallback["warning"] = "API Base必须使用HTTPS；本地模型服务可使用127.0.0.1或localhost。"
        return fallback
    body = {
        "model": model,
        "messages": build_messages(question, category, facts, topics, payload.get("history") or [], payload.get("mode") or "normal"),
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": 900,
        "temperature": 0.2
    }
    request = urllib.request.Request(f"{api_base}/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        result = json.loads(match.group(0) if match else content)
        result["mode"] = "api"
        result.setdefault("category", category)
        result.setdefault("answer", "")
        result["followups"] = (result.get("followups") or [])[:3]
        result.setdefault("risk", "回答时保持真实，不补充未经证实的经历。")
        result.setdefault("evidence_ids", [item["id"] for item in facts])
        result.setdefault("topic_ids", [item["id"] for item in topics])
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fallback = local_outline(question, category, facts, topics)
        fallback["warning"] = f"模型暂不可用，已显示本地参考提纲：{type(exc).__name__}"
        return fallback


class Handler(BaseHTTPRequestHandler):
    server_version = "InterviewAgent/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/analyze" not in (fmt % args):
            print(fmt % args)

    def send_json(self, data: Any, status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            question = str(payload.get("question") or "").strip()
            if not question:
                self.send_json({"error": "请输入面试问题"}, HTTPStatus.BAD_REQUEST)
                return
            if len(question) > 2000:
                self.send_json({"error": "问题过长，请控制在2000字以内"}, HTTPStatus.BAD_REQUEST)
                return
            category = classify(question)
            facts = retrieve_facts(question)
            topics = retrieve_topics(question)
            answer = call_model(payload, question, category, facts, topics)
            self.send_json({"answer": answer, "evidence": facts, "topics": topics, "category": category})
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": "请求格式无效"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": f"处理失败：{type(exc).__name__}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self.send_json({"status": "ok", "facts": len(KNOWLEDGE["facts"]), "topics": len(ALL_TOPICS), "papers": len(PAPER_KNOWLEDGE["topics"]), "version": 4, "jobProfile": JOB_PROFILE["active_profile"]["title"], "modelConfigured": bool(SERVER_CONFIG["apiKey"]), "model": SERVER_CONFIG["model"] if SERVER_CONFIG["apiKey"] else ""})
            return
        if path == "/api/profile":
            self.send_json({"candidate": KNOWLEDGE["candidate"], "jobProfile": JOB_PROFILE["active_profile"], "scope": KNOWLEDGE["scope"], "boundaries": KNOWLEDGE["global_boundaries"], "facts": KNOWLEDGE["facts"], "topics": ALL_TOPICS, "paperCount": len(PAPER_KNOWLEDGE["topics"])})
            return
        files = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/voice.css": "voice.css", "/app.js": "app.js"}
        filename = files.get(path)
        if not filename:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        target = STATIC / filename
        content = target.read_bytes()
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith(("text/", "application/javascript")) else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"通用AI面试助手已启动：http://{HOST}:{PORT}")
    print(f"后端模型：{SERVER_CONFIG['model'] if SERVER_CONFIG['apiKey'] else '未配置'}。按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
