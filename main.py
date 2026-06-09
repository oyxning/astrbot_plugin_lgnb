"""
聊天灵感提取插件 - 主逻辑
记录群聊消息、自动归类灵感、支持自然语言查询与总结

v1.1:
- 每日总结推送群聊
- 数据清理（管理员删除 + 自动过期）
- 消息去重 (message_id)
- FTS5 全文搜索 + 多关键词
- 导出 Markdown / CSV / JSON
- 回复超长截断分段
- "正在思考"即时反馈
- 用户级操作隔离
"""
import asyncio
import csv
import io
import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from astrbot.api.star import Star, Context
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse
from astrbot.api import AstrBotConfig

from .database import InspirationDB

# ============================================================
# LLM Prompt
# ============================================================

INTENT_PARSE_SYSTEM = """你是 LGNB 聊天灵感插件的路由器。**所有工具都是用来查询/操作本群数据库中已存储的聊天记录、灵感和总结，不是通用工具。**

关键判断：用户是想查询**本群储存的历史聊天数据**，还是在进行**普通 AI 对话**（问知识、闲聊、让 AI 帮忙写东西等）？

工具:
1. query_inspirations — 查询数据库中已提取的灵感记录
   - start_date/end_date: YYYY-MM-DD (可选)
   - limit: 返回条数 (可选, 默认10)
2. generate_summary — 基于数据库中的聊天记录生成时间段总结
   - start_date/end_date: YYYY-MM-DD (必填, 未指定默认最近7天至今)
3. get_status — 查看本群数据库中的数据统计, 无参数
4. categorize_now — 手动触发灵感归类 (管理员), 无参数
5. export_data — 导出数据库中的数据 (管理员)
   - data_type: "all"/"range"
   - start_date/end_date: 仅range需要
   - format: "json"/"markdown"/"csv" (可选, 默认json)
6. analyze_chat — AI 对数据库中某段聊天记录发表看法
   - start_date/end_date (可选, 默认近7天)
   - topic: 分析角度 (可选)
7. search_messages — 在数据库中搜索包含关键词的聊天记录
   - keyword: 必填, 多个用逗号分隔
   - start_date/end_date (可选)
8. delete_data — 删除数据库中的数据 (管理员)
   - start_date/end_date: 必填
9. unknown — 不属于以上任何场景：普通聊天、知识问答、让AI帮忙写作/总结外部内容等

输出: {"tool":"...","params":{...},"reasoning":"解释为什么选这个工具而非unknown"}

判 unknown 的规则（仅以下情况判 unknown）:
- 用户问的是通用知识（如"董路的天赋是什么""什么叫灵感"）→ unknown，这是默认 Agent 的知识问答
- 用户让 AI 帮自己整理/写作（如"帮我整理董路的访谈素材"）但没提群聊历史/数据库 → unknown
判 LGNB 工具的规则:
- 用户明确提到查看/回顾本群的聊天记录、灵感、数据、总结 → 匹配对应工具
- "看看数据""数据状态""有什么灵感""最近聊了什么""帮我总结" → 匹配工具（不要判 unknown）
- 能从上下文明显推断用户想查群聊数据时 → 匹配工具"""


CATEGORIZE_PROMPT = """从以下群聊中提取灵感。JSON数组输出, 每个: {"content":"...","category":"分类"}。
分类: 技术讨论/生活感悟/创意点子/问题解决方案/行业见解/趣事/其他。20-100字, 忽略闲聊。无灵感返回[]。只输出JSON。"""

DAILY_SUMMARY_PROMPT = """总结今天群聊精华:
1. 核心话题 (1-2句)
2. 灵感精华 (3-5条)
3. 共识与结论
4. 待跟进"""

RANGE_SUMMARY_PROMPT = """总结 {date_range} 群聊 ({message_count}条):
1. 时间段概述
2. 关键灵感汇总(按主题分组)
3. 重要结论与共识
4. 趋势或持续话题
5. 统计数据小计"""

CHAT_ANALYSIS_PROMPT = """群聊分析。{time_info} {topic_hint}

对话 ({message_count}条):
---
{conversation}
---

输出:
1. 整体评价 (2-3句)
2. 亮点/闪光点
3. 值得深挖的方向
4. 总结 (一句话)"""

KEYWORD_SEARCH_PROMPT = """搜索「{keyword}」的聊天记录。{time_info}

对话 ({message_count}条, {user_count}人):
---
{conversation}
---

输出:
1. 相关讨论概述
2. 关键发言摘录 (3-8条, 附发言人和时间)
3. 讨论脉络 (按时间线)
4. 结论/共识
5. 补充说明 (消息少时说明, 勿编造)"""

# ============================================================
# Helpers
# ============================================================


def _safe_ts(ts) -> float:
    """安全转换时间戳为 float，失败返回 0"""
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def _parse_date_range(params: dict, default_days: int = 7) -> tuple[float, float, str, str]:
    """解析日期范围，返回 (start_ts, end_ts, start_str, end_str)"""
    today = datetime.now()
    start_str = params.get("start_date", (today - timedelta(days=default_days)).strftime("%Y-%m-%d"))
    end_str = params.get("end_date", today.strftime("%Y-%m-%d"))
    try:
        st = datetime.strptime(start_str, "%Y-%m-%d").timestamp()
        et = (datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)).timestamp()
    except ValueError:
        raise ValueError("日期格式错误，请用 YYYY-MM-DD")
    if st > et:
        st, et = et, st
        start_str, end_str = end_str, start_str
    return st, et, start_str, end_str


def _format_conversation(messages: list[dict]) -> tuple[str, set]:
    """格式化消息列表为文本, 返回 (text, users_set)"""
    lines = []
    users = set()
    for m in messages:
        ts_f = _safe_ts(m.get("timestamp", 0))
        ts_str = datetime.fromtimestamp(ts_f).strftime("%m-%d %H:%M") if ts_f > 0 else "??-?? ??:??"
        lines.append(f"[{ts_str}] {m.get('user_name', '')}: {m.get('content', '')}")
        users.add(m.get("user_id", ""))
    return "\n".join(lines), users


def _trim_reply(text: str, max_len: int) -> list[str]:
    """将超长文本切分为多段"""
    if max_len <= 0 or len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def _json_to_markdown(data: dict) -> str:
    """将导出数据转为 Markdown"""
    lines = [f"# LGNB 数据导出", f"群组: {data.get('group_id')}",
             f"导出时间: {data.get('exported_at', '')}", ""]
    if "range_start" in data:
        lines.append(f"范围: {data['range_start']} ~ {data['range_end']}")
    lines.append("")

    for label, rows in [("消息", data.get("messages", [])), ("灵感", data.get("inspirations", [])),
                         ("总结", data.get("summaries", [])), ("归类日志", data.get("categorize_log", []))]:
        if rows:
            lines.append(f"## {label} ({len(rows)} 条)")
            for r in rows:
                ts = _safe_ts(r.get("timestamp") or r.get("created_at", 0))
                ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts > 0 else ""
                content = r.get("content", "")
                if label == "消息":
                    lines.append(f"- [{ts_str}] **{r.get('user_name','')}**: {content}")
                else:
                    lines.append(f"- [{ts_str}] {content}")
            lines.append("")
    return "\n".join(lines)


def _json_to_csv(data: dict) -> str:
    """将导出数据转为 CSV (仅消息)"""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["时间", "用户ID", "用户名", "内容"])
    for m in data.get("messages", []):
        ts = _safe_ts(m.get("timestamp", 0))
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts > 0 else ""
        w.writerow([ts_str, m.get("user_id", ""), m.get("user_name", ""), m.get("content", "")])
    return buf.getvalue()


# ============================================================
# Plugin
# ============================================================

class LGNBPlugin(Star):

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(plugin_dir, "data", "lgnb_data.db")
        self.db = InspirationDB(db_path)
        self.db.init()
        self._summary_lock = asyncio.Lock()
        self._categorize_locks: dict[str, asyncio.Lock] = {}  # per-group 归类锁
        self._scheduler_task: Optional[asyncio.Task] = None
        self._start_scheduler()

    def _get_categorize_lock(self, gid: str) -> asyncio.Lock:
        if gid not in self._categorize_locks:
            self._categorize_locks[gid] = asyncio.Lock()
        return self._categorize_locks[gid]

    # ========== 调度器 ==========

    def _start_scheduler(self):
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.ensure_future(self._daily_summary_loop())

    async def _daily_summary_loop(self):
        hour = self.config.get("daily_summary_hour", 20)
        minute = self.config.get("daily_summary_minute", 0)
        retention = self.config.get("data_retention_days", 0)
        _last_summary_date = ""  # 防当天重复触发
        while True:
            try:
                # 计算下一次触发时间（精确到秒）
                now = datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                delay = (next_run - now).total_seconds()
                # 最多睡 120 秒后重新计算一次防止偏差累积
                await asyncio.sleep(min(delay, 120))
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")
                if now.hour == hour and now.minute == minute and _last_summary_date != today_str:
                    _last_summary_date = today_str
                    await self._trigger_daily_summaries()
                # 过期清理 (每天 3:00-3:02)
                if retention > 0 and now.hour == 3 and 0 <= now.minute <= 2 and _last_summary_date != today_str:
                    await self._auto_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LGNB] 调度器异常: {e}")
                await asyncio.sleep(60)

    async def _trigger_daily_summaries(self):
        whitelist = self.config.get("whitelist_groups", [])
        for group_id in whitelist:
            try:
                await self._do_daily_summary(group_id)
            except Exception as e:
                print(f"[LGNB] 群 {group_id} 每日总结失败: {e}")

    async def _auto_cleanup(self):
        retention = self.config.get("data_retention_days", 0)
        if retention <= 0:
            return
        whitelist = self.config.get("whitelist_groups", [])
        for group_id in whitelist:
            try:
                result = self.db.cleanup_expired(group_id, retention)
                if result["deleted_messages"] > 0:
                    print(f"[LGNB] 清理 {group_id}: 删除 {result['deleted_messages']} 条过期消息")
            except Exception as e:
                print(f"[LGNB] 清理 {group_id} 失败: {e}")

    # ========== 权限 ==========

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        sender_id = ""
        try:
            sender_id = event.get_sender_id()
        except Exception:
            pass
        try:
            if sender_id and self.context.is_admin(sender_id):
                return True
        except Exception:
            pass
        sub_admins: list = self.config.get("sub_admins", [])
        return sender_id in sub_admins

    def _is_whitelisted(self, group_id: str) -> bool:
        wl: list = self.config.get("whitelist_groups", [])
        if not wl:
            return False
        if group_id in wl:
            return True
        # UMO 格式如 平台_群ID，用户可能只配了其中一部分
        for w in wl:
            if len(w) >= 4 and w in group_id:
                return True
        return False

    def _query_scope(self) -> str:
        """用户级隔离配置: all / self_only"""
        return self.config.get("query_scope", "all")

    # ========== 消息存储 ==========

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """仅负责存储消息 + 阈值归类，不做任何 @bot 响应"""
        group_id = event.unified_msg_origin
        if not self._is_whitelisted(group_id):
            return
        content = event.message_str or ""
        if not content.strip():
            return
        user_id = event.get_sender_id() if hasattr(event, "get_sender_id") else ""
        user_name = event.get_sender_name() if hasattr(event, "get_sender_name") else ""
        try:
            msg_id = str(event.message_obj.message_id) if event.message_obj else ""
        except Exception:
            msg_id = ""

        self.db.store_message(
            group_id=group_id, group_name=group_id,
            user_id=user_id, user_name=user_name,
            content=content, message_id=msg_id,
        )

        # 阈值归类
        threshold = self.config.get("message_threshold", 50)
        if threshold > 0 and self.db.get_uncategorized_count(group_id) >= threshold:
            asyncio.ensure_future(self._auto_categorize(group_id))

    # ========== AI 回复存储 ==========

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """捕获 AI 正常聊天回复并存入数据库（跳过 LGNB 内部 LLM 调用）"""
        if not self.config.get("save_ai_replies", True):
            return
        group_id = event.unified_msg_origin
        if not self._is_whitelisted(group_id):
            return
        text = resp.completion_text or ""
        if not text.strip():
            return
        # 跳过 LGNB 自身的 LLM 输出：工具回复 + LLM 原始格式
        skip_prefixes = [
            "归类完成", "暂无灵感", "数据统计", "未找到包含",
            "搜索「", "AI看法", "总结 (", "导出完成",
            "已删除", "意图解析失败", "未知工具",
            "权限不足", "未配置 LLM", "LLM 调用失败",
            "正在思考", "可用操作:", "请提供",
            "[", "{", "```",  # LLM 原始 JSON/表格/代码块
        ]
        head = text[:80].strip()
        for pat in skip_prefixes:
            if head.startswith(pat):
                return
        # 跳过 LGNB 功能回复特征：以 "1." 开头的结构化输出
        if head.startswith("1.") and len(head) < 80:
            return
        bot_uid = "AI_ASSISTANT"
        bot_name = "AI"
        try:
            if event.message_obj:
                bot_uid = event.message_obj.self_id or bot_uid
        except Exception:
            pass
        self.db.store_message(
            group_id=group_id, group_name=group_id,
            user_id=bot_uid, user_name=bot_name,
            content=text, message_id="",
        )

    # ========== @bot 交互 ==========

    @filter.on_waiting_llm_request()
    async def on_waiting_llm(self, event: AstrMessageEvent):
        """在 LLM 调用前判断是否需要 LGNB 介入"""
        group_id = event.unified_msg_origin
        if not self._is_whitelisted(group_id):
            return

        # 快速预检：只有 @bot / 私聊 / 斜杠指令才值得调用 LLM 做意图识别
        if not self._worth_intent_check(event):
            return

        # LLM 判断用户是否真的想用 LGNB
        message = event.message_str or ""

        # 快速关键词映射（明确指令跳过 LLM，更快更可靠）
        kw_map = self._keyword_to_tool(message)
        if kw_map:
            event.stop_event()
            await event.send(event.plain_result("正在思考，请稍候..."))
            reply = await self._execute_and_format(event, kw_map)
            if reply:
                for chunk in _trim_reply(reply, self.config.get("max_reply_length", 0)):
                    await event.send(event.plain_result(chunk))
            return

        intent = await self._parse_intent(event, message)
        if intent is None:
            return  # LLM 调用失败，放行给默认 Agent

        tool = intent.get("tool", "unknown")
        if tool == "unknown":
            return  # 用户只是普通聊天，放行给默认 Agent

        # 是 LGNB 指令 → 拦截
        event.stop_event()
        await event.send(event.plain_result("正在思考，请稍候..."))
        reply = await self._execute_and_format(event, intent)
        if reply:
            for chunk in _trim_reply(reply, self.config.get("max_reply_length", 0)):
                await event.send(event.plain_result(chunk))

    def _worth_intent_check(self, event: AstrMessageEvent) -> bool:
        """快速判断是否值得调用 LLM 做意图识别（避免每条消息都调用）"""
        # 1. 私聊 → 值得
        try:
            if event.message_obj and event.message_obj.type and str(event.message_obj.type).endswith("PRIVATE_MESSAGE"):
                return True
        except Exception:
            pass
        # 2. @bot → 值得
        try:
            if event.message_obj:
                for c in event.message_obj.message:
                    c_type = c.get("type", "") if isinstance(c, dict) else getattr(c, "type", "")
                    if c_type and c_type.lower() == "at":
                        return True
        except Exception:
            pass
        # 3. 斜杠指令 → 值得（明确的功能调用意图）
        content = (event.message_str or "").lower()
        for cmd in ["/灵感", "/总结", "/状态", "/归类", "/lgnb", "/删除数据", "/导出"]:
            if cmd in content:
                return True
        # 普通群聊消息 → 不拦截
        return False

    _KEYWORD_TOOL_MAP: dict[str, str] = {
        "数据": "get_status",
        "状态": "get_status",
        "数据量": "get_status",
        "归类": "categorize_now",
        "灵感": "query_inspirations",
        "导出": "export_data",
        "删除数据": "delete_data",
    }

    def _keyword_to_tool(self, message: str) -> Optional[dict]:
        """对于明确的功能指令，直接用关键词映射跳过 LLM 意图识别"""
        msg = message.strip().lower()
        if not msg:
            return None
        # 纯功能指令（短消息且匹配关键词）
        if len(msg) > 30:
            return None  # 太长的消息不是纯指令
        for kw, tool in self._KEYWORD_TOOL_MAP.items():
            if kw in msg:
                return {"tool": tool, "params": {}, "reasoning": f"关键词匹配: {kw}"}
        # 看数据/查状态 → get_status
        for pat in ["看数据", "查数据", "数据统计", "当前数据", "群数据"]:
            if pat in msg:
                return {"tool": "get_status", "params": {}, "reasoning": f"模式匹配: {pat}"}
        return None

    async def _execute_and_format(self, event: AstrMessageEvent, intent: dict) -> str:
        """根据已解析的 intent 执行工具并格式化回复"""
        debug = self.config.get("debug_mode", False)
        tool = intent.get("tool", "unknown")
        params = intent.get("params", {})
        reasoning = intent.get("reasoning", "")
        group_id = event.unified_msg_origin

        result = await self._execute_tool(tool, params, group_id, event)

        if debug:
            result += (f"\n\n[Debug] LLM理解: {reasoning}\n"
                       f"[Debug] 调用工具: {tool}\n"
                       f"[Debug] 参数: {json.dumps(params, ensure_ascii=False)}")
        return result

    # ========== LLM ==========

    async def _get_provider_id(self, event: AstrMessageEvent) -> str:
        cfg = self.config.get("llm_provider", "").strip()
        if cfg:
            return cfg
        try:
            return await self.context.get_current_chat_provider_id(event.unified_msg_origin)
        except Exception:
            return ""

    async def _call_llm(self, event: AstrMessageEvent, system_prompt: str, user_prompt: str) -> Optional[str]:
        try:
            pid = await self._get_provider_id(event)
            if not pid:
                return "[LGNB] 未配置 LLM 模型提供商"
            from astrbot.core.agent.message import UserMessageSegment, TextPart
            full = f"{system_prompt}\n\n{user_prompt}"
            resp = await self.context.llm_generate(
                chat_provider_id=pid,
                contexts=[UserMessageSegment(content=[TextPart(text=full)])],
            )
            return resp.completion_text if resp else ""
        except Exception as e:
            print(f"[LGNB] LLM 调用失败: {e}")
            return None

    async def _call_llm_config(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """无 event 的 LLM 调用"""
        try:
            pid = self.config.get("llm_provider", "").strip()
            if not pid:
                try:
                    pid = await self.context.get_current_chat_provider_id("")
                except Exception:
                    pass
            if not pid:
                return "[LGNB] 未配置 LLM 模型提供商"
            from astrbot.core.agent.message import UserMessageSegment, TextPart
            full = f"{system_prompt}\n\n{user_prompt}"
            resp = await self.context.llm_generate(
                chat_provider_id=pid,
                contexts=[UserMessageSegment(content=[TextPart(text=full)])],
            )
            return resp.completion_text if resp else ""
        except Exception as e:
            print(f"[LGNB] LLM 调用失败(cfg): {e}")
            return None

    async def _parse_intent(self, event: AstrMessageEvent, message: str) -> Optional[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = f"当前日期: {today}\n用户消息: {message}\n请判断意图并输出JSON。"
        result = await self._call_llm(event, INTENT_PARSE_SYSTEM, prompt)
        if not result:
            return None
        try:
            js = result.strip()
            if js.startswith("```"):
                js = re.sub(r"^```(?:json)?\s*", "", js)
                js = re.sub(r"\s*```$", "", js)
            return json.loads(js)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*"tool"[\s\S]*\}', result)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return {"tool": "unknown", "params": {}, "reasoning": "解析失败"}

    # ========== 工具路由 ==========

    async def _execute_tool(self, tool: str, params: dict, gid: str, event: AstrMessageEvent) -> str:
        if tool == "query_inspirations":
            return self._query_inspirations(gid, params, event)
        elif tool == "generate_summary":
            return await self._generate_range_summary(gid, params, event)
        elif tool == "get_status":
            return self._get_status(gid)
        elif tool == "categorize_now":
            if not self._is_admin(event):
                return "权限不足：需要管理员权限。"
            return await self._manual_categorize(gid, event)
        elif tool == "export_data":
            if not self._is_admin(event):
                return "权限不足：需要管理员权限。"
            return self._export_data(gid, params)
        elif tool == "analyze_chat":
            return await self._analyze_chat(gid, params, event)
        elif tool == "search_messages":
            return await self._search_messages(gid, params, event)
        elif tool == "delete_data":
            if not self._is_admin(event):
                return "权限不足：需要管理员权限。"
            return self._delete_data(gid, params)
        elif tool == "unknown":
            return (
                "可用操作:\n"
                "- 查灵感: 「最近有什么灵感」\n"
                "- 生成总结: 「总结最近7天」\n"
                "- AI看法: 「你怎么看上周聊天」\n"
                "- 搜索: 「帮我想想提到XX时聊了什么」\n"
                "- 统计: 「看看数据」\n"
                "- 管理员: 归类 / 导出 / 删除数据"
            )
        return f"未知工具: {tool}"

    # ========== 查询灵感 ==========

    def _query_inspirations(self, gid: str, params: dict, event: AstrMessageEvent) -> str:
        try:
            st, et, _, _ = _parse_date_range(params, default_days=30)
        except ValueError as e:
            return str(e)
        limit = min(int(params.get("limit", 10)), 50)

        uid = None
        if self._query_scope() == "self_only":
            uid = event.get_sender_id() if hasattr(event, "get_sender_id") else ""

        rows = self.db.query_inspirations(gid, start_time=st, end_time=et, limit=limit, user_id=uid)
        if not rows:
            return "暂无灵感记录。"
        lines = [f"灵感记录 ({len(rows)}条):"]
        for i, r in enumerate(rows, 1):
            ts = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d %H:%M")
            lines.append(f"{i}. [{r['category']}] {r['content']} ({ts})")
        return "\n".join(lines)

    # ========== 查询状态 ==========

    def _get_status(self, gid: str) -> str:
        s = self.db.get_data_stats(gid)
        lt = s["last_categorize"]
        lt_time = lt.get("time")
        if lt_time:
            try:
                ts = float(lt_time)
                lt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError, OSError):
                lt_str = str(lt_time)
        else:
            lt_str = ""
        return (f"数据统计\n消息: {s['total_messages']} (未归类: {s['uncategorized_messages']})\n"
                f"灵感: {s['total_inspirations']}\n总结: {s['total_summaries']}\n"
                f"上次归类: {lt_str} ({lt['trigger_type']}, 处理{lt['message_count']}条, 提取{lt['inspiration_count']}条)")

    # ========== 归类 ==========

    async def _auto_categorize(self, gid: str):
        lock = self._get_categorize_lock(gid)
        async with lock:
            if self.db.get_uncategorized_count(gid) < self.config.get("message_threshold", 50):
                return
            await self._do_categorize(gid, "auto")

    async def _manual_categorize(self, gid: str, event: AstrMessageEvent) -> str:
        count = self.db.get_uncategorized_count(gid)
        if count == 0:
            return "没有未归类的消息。"
        return await self._do_categorize(gid, "manual")

    async def _do_categorize(self, gid: str, trigger: str = "auto") -> str:
        limit = self.config.get("max_query_messages", 200)
        msgs = self.db.get_uncategorized_messages(gid, limit=limit)
        if not msgs:
            return "没有未归类的消息。"

        conv, _ = _format_conversation(msgs)
        result = await self._call_llm_config(CATEGORIZE_PROMPT, conv)
        if not result:
            return "LLM 调用失败。"
        # 检查 LLM 是否返回了错误而非灵感数据
        if result.startswith("[LGNB]") or result.startswith("LLM 调用失败"):
            return result
        insps = self._parse_inspiration_json(result)
        if not insps:
            return "本次未提取到灵感。"

        msg_ids = [m["id"] for m in msgs]
        ts_list = [m["timestamp"] for m in msgs]
        rs, re_ts = (min(ts_list), max(ts_list)) if ts_list else (None, None)

        for insp in insps:
            self.db.store_inspiration(gid, insp.get("content", ""), msg_ids[:20],
                                       insp.get("category", "未分类"), rs, re_ts)
        self.db.mark_messages_categorized(msg_ids)
        self.db.log_categorize(gid, len(msgs), len(insps), trigger)
        return f"归类完成！处理 {len(msgs)} 条消息，提取 {len(insps)} 条灵感。"

    @staticmethod
    def _parse_inspiration_json(text: str) -> list[dict]:
        try:
            t = text.strip()
            if t.startswith("```"):
                t = re.sub(r"^```(?:json)?\s*", "", t)
                t = re.sub(r"\s*```$", "", t)
            r = json.loads(t)
            return r if isinstance(r, list) else []
        except json.JSONDecodeError:
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                try:
                    r = json.loads(m.group())
                    return r if isinstance(r, list) else []
                except json.JSONDecodeError:
                    pass
            return []

    # ========== 每日总结 ==========

    async def _do_daily_summary(self, gid: str):
        """生成并推送每日总结"""
        if self.db.has_daily_summary_today(gid):
            return
        now = datetime.now()
        sod = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        eod = now.timestamp()
        msgs = self.db.get_messages_in_range(gid, sod, eod, limit=300)
        if not msgs:
            return
        conv, _ = _format_conversation(msgs)
        result = await self._call_llm_config(DAILY_SUMMARY_PROMPT, conv)
        if not result:
            return
        self.db.store_summary(gid, "daily", result, sod, eod)
        # 推送群聊
        if self.config.get("push_daily_summary", True):
            try:
                header = f"今日灵感总结\n{'='*20}\n"
                await self.context.send_message_by_umo(gid, header + result)
            except Exception as e:
                print(f"[LGNB] 推送每日总结失败 ({gid}): {e}")

    # ========== 时间段总结 ==========

    async def _generate_range_summary(self, gid: str, params: dict, event: AstrMessageEvent) -> str:
        try:
            st, et, s1, s2 = _parse_date_range(params)
        except ValueError as e:
            return str(e)
        limit = self.config.get("max_query_messages", 200)
        msgs = self.db.get_messages_in_range(gid, st, et, limit=limit)
        if not msgs:
            return f"{s1} ~ {s2} 无聊天记录。"
        conv, users = _format_conversation(msgs)
        sp = RANGE_SUMMARY_PROMPT.format(date_range=f"{s1} ~ {s2}", message_count=len(msgs))
        result = await self._call_llm(event, sp, conv)
        if not result:
            return "总结生成失败。"
        self.db.store_summary(gid, "range", result, st, et)
        return f"总结 ({s1}~{s2}, {len(msgs)}条, {len(users)}人)\n{'='*30}\n{result}"

    # ========== AI 看法 ==========

    async def _analyze_chat(self, gid: str, params: dict, event: AstrMessageEvent) -> str:
        try:
            st, et, s1, s2 = _parse_date_range(params)
        except ValueError as e:
            return str(e)
        limit = self.config.get("max_query_messages", 200)
        msgs = self.db.get_messages_in_range(gid, st, et, limit=limit)
        if not msgs:
            return f"{s1} ~ {s2} 无聊天记录。"
        conv, _ = _format_conversation(msgs)
        topic = params.get("topic", "")
        th = f"分析角度: {topic}\n请重点关注。" if topic else ""
        sp = CHAT_ANALYSIS_PROMPT.format(time_info=f"时段: {s1}~{s2}", topic_hint=th, message_count=len(msgs), conversation=conv)
        result = await self._call_llm(event, sp, conv)
        if not result:
            return "分析失败。"
        return f"AI看法 ({s1}~{s2}, {len(msgs)}条)\n{'='*30}\n{result}"

    # ========== 搜索消息 (FTS5 + 多关键词) ==========

    async def _search_messages(self, gid: str, params: dict, event: AstrMessageEvent) -> str:
        kw_raw = params.get("keyword", "").strip()
        if not kw_raw:
            return "请提供搜索关键词。"
        keywords = [k.strip() for k in kw_raw.replace("，", ",").split(",") if k.strip()]
        if not keywords:
            return "请提供搜索关键词。"

        st = et = None
        s1 = s2 = ""
        if params.get("start_date") or params.get("end_date"):
            try:
                st, et, s1, s2 = _parse_date_range(params)
            except ValueError as e:
                return str(e)

        limit = self.config.get("max_query_messages", 200)
        msgs = self.db.search_messages_fts(gid, keywords, st, et, limit)

        if not msgs:
            hint = f"({s1}~{s2})" if s1 else ""
            return f"未找到包含「{'/'.join(keywords)}」的聊天记录。{hint}"

        conv, users = _format_conversation(msgs)
        ti = f"时间: {s1}~{s2}" if s1 else ""
        sp = KEYWORD_SEARCH_PROMPT.format(keyword="/".join(keywords), time_info=ti,
                                            message_count=len(msgs), user_count=len(users), conversation=conv)
        result = await self._call_llm(event, sp, conv)
        if not result:
            return f"找到 {len(msgs)} 条「{'/'.join(keywords)}」相关消息，但分析失败。"
        return f"搜索「{'/'.join(keywords)}」({len(msgs)}条, {len(users)}人)\n{'='*30}\n{result}"

    # ========== 导出 ==========

    def _export_data(self, gid: str, params: dict) -> str:
        fmt = params.get("format", "json").lower()
        d_type = params.get("data_type", "all")
        s1 = params.get("start_date", "")
        s2 = params.get("end_date", "")

        pdir = os.path.dirname(os.path.abspath(__file__))
        edir = os.path.join(pdir, "data", "exports")
        os.makedirs(edir, exist_ok=True)

        if d_type == "range" and s1 and s2:
            try:
                st, et, s1, s2 = _parse_date_range(params)
            except ValueError as e:
                return str(e)
            data = self.db.export_range(gid, st, et)
            base = f"lgnb_{gid}_{s1}_{s2}"
        else:
            data = self.db.export_all(gid)
            base = f"lgnb_{gid}_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if fmt == "markdown":
            content = _json_to_markdown(data)
            fp = os.path.join(edir, f"{base}.md")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "csv":
            content = _json_to_csv(data)
            fp = os.path.join(edir, f"{base}.csv")
            with open(fp, "w", encoding="utf-8", newline="") as f:
                f.write(content)
        else:
            fp = os.path.join(edir, f"{base}.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        mc = len(data.get("messages", []))
        ic = len(data.get("inspirations", []))
        sc = len(data.get("summaries", []))
        return f"导出完成 ({fmt})\n消息: {mc}\n灵感: {ic}\n总结: {sc}\n文件: {fp}"

    # ========== 数据删除 ==========

    def _delete_data(self, gid: str, params: dict) -> str:
        try:
            st, et, s1, s2 = _parse_date_range(params)
        except ValueError as e:
            return str(e)
        count = self.db.delete_messages_range(gid, st, et)
        return f"已删除 {s1} ~ {s2} 的消息 ({count} 条)。灵感和总结未被删除。"
