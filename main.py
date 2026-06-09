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
from astrbot.api import AstrBotConfig

from .database import InspirationDB

# ============================================================
# LLM Prompt
# ============================================================

INTENT_PARSE_SYSTEM = """你是 LGNB 聊天灵感插件的意图解析器。输出 JSON。

工具:
1. query_inspirations — 查灵感
   - start_date/end_date: YYYY-MM-DD (可选)
   - limit: 返回条数 (可选, 默认10)
2. generate_summary — 生成总结
   - start_date/end_date: YYYY-MM-DD (必填, 未指定默认最近7天至今)
3. get_status — 数据统计, 无参数
4. categorize_now — 手动归类 (管理员)
5. export_data — 导出数据 (管理员)
   - data_type: "all"/"range"
   - start_date/end_date: 仅range需要
   - format: "json"/"markdown"/"csv" (可选, 默认json)
6. analyze_chat — AI 对聊天发表看法
   - start_date/end_date (可选, 默认近7天)
   - topic: 分析角度 (可选)
7. search_messages — 搜关键词
   - keyword: 必填, 多个用逗号分隔
   - start_date/end_date (可选)
8. delete_data — 删除数据 (管理员)
   - start_date/end_date: 必填
9. unknown

输出: {"tool":"...","params":{...},"reasoning":"..."}"""


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

RE_DATE_RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*[~到至]\s*(\d{4}-\d{2}-\d{2})")


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
        ts = datetime.fromtimestamp(m["timestamp"]).strftime("%m-%d %H:%M")
        lines.append(f"[{ts}] {m['user_name']}: {m['content']}")
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
                ts = r.get("timestamp") or r.get("created_at", 0)
                ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
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
        ts = datetime.fromtimestamp(m.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
        w.writerow([ts, m.get("user_id", ""), m.get("user_name", ""), m.get("content", "")])
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
        self._scheduler_task: Optional[asyncio.Task] = None
        self._start_scheduler()

    # ========== 调度器 ==========

    def _start_scheduler(self):
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.ensure_future(self._daily_summary_loop())

    async def _daily_summary_loop(self):
        hour = self.config.get("daily_summary_hour", 20)
        minute = self.config.get("daily_summary_minute", 0)
        retention = self.config.get("data_retention_days", 0)
        while True:
            try:
                await asyncio.sleep(45)
                now = datetime.now()
                if now.hour == hour and now.minute == minute:
                    await self._trigger_daily_summaries()
                # 每天 3 点做一次过期清理
                if retention > 0 and now.hour == 3 and now.minute == 0:
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
        user_id = event.unified_msg_origin
        try:
            if self.context.is_admin(user_id):
                return True
        except Exception:
            pass
        sub_admins: list = self.config.get("sub_admins", [])
        sender_uid = event.get_sender_id() if hasattr(event, "get_sender_id") else user_id
        return sender_uid in sub_admins

    def _is_whitelisted(self, group_id: str) -> bool:
        wl: list = self.config.get("whitelist_groups", [])
        if not wl:
            return False
        if group_id in wl:
            return True
        for w in wl:
            if group_id.endswith(w) or w.endswith(group_id):
                return True
        return False

    def _query_scope(self) -> str:
        """用户级隔离配置: all / self_only"""
        return self.config.get("query_scope", "all")

    # ========== 消息处理 ==========

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        group_id = event.unified_msg_origin
        if not self._is_whitelisted(group_id):
            return
        content = event.message_str or ""
        if not content.strip():
            return
        user_id = event.get_sender_id() if hasattr(event, "get_sender_id") else ""
        user_name = event.get_sender_name() if hasattr(event, "get_sender_name") else ""
        # 获取 message_id 用于去重
        try:
            msg_id = str(event.message_obj.message_id) if event.message_obj else ""
        except Exception:
            msg_id = ""

        # 存储 (带 dedup)
        stored = self.db.store_message(
            group_id=group_id, group_name=group_id,
            user_id=user_id, user_name=user_name,
            content=content, message_id=msg_id,
        )

        # 阈值归类
        threshold = self.config.get("message_threshold", 50)
        if threshold > 0 and self.db.get_uncategorized_count(group_id) >= threshold:
            asyncio.ensure_future(self._auto_categorize(group_id))

        # @bot 交互
        if self._is_at_bot(event):
            if stored is not None:  # 不重复消息时才响应
                event.stop_event()
                # 即时反馈
                yield event.plain_result("正在思考，请稍候...")
                reply = await self._build_bot_reply(event)
                if reply:
                    chunks = _trim_reply(reply, self.config.get("max_reply_length", 0))
                    for chunk in chunks:
                        yield event.plain_result(chunk)

    def _is_at_bot(self, event: AstrMessageEvent) -> bool:
        content = event.message_str or ""
        # 检查消息链中是否包含 @ 消息段
        try:
            if event.message_obj and hasattr(event.message_obj, "message"):
                for c in event.message_obj.message:
                    t = c.get("type", "") if isinstance(c, dict) else getattr(c, "type", "")
                    if t and t.lower() == "at":
                        return True
        except Exception:
            pass
        # 关键词触发
        for kw in ["@bot", "@机器人", "/灵感", "/总结", "/状态", "/归类", "/lgnb", "/删除数据"]:
            if kw in content.lower():
                return True
        # 私聊自动响应
        try:
            if event.message_obj and hasattr(event.message_obj, "type"):
                from astrbot.api.event.filter import EventMessageType
                if event.message_obj.type == EventMessageType.PRIVATE_MESSAGE:
                    return True
        except Exception:
            pass
        return False

    async def _build_bot_reply(self, event: AstrMessageEvent) -> str:
        message = event.message_str or ""
        debug = self.config.get("debug_mode", False)

        intent = await self._parse_intent(event, message)
        if intent is None:
            return "意图解析失败，请稍后重试。"

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
        lt_str = datetime.fromtimestamp(lt["time"]).strftime("%Y-%m-%d %H:%M") if lt["time"] else ""
        return (f"数据统计\n消息: {s['total_messages']} (未归类: {s['uncategorized_messages']})\n"
                f"灵感: {s['total_inspirations']}\n总结: {s['total_summaries']}\n"
                f"上次归类: {lt_str} ({lt['trigger_type']}, 处理{lt['message_count']}条, 提取{lt['inspiration_count']}条)")

    # ========== 归类 ==========

    async def _auto_categorize(self, gid: str):
        async with self._summary_lock:
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
