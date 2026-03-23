import json
import re
from typing import Any, Dict, Optional

from backend.agent import AgentRuntime
from backend.agent.tool_registry import ToolRegistry
from backend.embedding_service import EmbeddingService
from backend.ollama_service import OllamaService
from backend.semantic_matcher import SemanticMatcher
from config.db_config import db


CAMERA_KEYWORDS = (
    "camera", "cam", "cv", "nguoi nga", "te nga", "fall", "lying", "nam",
    "dung", "ngoi", "person_id", "person_name", "cam_id", "giam sat", "nhan dien",
)
STATUS_KEYWORDS = (
    "trang thai", "thang may", "elevator status", "qua tai", "overload", "door",
    "cua", "tang hien tai", "floor", "status",
)
GREETING_PATTERNS = (
    r"^\s*(hi|hello|xin chao|chao|helo|hey)\b",
    r"^\s*(cam on|thanks|thank you)\b",
)


class ChatbotEngine:
    def __init__(
        self,
        matcher: Optional[SemanticMatcher] = None,
        embedder: Optional[EmbeddingService] = None,
        ollama: Optional[OllamaService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        agent: Optional[AgentRuntime] = None,
    ):
        self.matcher = matcher or SemanticMatcher()
        self.matcher.load_from_db()
        self.embedder = embedder or EmbeddingService()
        self.ollama = ollama or OllamaService()
        self.tool_registry = tool_registry or ToolRegistry(
            matcher=self.matcher,
            embedder=self.embedder,
            ollama=self.ollama,
        )
        self.agent = agent or AgentRuntime(tool_registry=self.tool_registry)

    def reload_knowledge(self) -> Dict[str, Any]:
        self.matcher.load_from_db()
        return {
            "ok": True,
            "matcher_items": self.matcher.item_count,
        }

    def _normalize_scope(self, scope: Optional[str], persona: Optional[str]) -> str:
        raw = (scope or persona or "customer").strip().lower()
        if raw in {"maintenance", "maint", "console", "operator", "admin"}:
            return "maintenance"
        return "customer"

    def _default_persona(self, scope: str, persona: Optional[str]) -> str:
        if persona:
            return str(persona).strip().lower()
        return "maintenance_console" if scope == "maintenance" else "customer_assistant"

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _is_greeting(self, normalized_text: str) -> bool:
        return any(re.search(pattern, normalized_text) for pattern in GREETING_PATTERNS)

    def _looks_like_camera_query(self, normalized_text: str) -> bool:
        return any(keyword in normalized_text for keyword in CAMERA_KEYWORDS)

    def _looks_like_status_query(self, normalized_text: str) -> bool:
        return any(keyword in normalized_text for keyword in STATUS_KEYWORDS)

    def _make_result(
        self,
        answer: str,
        source: str,
        *,
        intent: str,
        confidence: float,
        session_id: Optional[str],
        scope: str,
        persona: str,
        query_type: str,
        tool_trace: Optional[list] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "source": source,
            "intent": intent,
            "confidence": confidence,
            "session_id": session_id,
            "scope": scope,
            "persona": persona,
            "query_type": query_type,
            "tool_trace": tool_trace or [],
        }

    def _handle_empty_message(self, session_id: Optional[str], scope: str, persona: str) -> Dict[str, Any]:
        return self._make_result(
            "Bạn hãy nhập câu hỏi rõ hơn. Ví dụ: trạng thái thang máy hiện tại hoặc hướng dẫn sử dụng thang máy.",
            "RULE",
            intent="empty_input",
            confidence=1.0,
            session_id=session_id,
            scope=scope,
            persona=persona,
            query_type="guardrail",
        )

    def _handle_greeting(self, session_id: Optional[str], scope: str, persona: str) -> Dict[str, Any]:
        prefix = "Sunybot bảo trì" if scope == "maintenance" else "Sunybot"
        return self._make_result(
            f"Xin chào, tôi là {prefix}. Bạn cần hỗ trợ về trạng thái thang máy, hướng dẫn sử dụng hay kiểm tra vận hành?",
            "RULE",
            intent="greeting",
            confidence=0.99,
            session_id=session_id,
            scope=scope,
            persona=persona,
            query_type="small_talk",
        )

    def _handle_scope_guard(self, session_id: Optional[str], scope: str, persona: str) -> Dict[str, Any]:
        return self._make_result(
            "Kênh chat khách hàng không được truy cập dữ liệu camera hoặc dữ liệu nhận diện người. Hãy dùng LLM Console bảo trì để hỏi các sự kiện CV và cảnh báo an toàn.",
            "POLICY",
            intent="cv_access_denied",
            confidence=1.0,
            session_id=session_id,
            scope=scope,
            persona=persona,
            query_type="policy_guard",
        )

    def _handle_status_shortcut(self, session_id: Optional[str], scope: str, persona: str) -> Optional[Dict[str, Any]]:
        try:
            status = self.get_elevator_status(elevator_id=1) or {}
        except Exception:
            return None

        answer = (
            "Trạng thái hiện tại của thang máy: tầng {floor}, hướng {direction}, cửa {door}, "
            "số người {people_count}, quá tải {overload}, trạng thái hệ thống {status}."
        ).format(
            floor=status.get("floor", "?"),
            direction=status.get("direction", "UNKNOWN"),
            door=status.get("door", "UNKNOWN"),
            people_count=status.get("people_count", "?"),
            overload="có" if status.get("overload") else "không",
            status=status.get("status", "UNKNOWN"),
        )
        return self._make_result(
            answer,
            "TOOL:ELEVATOR_STATUS",
            intent="elevator_status",
            confidence=0.9,
            session_id=session_id,
            scope=scope,
            persona=persona,
            query_type="status",
            tool_trace=[{"tool": "get_elevator_status", "ok": True, "elevator_id": 1}],
        )

    def log_chat(self, result: Dict[str, Any], question: str) -> bool:
        trace_json = json.dumps(result.get("tool_trace", []), ensure_ascii=False)
        payload = (
            result.get("session_id"),
            question,
            result.get("intent"),
            float(result.get("confidence") or 0.0),
            result.get("source"),
            (result.get("answer") or "")[:250],
            trace_json,
            int(len(result.get("tool_trace", []))),
        )
        try:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_logs(
                            session_id, question, intent_name, confidence, source,
                            answer_preview, tool_trace_json, tool_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        payload,
                    )
            return True
        except Exception:
            return False

    def handle(
        self,
        user_text: str,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_text = self._normalize_text(user_text)
        normalized_scope = self._normalize_scope(scope, persona)
        normalized_persona = self._default_persona(normalized_scope, persona)

        if not normalized_text:
            result = self._handle_empty_message(session_id, normalized_scope, normalized_persona)
            self.log_chat(result, user_text)
            return result

        if self._is_greeting(normalized_text):
            result = self._handle_greeting(session_id, normalized_scope, normalized_persona)
            self.log_chat(result, user_text)
            return result

        if normalized_scope == "customer" and self._looks_like_camera_query(normalized_text):
            result = self._handle_scope_guard(session_id, normalized_scope, normalized_persona)
            self.log_chat(result, user_text)
            return result

        if self._looks_like_status_query(normalized_text):
            status_result = self._handle_status_shortcut(session_id, normalized_scope, normalized_persona)
            if status_result:
                self.log_chat(status_result, user_text)
                return status_result

        result = self.agent.run(user_text, session_id=session_id)
        result["session_id"] = result.get("session_id") or session_id
        result["scope"] = normalized_scope
        result["persona"] = normalized_persona
        result["query_type"] = (
            "maintenance_cv" if self._looks_like_camera_query(normalized_text) else "general"
        )
        result.setdefault("tool_trace", [])
        self.log_chat(result, user_text)
        return result

    def handle_request(self, req) -> Dict[str, Any]:
        if isinstance(req, dict):
            message = req.get("message") or req.get("question") or ""
            session_id = req.get("session_id")
            scope = req.get("scope") or req.get("role")
            persona = req.get("persona")
        else:
            message = getattr(req, "message", "") or getattr(req, "question", "")
            session_id = getattr(req, "session_id", None)
            scope = getattr(req, "scope", None) or getattr(req, "role", None)
            persona = getattr(req, "persona", None)
        return self.handle(message, session_id=session_id, scope=scope, persona=persona)

    def get_elevator_status(self, elevator_id: int = 1) -> Dict[str, Any]:
        return self.tool_registry.tool_get_elevator_status(elevator_id=elevator_id)

    def call_elevator(
        self,
        elevator_id: int = 1,
        from_floor: Optional[int] = None,
        target_floor: Optional[int] = None,
        direction: str = "up",
    ) -> Dict[str, Any]:
        return self.tool_registry.tool_call_elevator(
            elevator_id=elevator_id,
            from_floor=from_floor,
            target_floor=target_floor,
            direction=direction,
        )

    def healthcheck(self) -> Dict[str, Any]:
        db_info = db.test_connection_details()
        ollama_info = self.ollama.healthcheck_details()
        return {
            "db_ok": db_info.get("ok", False),
            "db_backend": "postgresql",
            "db_info": db_info,
            "matcher_items": self.matcher.item_count,
            "ollama_ok": ollama_info.get("ok", False),
            "ollama_info": ollama_info,
            "tools": self.tool_registry.available_tools(),
            "engine_mode": "router_plus_agent",
            "scopes": ["customer", "maintenance"],
        }
