import time
import uuid
from typing import Any, Dict, List, Optional

from backend.embedding_service import EmbeddingService
from backend.employee_service import (
    find_employee_by_code,
    find_employee_by_name,
    format_employee_answer,
    is_employee_code,
)
from backend.ollama_service import FALLBACK_TEXT, OllamaService
from backend.semantic_matcher import SemanticMatcher


class ToolRegistry:
    def __init__(
        self,
        matcher: Optional[SemanticMatcher] = None,
        embedder: Optional[EmbeddingService] = None,
        ollama: Optional[OllamaService] = None,
    ):
        self.matcher = matcher or SemanticMatcher()
        self.matcher.load_from_db()
        self.embedder = embedder or EmbeddingService()
        self.ollama = ollama or OllamaService()
        self._tools = {
            "employee_lookup": self.tool_employee_lookup,
            "kb_search": self.tool_kb_search,
            "get_elevator_status": self.tool_get_elevator_status,
            "call_elevator": self.tool_call_elevator,
            "general_llm": self.tool_general_llm,
        }

    def available_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def run(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        handler = self._tools.get(tool_name)
        if not handler:
            raise ValueError("Tool không tồn tại: {0}".format(tool_name))
        return handler(**(args or {}))

    def tool_employee_lookup(self, query: str) -> Dict[str, Any]:
        query = (query or "").strip()
        emp = None
        if is_employee_code(query):
            emp = find_employee_by_code(query)
        if not emp and query:
            emp = find_employee_by_name(query)
        if not emp:
            return {
                "ok": False,
                "source": "EMPLOYEE",
                "message": "Không tìm thấy nhân viên phù hợp. Bạn hãy thử nhập mã nhân viên hoặc họ tên đầy đủ hơn.",
            }
        return {
            "ok": True,
            "source": "EMPLOYEE",
            "employee": emp,
            "message": format_employee_answer(emp),
        }

    def tool_kb_search(self, query: str, top_k: int = 4, threshold: float = 0.72) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {
                "ok": False,
                "source": "KB",
                "matches": [],
                "passages": [],
                "citations": [],
                "message": "Thiếu câu hỏi để tìm trong knowledge base.",
            }

        user_emb = self.embedder.embed(query, task="query")
        results = self.matcher.search(user_text=query, user_embedding=user_emb, top_k=max(1, int(top_k)))
        dynamic_threshold = threshold
        if len(query.split()) <= 4:
            dynamic_threshold = min(dynamic_threshold, 0.68)

        filtered = [item for item in results if float(item.get("confidence") or 0.0) >= dynamic_threshold]
        if not filtered and results:
            filtered = results[:1]

        passages = []
        citations = []
        for item in filtered:
            answer_text = item.get("answer_text", "")
            prompt_text = item.get("prompt_text", "")
            passages.append(answer_text)
            citations.append(
                {
                    "source": "intent:{0}".format(item.get("intent_name", "unknown")),
                    "content": "Q: {0} | A: {1}".format(prompt_text, answer_text)[:600],
                    "score": round(float(item.get("confidence") or 0.0), 4),
                }
            )

        return {
            "ok": bool(filtered),
            "source": "KB",
            "matches": filtered,
            "passages": passages,
            "citations": citations,
            "message": passages[0] if passages else "Không tìm thấy tri thức phù hợp trong database.",
            "retrieval_mode": filtered[0].get("retrieval_mode") if filtered else None,
        }

    def tool_get_elevator_status(self, elevator_id: int = 1) -> Dict[str, Any]:
        status = {
            "elevator_id": int(elevator_id),
            "floor": 5,
            "direction": "UP",
            "door": "CLOSED",
            "people_count": 4,
            "overload": False,
            "status": "NORMAL",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "SIMULATED",
        }
        return {
            "ok": True,
            "source": "ELEVATOR_STATUS",
            "status_data": status,
            "message": (
                "[Mô phỏng] Thang máy {0} đang ở tầng {1}, hướng {2}, cửa {3}, số người {4}, quá tải: {5}."
            ).format(
                status["elevator_id"],
                status["floor"],
                status["direction"],
                status["door"],
                status["people_count"],
                "có" if status["overload"] else "không",
            ),
        }

    def tool_call_elevator(
        self,
        elevator_id: int = 1,
        from_floor: Optional[int] = None,
        target_floor: Optional[int] = None,
        direction: str = "up",
    ) -> Dict[str, Any]:
        if from_floor is None:
            return {
                "ok": False,
                "source": "COMMAND",
                "message": "Bạn cần chỉ rõ tầng gọi thang, ví dụ: gọi thang tại tầng 3.",
            }

        from_floor = int(from_floor)
        target_floor = int(target_floor) if target_floor is not None else None
        eta = max(8, abs(from_floor - 5) * 4)
        command_id = uuid.uuid4().hex[:10]
        suffix = ""
        if target_floor is not None:
            suffix = " Mục tiêu dự kiến là tầng {0}.".format(target_floor)

        return {
            "ok": True,
            "source": "COMMAND",
            "command": {
                "command_id": command_id,
                "elevator_id": int(elevator_id),
                "from_floor": from_floor,
                "target_floor": target_floor,
                "direction": direction,
                "eta_seconds": eta,
                "mode": "SIMULATED",
            },
            "message": (
                "[Mô phỏng] Đã tạo lệnh gọi thang máy {0} tại tầng {1} theo hướng {2}. "
                "ETA dự kiến khoảng {3} giây.{4}"
            ).format(int(elevator_id), from_floor, direction, eta, suffix),
        }

    def tool_general_llm(
        self,
        query: str,
        context_blocks: Optional[List[str]] = None,
        memory_summary: str = "",
        intent_hint: str = "general",
    ) -> Dict[str, Any]:
        answer = self.ollama.chat(
            query,
            context_blocks=context_blocks or [],
            memory_summary=memory_summary,
            intent_hint=intent_hint,
        )
        if answer == FALLBACK_TEXT:
            return {
                "ok": False,
                "source": "LLM",
                "message": FALLBACK_TEXT,
            }
        return {
            "ok": True,
            "source": "LLM",
            "message": answer,
        }
