import logging
import os
from typing import List, Optional

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")
REQUEST_CONNECT_TIMEOUT = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "3"))
REQUEST_READ_TIMEOUT = int(os.getenv("OLLAMA_READ_TIMEOUT", "45"))

FALLBACK_TEXT = "Sunybot hiện không thể trả lời câu hỏi này một cách đáng tin cậy."
LOGGER = logging.getLogger(__name__)


class OllamaService:
    def __init__(self):
        self._session = requests.Session()

    def _role_block(self, persona: str) -> str:
        persona = (persona or "customer_assistant").strip().lower()
        if persona in {"maintenance", "maintenance_console", "console", "operator"}:
            return (
                "Vai trò hiện tại: trợ lý kỹ thuật cho LLM Console bảo trì.\n"
                "- Ưu tiên sự kiện vận hành, cảnh báo, dữ liệu nội bộ và dữ liệu camera đã được trích xuất sẵn.\n"
                "- Khi chưa có dữ liệu camera hoặc log, phải nói rõ là chưa có dữ liệu thay vì suy diễn.\n"
                "- Nếu câu hỏi cần dữ liệu CV nhưng không có tool hoặc dữ liệu tham chiếu, hãy nêu thiếu hụt cụ thể.\n"
            )
        return (
            "Vai trò hiện tại: trợ lý cho khách hàng/người dùng cuối.\n"
            "- Chỉ trả lời về hướng dẫn sử dụng, an toàn, trạng thái công khai và hỗ trợ chung.\n"
            "- Không cung cấp dữ liệu camera, danh tính người, person_id, person_name hay lịch sử giám sát.\n"
            "- Nếu bị hỏi dữ liệu CV, hãy từ chối lịch sự và hướng sang kênh bảo trì.\n"
        )

    def _build_prompt(
        self,
        user_text: str,
        context_blocks: Optional[List[str]] = None,
        memory_summary: str = "",
        intent_hint: str = "general",
        persona: str = "customer_assistant",
    ) -> str:
        context_blocks = context_blocks or []
        context_text = "\n".join("- {0}".format(item) for item in context_blocks if item)
        return (
            "Bạn là Sunybot, trợ lý AI cho hệ thống thang máy thông minh.\n"
            "Nguyên tắc bắt buộc:\n"
            "1) Ưu tiên dữ liệu tham chiếu nếu đã được cung cấp.\n"
            "2) Không được bịa thêm sự kiện, số liệu hay quy trình không có trong dữ liệu.\n"
            "3) Nếu thiếu dữ liệu, hãy nói rõ là chưa đủ dữ liệu thay vì đoán.\n"
            "4) Trả lời ngắn gọn, đúng chuyên môn, bằng tiếng Việt.\n"
            "5) Nếu là câu chào/cảm ơn, trả lời lịch sự và ngắn gọn.\n"
            "\n"
            "{0}"
            "Tín hiệu ý định: {1}\n"
            "Tóm tắt hội thoại gần đây: {2}\n"
            "Dữ liệu tham chiếu:\n{3}\n"
            "Câu hỏi người dùng: {4}\n"
            "\n"
            "Yêu cầu định dạng: trả lời tối đa 6 câu. Nếu đang dựa trên dữ liệu tham chiếu, bám sát dữ liệu đó."
        ).format(
            self._role_block(persona),
            intent_hint or "general",
            memory_summary or "chưa có",
            context_text or "- không có dữ liệu KB",
            user_text,
        )

    def _sanitize_answer(self, answer: str) -> str:
        text = " ".join((answer or "").split())
        if not text:
            return FALLBACK_TEXT
        return text[:1200]

    def generate(
        self,
        user_text: str,
        context_blocks: Optional[List[str]] = None,
        memory_summary: str = "",
        intent_hint: str = "general",
        persona: str = "customer_assistant",
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
    ) -> str:
        url = "{0}/api/generate".format(OLLAMA_HOST.rstrip("/"))
        payload = {
            "model": LLM_MODEL,
            "prompt": self._build_prompt(
                user_text,
                context_blocks=context_blocks,
                memory_summary=memory_summary,
                intent_hint=intent_hint,
                persona=persona,
            ),
            "stream": False,
            "options": {
                "num_predict": 260,
                "num_ctx": 3072,
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=(connect_timeout or REQUEST_CONNECT_TIMEOUT, read_timeout or REQUEST_READ_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json() or {}
            answer = data.get("response")
            return self._sanitize_answer(answer)
        except Exception as exc:
            LOGGER.warning("Ollama generate failed: %s", exc)
            return FALLBACK_TEXT

    def chat(
        self,
        user_text: str,
        context_blocks: Optional[List[str]] = None,
        memory_summary: str = "",
        timeout_sec: int = REQUEST_READ_TIMEOUT,
        intent_hint: str = "general",
        persona: str = "customer_assistant",
    ) -> str:
        return self.generate(
            user_text=user_text,
            context_blocks=context_blocks,
            memory_summary=memory_summary,
            intent_hint=intent_hint,
            persona=persona,
            read_timeout=timeout_sec,
        )

    def healthcheck(self) -> bool:
        return self.healthcheck_details().get("ok", False)

    def healthcheck_details(self):
        try:
            response = self._session.get("{0}/api/tags".format(OLLAMA_HOST.rstrip("/")), timeout=REQUEST_CONNECT_TIMEOUT)
            response.raise_for_status()
            data = response.json() or {}
            models = data.get("models") or []
            names = [item.get("name") for item in models if item.get("name")]
            return {
                "ok": True,
                "host": OLLAMA_HOST,
                "model": LLM_MODEL,
                "model_available": LLM_MODEL in names if names else None,
                "models": names[:10],
            }
        except Exception as exc:
            return {
                "ok": False,
                "host": OLLAMA_HOST,
                "model": LLM_MODEL,
                "error": str(exc),
            }
