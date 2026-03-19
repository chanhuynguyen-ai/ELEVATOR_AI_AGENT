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

    def _build_prompt(
        self,
        user_text: str,
        context_blocks: Optional[List[str]] = None,
        memory_summary: str = "",
        intent_hint: str = "general",
    ) -> str:
        context_blocks = context_blocks or []
        context_text = "\n".join("- {0}".format(item) for item in context_blocks if item)
        return (
            "Bạn là Sunybot, trợ lý AI cho hệ thống thang máy thông minh.\n"
            "Nguyên tắc bắt buộc:\n"
            "1) Ưu tiên dữ liệu tham chiếu nếu đã được cung cấp.\n"
            "2) Không được bịa thêm sự kiện, số liệu hay quy trình không có trong dữ liệu.\n"
            "3) Nếu thiếu dữ liệu, hãy nói rõ là chưa đủ dữ liệu thay vì đoán.\n"
            "4) Chỉ hỗ trợ các chủ đề: thang máy, vận hành, bảo trì, an toàn, nhân viên nội bộ liên quan.\n"
            "5) Với lời chào/cảm ơn, trả lời lịch sự và ngắn gọn.\n"
            "\n"
            "Tín hiệu ý định: {0}\n"
            "Tóm tắt hội thoại gần đây: {1}\n"
            "Dữ liệu tham chiếu:\n{2}\n"
            "Câu hỏi người dùng: {3}\n"
            "\n"
            "Hãy trả lời bằng tiếng Việt, rõ ràng, tối đa 5 câu. "
            "Nếu đang dựa trên dữ liệu tham chiếu, bám sát dữ liệu đó."
        ).format(intent_hint or "general", memory_summary or "chưa có", context_text or "- không có dữ liệu KB", user_text)

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
            ),
            "stream": False,
            "options": {
                "num_predict": 220,
                "num_ctx": 2048,
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
    ) -> str:
        return self.generate(
            user_text=user_text,
            context_blocks=context_blocks,
            memory_summary=memory_summary,
            intent_hint=intent_hint,
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
                "model_available": LLM_MODEL in names if names else None,
                "models": names[:10],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }
