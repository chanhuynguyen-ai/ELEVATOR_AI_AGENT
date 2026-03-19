from typing import Dict
from backend.text_utils import normalize_vi


class SafetyGuardrails:
    EMERGENCY_KEYWORDS = {
        "sos", "khan cap", "cuu ho", "mac ket", "ket trong thang", "nga", "ngat", "chay", "khoi"
    }
    INJECTION_KEYWORDS = {
        "ignore previous", "system prompt", "developer prompt", "reveal prompt", "bypass", "sudo", "rm -rf"
    }
    ALLOWED_TOOLS = {
        "employee_lookup",
        "kb_search",
        "get_elevator_status",
        "call_elevator",
        "general_llm",
    }

    def normalize(self, text: str) -> str:
        return normalize_vi(text)

    def precheck(self, text: str) -> Dict[str, str]:
        norm = self.normalize(text)
        if any(token in norm for token in self.INJECTION_KEYWORDS):
            return {
                "status": "blocked",
                "answer": "Yêu cầu này không hợp lệ. Sunybot chỉ hỗ trợ các tác vụ liên quan đến thang máy, bảo trì và thông tin nội bộ an toàn.",
                "intent": "blocked_request",
            }
        if any(token in norm for token in self.EMERGENCY_KEYWORDS):
            return {
                "status": "emergency",
                "answer": "Tôi đã chuyển sang chế độ hỗ trợ khẩn cấp. Hãy giữ bình tĩnh, nhấn nút SOS, không tự cạy cửa và chờ bộ phận kỹ thuật phản hồi.",
                "intent": "emergency_support",
            }
        return {"status": "ok", "intent": "unknown"}

    def allow_tool(self, tool_name: str) -> bool:
        return tool_name in self.ALLOWED_TOOLS
