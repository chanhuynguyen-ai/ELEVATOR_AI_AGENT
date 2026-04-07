import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from backend.embedding_service import EmbeddingService
from backend.employee_service import (
    find_employee_by_code,
    find_employee_by_name,
    format_employee_answer,
    is_employee_code,
)
from backend.ollama_service import FALLBACK_TEXT, OllamaService
from backend.semantic_matcher import SemanticMatcher

try:
    import psycopg
    from psycopg.rows import dict_row as psycopg3_dict_row
except Exception:
    psycopg = None
    psycopg3_dict_row = None

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None


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

        self.pg_host = os.getenv("PGHOST", os.getenv("DB_HOST", "127.0.0.1"))
        self.pg_port = int(os.getenv("PGPORT", os.getenv("DB_PORT", "5432")))
        self.pg_user = os.getenv("PGUSER", os.getenv("DB_USER", "elevator_ai"))
        self.pg_password = os.getenv("PGPASSWORD", os.getenv("DB_PASSWORD", "elevator123"))
        self.cv_db_name = os.getenv("ELEVATOR_CV_DB_NAME", "elevator_cv")

        self._tools = {
            "employee_lookup": self.tool_employee_lookup,
            "kb_search": self.tool_kb_search,
            "get_elevator_status": self.tool_get_elevator_status,
            "call_elevator": self.tool_call_elevator,
            "get_cv_status": self.tool_get_cv_status,
            "get_recent_cv_events": self.tool_get_recent_cv_events,
            "get_today_fall_count": self.tool_get_today_fall_count,
            "get_peak_hour": self.tool_get_peak_hour,
            "get_daily_density": self.tool_get_daily_density,
            "get_latest_person_seen": self.tool_get_latest_person_seen,
            "answer_cv_query": self.tool_answer_cv_query,
            "general_llm": self.tool_general_llm,
        }

    def available_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def run(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        handler = self._tools.get(tool_name)
        if not handler:
            raise ValueError("Tool không tồn tại: {0}".format(tool_name))
        return handler(**(args or {}))

    @property
    def cv_db_available(self) -> bool:
        return psycopg is not None or psycopg2 is not None

    @contextmanager
    def _cv_connection(self):
        if not self.cv_db_available:
            raise RuntimeError("Thiếu PostgreSQL driver (psycopg/psycopg2).")
        conn = None
        try:
            if psycopg is not None:
                conn = psycopg.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    user=self.pg_user,
                    password=self.pg_password,
                    dbname=self.cv_db_name,
                    autocommit=False,
                )
            else:
                conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    user=self.pg_user,
                    password=self.pg_password,
                    dbname=self.cv_db_name,
                )
            yield conn
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _fetch_all(self, conn, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        if psycopg is not None and conn.__class__.__module__.startswith("psycopg"):
            with conn.cursor(row_factory=psycopg3_dict_row) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def _fetch_one(self, conn, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        rows = self._fetch_all(conn, sql, params)
        return rows[0] if rows else None

    def _safe_json(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

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

    def tool_get_cv_status(self) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {
                "ok": False,
                "source": "CV_DB",
                "message": "Chưa có PostgreSQL driver nên chưa đọc được elevator_cv.",
            }
        try:
            with self._cv_connection() as conn:
                occ = self._fetch_one(
                    conn,
                    """
                    SELECT cam_id, sample_ts, people_count, unknown_count, lying_count, fall_count, extra
                    FROM camera_occupancy_samples
                    ORDER BY sample_ts DESC
                    LIMIT 1
                    """,
                ) or {}
                evt = self._fetch_one(
                    conn,
                    """
                    SELECT event_ts, cam_id, event_type, person_name, confidence, posture
                    FROM camera_events
                    ORDER BY event_ts DESC
                    LIMIT 1
                    """,
                ) or {}
            message = (
                "CV hiện tại: camera {cam_id}, số người {people_count}, người chưa nhận diện {unknown_count}, "
                "nằm {lying_count}, té ngã {fall_count}."
            ).format(
                cam_id=occ.get("cam_id", "cam_01"),
                people_count=occ.get("people_count", 0),
                unknown_count=occ.get("unknown_count", 0),
                lying_count=occ.get("lying_count", 0),
                fall_count=occ.get("fall_count", 0),
            )
            if evt:
                message += " Event gần nhất: {event_type} lúc {event_ts}.".format(
                    event_type=evt.get("event_type", "UNKNOWN"),
                    event_ts=evt.get("event_ts", "?"),
                )
            return {
                "ok": True,
                "source": "CV_DB",
                "status_data": {"occupancy": occ, "latest_event": evt},
                "message": message,
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "message": f"Không đọc được CV DB: {exc}"}

    def tool_get_recent_cv_events(
        self,
        limit: int = 5,
        cam_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {"ok": False, "source": "CV_DB", "events": [], "message": "Thiếu driver PostgreSQL."}
        try:
            sql = "SELECT event_ts, cam_id, event_type, person_name, people_count, confidence, posture, extra FROM camera_events WHERE 1=1"
            params: List[Any] = []
            if cam_id:
                sql += " AND cam_id = %s"
                params.append(cam_id)
            if event_type:
                sql += " AND event_type = %s"
                params.append(event_type.upper())
            sql += " ORDER BY event_ts DESC LIMIT %s"
            params.append(max(1, int(limit)))
            with self._cv_connection() as conn:
                rows = self._fetch_all(conn, sql, tuple(params))
            for row in rows:
                row["extra"] = self._safe_json(row.get("extra"))
            if not rows:
                return {"ok": True, "source": "CV_DB", "events": [], "message": "Chưa có sự kiện CV phù hợp."}
            bullets = []
            for row in rows[:5]:
                bullets.append(
                    "- {event_ts}: {event_type} | cam={cam_id} | person={person_name}".format(
                        event_ts=row.get("event_ts", "?"),
                        event_type=row.get("event_type", "UNKNOWN"),
                        cam_id=row.get("cam_id", "cam_01"),
                        person_name=row.get("person_name") or "unknown",
                    )
                )
            return {
                "ok": True,
                "source": "CV_DB",
                "events": rows,
                "message": "Sự kiện CV gần nhất:\n" + "\n".join(bullets),
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "events": [], "message": f"Lỗi đọc camera_events: {exc}"}

    def tool_get_today_fall_count(self, cam_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {"ok": False, "source": "CV_DB", "message": "Thiếu driver PostgreSQL."}
        try:
            sql = "SELECT COUNT(*) AS total FROM camera_events WHERE event_type = 'FALL' AND event_ts::date = CURRENT_DATE"
            params: List[Any] = []
            if cam_id:
                sql += " AND cam_id = %s"
                params.append(cam_id)
            with self._cv_connection() as conn:
                row = self._fetch_one(conn, sql, tuple(params)) or {"total": 0}
            total = int(row.get("total", 0))
            label = f" ở {cam_id}" if cam_id else ""
            return {
                "ok": True,
                "source": "CV_DB",
                "total": total,
                "message": f"Hôm nay có {total} sự kiện té ngã{label}.",
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "message": f"Lỗi đếm sự kiện FALL: {exc}"}

    def tool_get_peak_hour(self, cam_id: str = "cam_01", days: int = 1) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {"ok": False, "source": "CV_DB", "message": "Thiếu driver PostgreSQL."}
        try:
            with self._cv_connection() as conn:
                row = self._fetch_one(
                    conn,
                    """
                    SELECT EXTRACT(HOUR FROM sample_ts) AS hour_slot,
                           AVG(people_count)::float AS avg_people,
                           MAX(people_count) AS peak_people
                    FROM camera_occupancy_samples
                    WHERE cam_id = %s AND sample_ts >= NOW() - (%s || ' day')::interval
                    GROUP BY hour_slot
                    ORDER BY avg_people DESC, peak_people DESC
                    LIMIT 1
                    """,
                    (cam_id, int(days)),
                )
            if not row:
                return {"ok": True, "source": "CV_DB", "message": f"Chưa có dữ liệu occupancy cho {cam_id}."}
            return {
                "ok": True,
                "source": "CV_DB",
                "data": row,
                "message": (
                    "Khung giờ đông nhất của {cam_id} trong {days} ngày gần đây là khoảng {hour}:00, "
                    "trung bình {avg:.2f} người, đỉnh {peak} người."
                ).format(
                    cam_id=cam_id,
                    days=days,
                    hour=int(float(row.get("hour_slot", 0))),
                    avg=float(row.get("avg_people", 0.0)),
                    peak=int(row.get("peak_people", 0) or 0),
                ),
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "message": f"Lỗi tính peak hour: {exc}"}

    def tool_get_daily_density(self, cam_id: str = "cam_01", days: int = 7) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {"ok": False, "source": "CV_DB", "data": [], "message": "Thiếu driver PostgreSQL."}
        try:
            with self._cv_connection() as conn:
                rows = self._fetch_all(
                    conn,
                    """
                    SELECT date_trunc('day', sample_ts) AS day,
                           AVG(people_count)::float AS avg_people,
                           MAX(people_count) AS peak_people
                    FROM camera_occupancy_samples
                    WHERE cam_id = %s AND sample_ts >= NOW() - (%s || ' day')::interval
                    GROUP BY 1
                    ORDER BY 1 DESC
                    """,
                    (cam_id, int(days)),
                )
            if not rows:
                return {"ok": True, "source": "CV_DB", "data": [], "message": f"Chưa có density cho {cam_id}."}
            preview = []
            for row in rows[:5]:
                preview.append(
                    "- {day}: trung bình {avg:.2f}, đỉnh {peak}".format(
                        day=str(row.get("day", ""))[:10],
                        avg=float(row.get("avg_people", 0.0)),
                        peak=int(row.get("peak_people", 0) or 0),
                    )
                )
            return {
                "ok": True,
                "source": "CV_DB",
                "data": rows,
                "message": "Mật độ người theo ngày:\n" + "\n".join(preview),
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "data": [], "message": f"Lỗi truy vấn density: {exc}"}

    def tool_get_latest_person_seen(self, cam_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.cv_db_available:
            return {"ok": False, "source": "CV_DB", "message": "Thiếu driver PostgreSQL."}
        try:
            sql = """
                SELECT event_ts, cam_id, person_id, person_name, event_type, confidence
                FROM camera_events
                WHERE person_name IS NOT NULL AND person_name <> ''
            """
            params: List[Any] = []
            if cam_id:
                sql += " AND cam_id = %s"
                params.append(cam_id)
            sql += " ORDER BY event_ts DESC LIMIT 1"
            with self._cv_connection() as conn:
                row = self._fetch_one(conn, sql, tuple(params))
            if not row:
                return {"ok": True, "source": "CV_DB", "message": "Chưa có người nào được nhận diện trong camera_events."}
            return {
                "ok": True,
                "source": "CV_DB",
                "data": row,
                "message": (
                    "Người được nhận diện gần nhất là {person_name} tại {cam_id} lúc {event_ts}."
                ).format(
                    person_name=row.get("person_name", "unknown"),
                    cam_id=row.get("cam_id", "cam_01"),
                    event_ts=row.get("event_ts", "?"),
                ),
            }
        except Exception as exc:
            return {"ok": False, "source": "CV_DB", "message": f"Lỗi đọc person_name mới nhất: {exc}"}

    def tool_answer_cv_query(self, query: str) -> Dict[str, Any]:
        text = (query or "").strip().lower()
        if not text:
            return {"ok": False, "source": "CV_DB", "message": "Thiếu câu hỏi CV."}

        if "te nga" in text or "té ngã" in text or "fall" in text:
            return self.tool_get_today_fall_count()
        if "dong nhat" in text or "đông nhất" in text or "peak" in text:
            return self.tool_get_peak_hour()
        if "mat do" in text or "mật độ" in text or "density" in text:
            return self.tool_get_daily_density()
        if "nguoi gan nhat" in text or "người gần nhất" in text or "latest person" in text or "nhan dien" in text:
            return self.tool_get_latest_person_seen()
        return self.tool_get_recent_cv_events(limit=5)

    def tool_get_elevator_status(self, elevator_id: int = 1) -> Dict[str, Any]:
        people_count = 4
        cv_source = "SIMULATED"
        cv_status = self.tool_get_cv_status()
        if cv_status.get("ok"):
            occupancy = (cv_status.get("status_data") or {}).get("occupancy") or {}
            if occupancy.get("people_count") is not None:
                people_count = occupancy.get("people_count")
            cv_source = "CV_DB"

        status = {
            "elevator_id": int(elevator_id),
            "floor": 5,
            "direction": "UP",
            "door": "CLOSED",
            "people_count": people_count,
            "overload": bool(int(people_count or 0) >= 8),
            "status": "OVERLOAD" if int(people_count or 0) >= 8 else "NORMAL",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "HYBRID",
            "people_source": cv_source,
        }
        return {
            "ok": True,
            "source": "ELEVATOR_STATUS",
            "status_data": status,
            "message": (
                "Thang máy {0} đang ở tầng {1}, hướng {2}, cửa {3}, số người {4}, quá tải: {5}."
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
