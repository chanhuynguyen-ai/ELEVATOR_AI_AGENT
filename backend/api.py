from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import time
import socket

from backend.chatbot_engine import ChatbotEngine

# =========================
# App & Engine
# =========================
app = FastAPI(title="Sunybot Elevator Chatbot", version="1.0.1")
engine = ChatbotEngine()

# =========================
# Path config
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEB_DIR = os.path.join(BASE_DIR, "gui", "web")
PAGES_DIR = os.path.join(WEB_DIR, "pages")
STATIC_DIR = os.path.join(WEB_DIR, "static")

# New Vite/React build output
DIST_DIR = os.path.join(WEB_DIR, "dist")
DIST_INDEX = os.path.join(DIST_DIR, "index.html")
DIST_ASSETS_DIR = os.path.join(DIST_DIR, "assets")
DIST_FAVICON = os.path.join(DIST_DIR, "favicon.ico")

# Old UI favicon
FAVICON_PATH = os.path.join(STATIC_DIR, "favicon.ico")


def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


def _dir_exists(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except Exception:
        return False


def get_local_ip() -> str:
    """
    Lấy IP LAN hiện tại của Jetson để in link truy cập từ máy khác.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_ui_links(port: int = 8000):
    lan_ip = get_local_ip()

    print("\n================ SUNYBOT UI ================")
    print(f"Local UI  : http://127.0.0.1:{port}/")
    print(f"LAN UI    : http://{lan_ip}:{port}/")
    print(f"Health    : http://{lan_ip}:{port}/health")
    print(f"Legacy UI : http://{lan_ip}:{port}/pages/assistant.html")
    if _file_exists(DIST_INDEX):
        print("Frontend  : React dist đang được ưu tiên tại /")
    else:
        print("Frontend  : Chưa thấy dist/index.html, có thể đang fallback UI cũ")
    print("============================================\n")


@app.on_event("startup")
def startup_debug():
    print("========== SUNYBOT UI STARTUP ==========")
    print(f"cwd={os.getcwd()}")
    print(f"BASE_DIR={BASE_DIR}")
    print(f"WEB_DIR={WEB_DIR} | exists={_dir_exists(WEB_DIR)}")
    print(f"DIST_DIR={DIST_DIR} | exists={_dir_exists(DIST_DIR)}")
    print(f"DIST_INDEX={DIST_INDEX} | exists={_file_exists(DIST_INDEX)}")
    print(f"DIST_ASSETS_DIR={DIST_ASSETS_DIR} | exists={_dir_exists(DIST_ASSETS_DIR)}")
    print("========================================")
    print_ui_links(8000)


# =========================
# Static mounts
# =========================
# Old static: /static/...
if _dir_exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Vite build assets: /assets/...
# (Vite build thường tham chiếu <script src="/assets/...">)
if _dir_exists(DIST_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=DIST_ASSETS_DIR), name="assets")


# =========================
# Favicon
# =========================
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Prefer new Vite favicon first
    if _file_exists(DIST_FAVICON):
        return FileResponse(DIST_FAVICON)

    # Fallback to old favicon
    if _file_exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH)

    return Response(status_code=204)


# =========================
# UI Routes
# =========================
@app.get("/", include_in_schema=False)
def home():
    """
    Serve UI:
    - Prefer Vite/React build at gui/web/dist/index.html
    - Fallback to old gui/web/index.html
    """
    if _file_exists(DIST_INDEX):
        return FileResponse(DIST_INDEX)

    old_index = os.path.join(WEB_DIR, "index.html")
    if _file_exists(old_index):
        return FileResponse(old_index)

    return JSONResponse(status_code=404, content={"error": "UI not found"})


@app.get("/pages/{page}", include_in_schema=False)
def serve_pages(page: str):
    """
    Serve old UI pages (legacy mode):
    /pages/call.html
    /pages/assistant.html
    /pages/guide.html
    /pages/sos.html
    /pages/maintenance.html
    """
    safe_page = os.path.basename(page)  # chống ../
    file_path = os.path.join(PAGES_DIR, safe_page)

    if not _file_exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Page not found"})

    return FileResponse(file_path)


# =========================
# Healthcheck
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# =========================
# Elevator Status (PHASE 1 - MOCK)
# =========================
@app.get("/api/elevator/status")
def elevator_status():
    """
    Mock realtime status.
    PHASE 2+ sẽ thay bằng dữ liệu thật (PLC/CV)
    """
    return {
        "elevator_id": 1,
        "floor": 5,
        "direction": "UP",          # UP / DOWN / IDLE
        "door": "CLOSED",           # OPEN / CLOSED / JAM
        "people_count": 4,
        "overload": False,
        "status": "NORMAL",         # NORMAL / WARNING / ERROR
        "time": time.strftime("%H:%M:%S"),
    }


# =========================
# Chatbot API
# =========================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    source: str
    intent: Optional[str] = None
    confidence: Optional[float] = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    r = engine.handle(req.message)
    return {
        "answer": r.get("answer", ""),
        "source": r.get("source", "UNKNOWN"),
        "intent": r.get("intent"),
        "confidence": r.get("confidence"),
    }


# =========================
# SPA fallback for React Router
# =========================
@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    """
    Nếu dùng React Router / routes client-side, refresh URL sẽ gọi vào đây.
    - Không đụng các API route và static routes
    - Nếu có dist/index.html -> trả về SPA
    - Nếu không -> fallback old index
    """
    # Không chặn các route hệ thống / API
    if full_path.startswith(("api", "chat", "health", "static", "assets", "pages")):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    # Prefer React build
    if _file_exists(DIST_INDEX):
        return FileResponse(DIST_INDEX)

    # Fallback old index
    old_index = os.path.join(WEB_DIR, "index.html")
    if _file_exists(old_index):
        return FileResponse(old_index)

    return JSONResponse(status_code=404, content={"error": "UI not found"})
