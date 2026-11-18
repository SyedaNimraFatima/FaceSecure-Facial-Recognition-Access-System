"""
FaceSecure demo UI server (FastAPI + Uvicorn).

    python frontend/app.py        -> open http://localhost:8000

The Live page (camera recognition) is open to everyone, like a door kiosk.
Registering, editing, deleting people and viewing logs or attendance need the
admin password.
"""
import os
import sys
import time
import secrets
import threading
import webbrowser
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from backend import (FaceSecure, FaceSecureError, NeedsConfirmation, decode_image, now_local,   # noqa: E402
                     MATCH_THRESHOLD, MODEL_NAME, SNAPSHOTS_DIR, SHIFTS, GRACE_MINUTES, MIN_STAY_MINUTES,
                     LIVE_THRESHOLD)

fs = FaceSecure()
app = FastAPI(title="FaceSecure")
SESSIONS = set()          # admin login tokens (cleared when the server restarts)
COOKIE = "facesecure_admin"


def friendly(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NeedsConfirmation as e:
        raise HTTPException(409, str(e))
    except FaceSecureError as e:
        raise HTTPException(400, str(e))


def admin_only(request: Request):
    if request.cookies.get(COOKIE) not in SESSIONS:
        raise HTTPException(401, "Admin login required.")


def with_urls(log):
    return {**log, "snapshot": f"/api/snapshots/{log['snapshot']}" if log.get("snapshot") else None}


# ---- page + public endpoints (kiosk) ---------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "FaceSecure.html"))


@app.get("/api/status")
def status():
    today = now_local()[:10]
    todays = [l for l in fs.logs(300) if l["ts"].startswith(today)]
    return {"storage": fs.store.label, "model": MODEL_NAME, "threshold": MATCH_THRESHOLD,
            "users": len(fs.users()), "shifts": SHIFTS,
            "liveness": fs.engine.liveness is not None, "live_threshold": LIVE_THRESHOLD, "grace": GRACE_MINUTES, "min_stay": MIN_STAY_MINUTES,
            "granted": sum(l["event"] == "granted" for l in todays),
            "denied": sum(l["event"] == "denied" for l in todays)}


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...)):
    img = friendly(decode_image, file.file.read())
    return {"width": img.shape[1], "height": img.shape[0], "faces": fs.recognize(img)}


# ---- admin login -----------------------------------------------------------
def _login(response: Response):
    token = secrets.token_urlsafe(32)
    SESSIONS.add(token)
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/api/admin")
def admin_state(request: Request):
    return {"password_set": fs.admin.is_set(), "logged_in": request.cookies.get(COOKIE) in SESSIONS}


@app.post("/api/admin/setup")
def admin_setup(response: Response, password: str = Form(...)):
    if fs.admin.is_set():
        raise HTTPException(400, "An admin password already exists.")
    friendly(fs.admin.set_password, password)
    return _login(response)


@app.post("/api/admin/login")
def admin_login(response: Response, password: str = Form(...)):
    if not fs.admin.check(password):
        time.sleep(1)                       # slows down password guessing
        raise HTTPException(401, "Wrong password.")
    return _login(response)


@app.post("/api/admin/logout")
def admin_logout(request: Request, response: Response):
    SESSIONS.discard(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ---- people (admin) --------------------------------------------------------
@app.get("/api/users", dependencies=[Depends(admin_only)])
def users():
    return [{**u, "photo": f"/api/users/{u['id']}/photo?v={u['updated_at']}"} for u in fs.users()]


@app.get("/api/users/{user_id}/photo", dependencies=[Depends(admin_only)])
def user_photo(user_id: str):
    path = friendly(fs.store.photo_path, user_id)
    if not os.path.exists(path):
        raise HTTPException(404, "No photo")
    return FileResponse(path)


@app.post("/api/users", dependencies=[Depends(admin_only)])
def register(name: str = Form(...), files: List[UploadFile] = File(...), employee_id: str = Form(""),
             department: str = Form(""), shift_start: str = Form("09:00"), shift_end: str = Form("17:00"),
             confirmed: bool = Form(False)):
    frames = [friendly(decode_image, f.file.read()) for f in files]
    return friendly(fs.register, name, frames, employee_id, department, shift_start, shift_end, confirmed)


@app.put("/api/users/{user_id}", dependencies=[Depends(admin_only)])
def edit_details(user_id: str, employee_id: str = Form(""), department: str = Form(""),
                 shift_start: str = Form(...), shift_end: str = Form(...)):
    return friendly(fs.update_details, user_id, employee_id, department, shift_start, shift_end)


@app.put("/api/users/{user_id}/face", dependencies=[Depends(admin_only)])
def update_face(user_id: str, files: List[UploadFile] = File(...), confirmed: bool = Form(False)):
    frames = [friendly(decode_image, f.file.read()) for f in files]
    return friendly(fs.update_face, user_id, frames, confirmed)


@app.post("/api/users/{user_id}/looks", dependencies=[Depends(admin_only)])
def add_look(user_id: str, label: str = Form(...), files: List[UploadFile] = File(...), confirmed: bool = Form(False)):
    frames = [friendly(decode_image, f.file.read()) for f in files]
    return friendly(fs.add_look, user_id, label, frames, confirmed)


@app.delete("/api/users/{user_id}/looks/{index}", dependencies=[Depends(admin_only)])
def remove_look(user_id: str, index: int):
    friendly(fs.remove_look, user_id, index)
    return {"ok": True}


@app.delete("/api/users/{user_id}", dependencies=[Depends(admin_only)])
def delete_user(user_id: str):
    return friendly(fs.delete_user, user_id)


# ---- logs + attendance (admin) ---------------------------------------------
@app.get("/api/logs", dependencies=[Depends(admin_only)])
def logs(limit: int = 100):
    return [with_urls(l) for l in fs.logs(min(limit, 300))]


@app.get("/api/snapshots/{day}/{filename}", dependencies=[Depends(admin_only)])
def snapshot(day: str, filename: str):
    path = os.path.realpath(os.path.join(SNAPSHOTS_DIR, day, filename))
    if not path.startswith(os.path.realpath(SNAPSHOTS_DIR) + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path)


@app.get("/api/attendance", dependencies=[Depends(admin_only)])
def attendance(day: Optional[str] = None):
    return friendly(fs.attendance, day or now_local()[:10])


@app.post("/api/attendance/correct", dependencies=[Depends(admin_only)])
def correct(user_id: str = Form(...), day: str = Form(...), field: str = Form(...),
            time_value: str = Form(...), note: str = Form("")):
    friendly(fs.correct_attendance, user_id, day, field, time_value, note)
    return friendly(fs.attendance, day)


@app.get("/api/attendance/export", dependencies=[Depends(admin_only)])
def export(month: str):
    path = friendly(fs.export_month, month)
    return FileResponse(path, filename=os.path.basename(path), media_type="text/csv")


if __name__ == "__main__":
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, ["http://localhost:8000"]).start()
    print("[INFO] FaceSecure demo running at http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
