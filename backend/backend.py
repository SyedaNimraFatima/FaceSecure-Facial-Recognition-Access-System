"""
FaceSecure backend: face detection + recognition (InsightFace) and storage.

Everything is saved as plain files in the project's data/ folder:

    data/users/<Name>/profile.json   name, id, dates and the face embedding
    data/users/<Name>/photo.jpg      profile picture taken at registration
    data/logs/<YYYY-MM-DD>.csv       every event that day (opens in Excel)
    data/snapshots/<YYYY-MM-DD>/     photos of faces that were denied access
    data/attendance/                 admin corrections and monthly attendance exports
    data/admin.json                  admin password (salted hash, never plain text)

Both the CMD menu (main.py) and the demo UI (frontend/app.py) use this module.
"""
import os
import re
import csv
import json
import time
import uuid
import shutil
import hashlib
import secrets
import calendar
import threading
from collections import defaultdict
from datetime import datetime, date, timedelta

import cv2
import numpy as np
from insightface.app import FaceAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
USERS_DIR = os.path.join(DATA_DIR, "users")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")
ADMIN_FILE = os.path.join(DATA_DIR, "admin.json")

# ---- Settings (override with environment variables if needed) -------------
MATCH_THRESHOLD = float(os.getenv("FACESECURE_THRESHOLD", "0.55"))  # cosine similarity
COOLDOWN_SECONDS = float(os.getenv("FACESECURE_COOLDOWN", "60"))    # out of view this long = a new arrival
MODEL_NAME = os.getenv("FACESECURE_MODEL", "buffalo_l")
MIN_SAMPLES = 3
LOOK_MIN_SIMILARITY = 0.20   # a new look this unlike the person's other looks needs admin confirmation
SNAPSHOT_GRANTED = os.getenv("FACESECURE_SNAPSHOT_GRANTED", "1") != "0"     # also photograph granted entries
GRANTED_PHOTO_DAYS = int(os.getenv("FACESECURE_GRANTED_PHOTO_DAYS", "30"))   # then deleted automatically
DENIED_PHOTO_DAYS = int(os.getenv("FACESECURE_DENIED_PHOTO_DAYS", "90"))
LIVENESS_ON = os.getenv("FACESECURE_LIVENESS", "1") != "0"
LIVE_THRESHOLD = float(os.getenv("FACESECURE_LIVE_THRESHOLD", "0.8"))   # live probability needed
LIVE_CONFIRM_FRAMES = 2       # a recognised face must look live this many frames in a row
SPOOF_CONFIRM_FRAMES = 3      # a face must look fake this many frames in a row to be logged as a spoof
HEAD_TURN_MIN = float(os.getenv("FACESECURE_HEAD_TURN", "0.15"))         # registration: required yaw range
UNKNOWN_CONFIRM_FRAMES = int(os.getenv("FACESECURE_UNKNOWN_FRAMES", "4"))   # deny only if unrecognised this many frames in a row
LOG_CACHE = 300
LOG_COLUMNS = ["date", "time", "event", "name", "user_id", "similarity", "snapshot", "note"]

# ---- Attendance rules ------------------------------------------------------
SHIFTS = {"9-5": ("09:00", "17:00"), "9-6": ("09:00", "18:00"), "10-6": ("10:00", "18:00")}
DEFAULT_SHIFT = SHIFTS["9-5"]
GRACE_MINUTES = int(os.getenv("FACESECURE_GRACE", "15"))           # late after shift start + grace
MIN_STAY_MINUTES = int(os.getenv("FACESECURE_MIN_STAY", "30"))     # last sighting this much after first = check-out
WORK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]     # weekends: staff use their keys, no attendance
CORRECTION_COLUMNS = ["date", "user_id", "name", "field", "time", "note", "recorded_at"]


class FaceSecureError(Exception):
    """A problem worth showing to the user (no face found, duplicate name...)."""


class NeedsConfirmation(FaceSecureError):
    """The action is allowed but unusual; ask the admin to confirm and retry."""


def now_local():
    return datetime.now().isoformat(timespec="seconds")       # e.g. 2026-11-17T14:05:22


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceSecureError("Could not read the image.")
    return img


def crop_face(bgr, box, pad=0.35):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return bgr[b:d, a:c]


def save_jpg(path, bgr, max_side=320):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise FaceSecureError("Could not encode image.")
    with open(path, "wb") as f:          # not cv2.imwrite: it fails on Windows paths with "–", accents, Urdu...
        f.write(buf.tobytes())


def safe_folder_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .") or "user"


def to_minutes(hhmm):
    h, m = hhmm.split(":")[:2]
    return int(h) * 60 + int(m)


def from_minutes(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def check_time(value, label="Time"):
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?", value or ""):
        raise FaceSecureError(f"{label} must look like 09:00.")
    return value if len(value) == 8 else value + ":00"


def check_shift(start, end):
    start, end = check_time(start, "Shift start")[:5], check_time(end, "Shift end")[:5]
    if to_minutes(end) <= to_minutes(start):
        raise FaceSecureError("Shift end must be after shift start.")
    return start, end


def append_csv(path, columns, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================================
# Admin password (only the admin may register, edit or delete people)
# ============================================================================

class AdminAuth:
    def is_set(self):
        return os.path.exists(ADMIN_FILE)

    @staticmethod
    def _hash(password, salt):
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()

    def set_password(self, password):
        if len(password or "") < 6:
            raise FaceSecureError("Password must be at least 6 characters.")
        salt = os.urandom(16)
        with open(ADMIN_FILE, "w") as f:
            json.dump({"salt": salt.hex(), "hash": self._hash(password, salt)}, f)

    def check(self, password):
        if not self.is_set():
            return False
        with open(ADMIN_FILE) as f:
            d = json.load(f)
        return secrets.compare_digest(d["hash"], self._hash(password or "", bytes.fromhex(d["salt"])))


# ============================================================================
# Face engine
# ============================================================================

class FaceEngine:
    """InsightFace buffalo_l: SCRFD face detector + ArcFace 512-d embeddings,
    plus InsightFace's passive liveness add-on (80x80 CNN, live probability)."""

    def __init__(self):
        print(f"[INFO] Loading face models ({MODEL_NAME})...")
        self.app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"],
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.lock = threading.Lock()
        self.liveness = None
        if LIVENESS_ON:
            try:
                from insightface.addons import Liveness, ensure_addon
                self.liveness = Liveness(ensure_addon("liveness"), threshold=LIVE_THRESHOLD,
                                         providers=["CPUExecutionProvider"])
                self.liveness.prepare(-1)
                print(f"[INFO] Liveness check on (threshold {LIVE_THRESHOLD}).")
            except Exception as e:
                print(f"[WARNING] Liveness check unavailable ({e.__class__.__name__}: {e}). "
                      "Needs insightface 2.1+ and a one-time internet download.")
        print("[INFO] Models ready.")

    def live_score(self, bgr, face):
        """Probability the face is a live person (not a photo or screen).
        Returns None when liveness is off, or "reposition" when the face is too close to the edge."""
        if self.liveness is None:
            return None
        try:
            with self.lock:
                r = self.liveness.predict(bgr, face.kps)
        except Exception:
            return None
        return r["live_score"] if r["status"] == "ok" else "reposition"

    @staticmethod
    def yaw(face):
        """Head turn from the 5 landmarks: nose offset from the eye midpoint, divided by eye distance.
        A flat photo turned in front of the camera keeps this ratio; a real head changes it."""
        le, re, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_dist = max(1e-6, abs(re[0] - le[0]))
        return float((nose[0] - (le[0] + re[0]) / 2) / eye_dist)

    def detect(self, bgr):
        """Faces sorted largest first. Each has .bbox and .normed_embedding."""
        with self.lock:
            faces = self.app.get(bgr)
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                      reverse=True)


# ============================================================================
# File storage
# ============================================================================

class FileStore:
    label = "Local files"

    def __init__(self):
        for d in (USERS_DIR, LOGS_DIR, SNAPSHOTS_DIR, ATTENDANCE_DIR):
            os.makedirs(d, exist_ok=True)
        ignore = os.path.join(DATA_DIR, ".gitignore")       # keep face data off GitHub
        if not os.path.exists(ignore):
            with open(ignore, "w") as f:
                f.write("*\n")
        self._pending = []          # log rows waiting because a CSV was open in Excel
        self.lock = threading.Lock()

    # ---- users -------------------------------------------------------------
    def _profiles(self):
        for folder in sorted(os.listdir(USERS_DIR)):
            path = os.path.join(USERS_DIR, folder, "profile.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    yield folder, json.load(f)

    def _folder_of(self, uid):
        for folder, p in self._profiles():
            if p["id"] == uid:
                return os.path.join(USERS_DIR, folder)
        raise FaceSecureError("User not found.")

    @staticmethod
    def _looks(p):
        if "looks" in p:
            return p["looks"]
        # profiles saved before "looks" existed had one averaged embedding
        return [{"label": "standard", "samples": p.get("samples"), "added_at": p.get("created_at"),
                 "embedding": p["embedding"]}]

    def list_users(self):
        users = []
        for _, p in self._profiles():
            u = {k: v for k, v in p.items() if k not in ("embedding", "looks")}
            u["looks"] = [l["label"] for l in self._looks(p)]
            users.append(u)
        return sorted(users, key=lambda u: u["created_at"])

    def load_embeddings(self):
        """One entry per look: (user_id, name, embedding, look_label)."""
        return [(p["id"], p["name"], np.array(l["embedding"], dtype=np.float32), l["label"])
                for _, p in self._profiles() for l in self._looks(p)]

    def photo_path(self, uid):
        return os.path.join(self._folder_of(uid), "photo.jpg")

    def _write_profile(self, folder, profile):
        with open(os.path.join(folder, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=1)

    def add_user(self, uid, name, emb, photo, samples, details):
        folder = os.path.join(USERS_DIR, safe_folder_name(name))
        if os.path.exists(folder):
            folder += "_" + uid[:6]
        os.makedirs(folder)
        t = now_local()
        self._write_profile(folder, {"id": uid, "name": name, **details, "samples": samples,
                                     "created_at": t, "updated_at": t,
                                     "looks": [{"label": "standard", "samples": samples, "added_at": t,
                                                "embedding": [round(float(x), 6) for x in emb]}]})
        save_jpg(os.path.join(folder, "photo.jpg"), photo, 240)

    def update_details(self, uid, details):
        folder = self._folder_of(uid)
        with open(os.path.join(folder, "profile.json"), encoding="utf-8") as f:
            p = json.load(f)
        p.update(details, updated_at=now_local())
        self._write_profile(folder, p)

    def _read(self, uid):
        folder = self._folder_of(uid)
        with open(os.path.join(folder, "profile.json"), encoding="utf-8") as f:
            return folder, json.load(f)

    def update_face(self, uid, emb, photo, samples):
        """Re-capture: replaces every look with one new standard look."""
        folder, p = self._read(uid)
        t = now_local()
        p.pop("embedding", None)
        p.update(samples=samples, updated_at=t,
                 looks=[{"label": "standard", "samples": samples, "added_at": t,
                         "embedding": [round(float(x), 6) for x in emb]}])
        self._write_profile(folder, p)
        save_jpg(os.path.join(folder, "photo.jpg"), photo, 240)

    def add_look(self, uid, label, emb, samples, photo=None):
        folder, p = self._read(uid)
        if photo is not None and not os.path.exists(os.path.join(folder, "photo.jpg")):
            save_jpg(os.path.join(folder, "photo.jpg"), photo, 240)     # fill in a missing profile photo
        looks = self._looks(p)
        t = now_local()
        looks.append({"label": label, "samples": samples, "added_at": t,
                      "embedding": [round(float(x), 6) for x in emb]})
        p.pop("embedding", None)
        p.update(looks=looks, updated_at=t)
        self._write_profile(folder, p)

    def remove_look(self, uid, index):
        folder, p = self._read(uid)
        looks = self._looks(p)
        if len(looks) <= 1:
            raise FaceSecureError("A person needs at least one look. Use re-capture instead.")
        if not 0 <= index < len(looks):
            raise FaceSecureError("Look not found.")
        removed = looks.pop(index)
        p.pop("embedding", None)
        p.update(looks=looks, updated_at=now_local())
        self._write_profile(folder, p)
        return removed["label"]

    def delete_user(self, uid):
        shutil.rmtree(self._folder_of(uid))

    # ---- logs --------------------------------------------------------------
    def save_snapshot(self, ts, bgr, event):
        """Save the face photo for a log entry; returns its path relative to data/snapshots."""
        rel = f"{ts[:10]}/{ts[11:].replace(':', '-')}_{event}_{uuid.uuid4().hex[:4]}.jpg"
        save_jpg(os.path.join(SNAPSHOTS_DIR, rel), bgr)
        return rel

    def delete_old_snapshots(self):
        """Retention: granted photos older than GRANTED_PHOTO_DAYS and denied photos older than
        DENIED_PHOTO_DAYS are deleted. Log entries are kept; they just lose their photo."""
        today, removed = date.today(), 0
        for day in os.listdir(SNAPSHOTS_DIR):
            folder = os.path.join(SNAPSHOTS_DIR, day)
            try:
                age = (today - date.fromisoformat(day)).days
            except ValueError:
                continue
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                keep_days = GRANTED_PHOTO_DAYS if "_granted_" in name else DENIED_PHOTO_DAYS
                if age > keep_days:
                    os.remove(os.path.join(folder, name))
                    removed += 1
            if not os.listdir(folder):
                os.rmdir(folder)
        if removed:
            print(f"[INFO] Deleted {removed} snapshot(s) past their retention period.")

    def add_log(self, entry):
        row = {"date": entry["ts"][:10], "time": entry["ts"][11:], "event": entry["event"],
               "name": entry["user_name"] or "", "user_id": entry["user_id"] or "",
               "similarity": "" if entry["score"] is None else f"{entry['score']:.3f}",
               "snapshot": entry["snapshot"] or "", "note": entry.get("note") or ""}
        with self.lock:
            self._pending.append(row)
            rows, self._pending = self._pending, []
            try:
                for day in sorted({r["date"] for r in rows}):
                    append_csv(os.path.join(LOGS_DIR, f"{day}.csv"), LOG_COLUMNS,
                               [r for r in rows if r["date"] == day])
                    rows = [r for r in rows if r["date"] != day]
            except PermissionError:
                self._pending = rows    # file is open in Excel: keep and retry on the next event
                print("[WARNING] Log file is open in another program; will write it later.")

    def list_logs(self, limit=LOG_CACHE):
        out = []
        for name in sorted(os.listdir(LOGS_DIR), reverse=True):
            if not name.endswith(".csv"):
                continue
            try:
                with open(os.path.join(LOGS_DIR, name), newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except OSError:
                continue
            for r in reversed(rows):
                out.append({"ts": f"{r['date']}T{r['time']}", "event": r["event"],
                            "user_name": r["name"] or None, "user_id": r["user_id"] or None,
                            "score": float(r["similarity"]) if r["similarity"] else None,
                            "snapshot": r["snapshot"] or None, "note": r.get("note") or None})
                if len(out) >= limit:
                    return out
        return out

    def read_day(self, day):
        try:
            return read_csv(os.path.join(LOGS_DIR, f"{day}.csv"))
        except OSError:
            return []

    def add_correction(self, row):
        append_csv(os.path.join(ATTENDANCE_DIR, "corrections.csv"), CORRECTION_COLUMNS, [row])

    def corrections(self, day):
        latest = {}
        for r in read_csv(os.path.join(ATTENDANCE_DIR, "corrections.csv")):
            if r["date"] == day:
                latest[(r["user_id"], r["field"])] = r      # later corrections win
        return latest


# ============================================================================
# FaceSecure: everything the CMD menu and the UI call
# ============================================================================

class FaceSecure:
    def __init__(self):
        self.engine = FaceEngine()
        self.store = FileStore()
        self.admin = AdminAuth()
        self._last_cleanup = None
        print(f"[INFO] Data folder: {DATA_DIR}")
        self._known = None              # cached [(id, name, embedding)]
        self._users = None              # cached user list (no embeddings)
        self._logs = None               # cached recent logs, newest first
        self._last_seen = {}            # identity -> time.time() it was last in view
        self._unknown_streak = 0        # consecutive frames containing an unrecognised face
        self._spoof_streak = 0          # consecutive frames containing a face that looks fake
        self._live_streak = {}          # user_id -> consecutive frames recognised and live
        self._lock = threading.Lock()

    # ---- cached reads ------------------------------------------------------
    def known(self):
        with self._lock:
            if self._known is None:
                self._known = self.store.load_embeddings()
            return self._known

    def users(self):
        with self._lock:
            if self._users is None:
                self._users = self.store.list_users()
            return list(self._users)

    def logs(self, limit=100):
        with self._lock:
            if self._logs is None:
                self._logs = self.store.list_logs(LOG_CACHE)
            return self._logs[:limit]

    def _invalidate(self):
        with self._lock:
            self._known = None
            self._users = None

    def best_match(self, emb):
        """Closest stored look of any person: (user_id, name, cosine similarity, look label)."""
        best = (None, None, 0.0, None)
        for uid, name, ref, look in self.known():
            score = float(np.dot(ref, emb))       # both L2-normalised -> cosine
            if score > best[2]:
                best = (uid, name, score, look)
        return best

    def find_user(self, id_or_name):
        key = id_or_name.strip().lower()
        for u in self.users():
            if u["id"] == id_or_name.strip() or u["name"].lower() == key:
                return u
        return None

    # ---- recognition -------------------------------------------------------
    def recognize(self, bgr, log=True):
        """Return one result per face, with status granted / denied / spoof / checking.

        - Liveness: a recognised face must look live for LIVE_CONFIRM_FRAMES frames in a row
          before access is granted; a face that looks like a photo or screen for
          SPOOF_CONFIRM_FRAMES frames in a row is logged as a spoof.
        - Unknown faces are only denied after UNKNOWN_CONFIRM_FRAMES unrecognised frames in a row,
          so a registered person turning their head is not logged as a stranger.
        - A person is logged once when they arrive; after COOLDOWN_SECONDS out of view the next
          sighting is a new arrival."""
        faces = self.engine.detect(bgr)
        rows = []
        for face in faces:
            uid, name, score, look = self.best_match(face.normed_embedding)
            live = self.engine.live_score(bgr, face)
            rows.append((face, uid, name, score, look, live))

        is_spoof = lambda live: isinstance(live, float) and live < LIVE_THRESHOLD
        any_spoof = any(is_spoof(r[5]) for r in rows)
        any_unknown = any((r[1] is None or r[3] < MATCH_THRESHOLD) and not is_spoof(r[5]) for r in rows)
        self._spoof_streak = self._spoof_streak + 1 if any_spoof else 0
        self._unknown_streak = self._unknown_streak + 1 if any_unknown else 0
        seen_live = set()

        results = []
        for face, uid, name, score, look, live in rows:
            box = [int(v) for v in face.bbox]
            matched = uid is not None and score >= MATCH_THRESHOLD
            live_ok = live is None or (isinstance(live, float) and live >= LIVE_THRESHOLD)
            hint = None
            if is_spoof(live):
                status = "spoof" if self._spoof_streak >= SPOOF_CONFIRM_FRAMES else "checking"
            elif live == "reposition":
                status, hint = "checking", "Move to the centre of the camera"
            elif matched and live_ok:
                seen_live.add(uid)
                streak = self._live_streak.get(uid, 0) + 1
                self._live_streak[uid] = streak
                status = "granted" if (live is None or streak >= LIVE_CONFIRM_FRAMES) else "checking"
            else:
                status = "denied" if self._unknown_streak >= UNKNOWN_CONFIRM_FRAMES else "checking"
            r = {"box": box, "status": status, "granted": status == "granted",
                 "confirmed": status != "checking", "score": round(score, 3),
                 "live": round(live, 3) if isinstance(live, float) else None, "hint": hint,
                 "look": look if matched else None, "user_id": uid if matched else None,
                 "name": name if matched else "Unknown", "logged": False}
            if log and status != "checking":
                key = uid if status == "granted" else status        # "denied" / "spoof"
                t = time.time()
                is_new_arrival = t - self._last_seen.get(key, 0) >= COOLDOWN_SECONDS
                self._last_seen[key] = t
                if is_new_arrival:
                    photo = crop_face(bgr, box, 1.0) if (status != "granted" or SNAPSHOT_GRANTED) else None
                    note = f"live {live:.2f}" if isinstance(live, float) else None
                    if status == "spoof" and matched:
                        note = f"{note}; looked like {name}"
                    self.log(status, r["user_id"] if status == "granted" else None,
                             r["name"] if status == "granted" else ("Unknown" if status == "denied" else "Spoof"),
                             score, photo, note=note)
                    r["logged"] = True
            results.append(r)
        for uid in list(self._live_streak):            # streaks only count consecutive frames
            if uid not in seen_live:
                del self._live_streak[uid]
        return results

    def log(self, event, user_id=None, user_name=None, score=None, snapshot_img=None, note=None):
        ts = now_local()
        entry = {"ts": ts, "event": event, "user_id": user_id, "user_name": user_name,
                 "score": None if score is None else round(float(score), 3), "snapshot": None,
                 "note": note}
        try:
            if snapshot_img is not None:
                entry["snapshot"] = self.store.save_snapshot(ts, snapshot_img, event)
            if ts[:10] != self._last_cleanup:           # once a day
                self._last_cleanup = ts[:10]
                self.store.delete_old_snapshots()
            self.store.add_log(entry)
        except Exception as e:  # never let logging crash recognition
            print(f"[WARNING] Could not write log: {e}")
        with self._lock:
            if self._logs is not None:
                self._logs = ([entry] + self._logs)[:LOG_CACHE]
        return entry

    # ---- registration ------------------------------------------------------
    def _embed_frames(self, frames):
        """Embed the registration photos and check they come from a live person:
        passive liveness on every photo, plus an active head-turn check across the photos
        (the person is asked to look straight, left and right)."""
        embs, photo, lives, yaws = [], None, [], []
        for img in frames:
            faces = self.engine.detect(img)
            if not faces:
                continue
            f = faces[0]
            embs.append(f.normed_embedding)
            yaws.append(self.engine.yaw(f))
            live = self.engine.live_score(img, f)
            if isinstance(live, float):
                lives.append(live)
            if photo is None:
                photo = crop_face(img, [int(v) for v in f.bbox])
        if len(embs) < MIN_SAMPLES:
            raise FaceSecureError(f"Only {len(embs)} of {len(frames)} photos had a clear face "
                                  f"(need {MIN_SAMPLES}). Face the camera in good light and retry.")
        problems = []
        if lives and sum(l < LIVE_THRESHOLD for l in lives) > len(lives) / 2:
            problems.append(f"most photos look like a printed photo or a screen "
                            f"(average live score {np.mean(lives):.2f}, needs {LIVE_THRESHOLD})")
        if len(frames) >= 3 and max(yaws) - min(yaws) < HEAD_TURN_MIN:
            problems.append(f"no head turn was seen (movement {max(yaws) - min(yaws):.2f}, needs {HEAD_TURN_MIN}); "
                            f"a flat photo cannot turn")
        mean = np.mean(embs, axis=0)
        check = (f"live {np.mean(lives):.2f}" if lives else "liveness off") + f", head turn {max(yaws) - min(yaws):.2f}"
        return (mean / np.linalg.norm(mean)).astype(np.float32), photo, len(embs), problems, check

    @staticmethod
    def _require_confirmation(problems, confirmed):
        """Raise one combined warning the admin must confirm; return a note for the log."""
        if problems and not confirmed:
            raise NeedsConfirmation("Please check: " + "; ".join(problems) + ". Only continue if this really is the "
                                    "right person standing in front of the camera (a niqab or mask can lower "
                                    "the live score).")
        return "; confirmed by admin despite: " + "; ".join(problems) if problems else ""

    def _details(self, employee_id="", department="", shift_start=None, shift_end=None, uid=None):
        employee_id, department = (employee_id or "").strip(), (department or "").strip()
        if employee_id and any(u.get("employee_id", "").lower() == employee_id.lower() and u["id"] != uid
                               for u in self.users()):
            raise FaceSecureError(f"Employee ID '{employee_id}' is already in use.")
        start, end = check_shift(shift_start or DEFAULT_SHIFT[0], shift_end or DEFAULT_SHIFT[1])
        return {"employee_id": employee_id, "department": department,
                "shift_start": start, "shift_end": end}

    def register(self, name, frames, employee_id="", department="", shift_start=None, shift_end=None,
                 confirmed=False):
        name = " ".join(name.split())
        if len(name) < 2:
            raise FaceSecureError("Name must be at least 2 characters.")
        if self.find_user(name):
            raise FaceSecureError(f"'{name}' is already registered.")
        details = self._details(employee_id, department, shift_start, shift_end)
        emb, photo, n, problems, check = self._embed_frames(frames)
        uid, other, score, _ = self.best_match(emb)
        if uid and score >= MATCH_THRESHOLD:
            raise FaceSecureError(f"This face is already registered as '{other}'.")
        override = self._require_confirmation(problems, confirmed)
        new_id = uuid.uuid4().hex[:12]
        self.store.add_user(new_id, name, emb, photo, n, details)
        self._invalidate()
        self.log("registered", new_id, name,
                 note=f"Shift {details['shift_start']}-{details['shift_end']}; {check}{override}")
        return {"id": new_id, "name": name, "samples": n, **details}

    def update_details(self, user_id, employee_id="", department="", shift_start=None, shift_end=None):
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        details = self._details(employee_id, department, shift_start, shift_end, uid=user["id"])
        self.store.update_details(user["id"], details)
        self._invalidate()
        self.log("edited", user["id"], user["name"],
                 note=f"ID {details['employee_id'] or '-'}, {details['department'] or '-'}, "
                      f"shift {details['shift_start']}-{details['shift_end']}")
        return {**user, **details}

    def update_face(self, user_id, frames, confirmed=False):
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        emb, photo, n, problems, check = self._embed_frames(frames)
        uid, other, score, _ = self.best_match(emb)
        if uid and uid != user["id"] and score >= MATCH_THRESHOLD:
            raise FaceSecureError(f"This face belongs to '{other}', not '{user['name']}'.")
        override = self._require_confirmation(problems, confirmed)
        self.store.update_face(user["id"], emb, photo, n)
        self._invalidate()
        self.log("updated", user["id"], user["name"], note=f"Face re-captured, extra looks removed; {check}{override}")
        return {"id": user["id"], "name": user["name"], "samples": n}

    def add_look(self, user_id, label, frames, confirmed=False):
        """Extra profile for the same person, e.g. with glasses, niqab or a mask.
        Recognition uses whichever look is closest."""
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        label = " ".join((label or "").split())[:30].lower()
        if not label:
            raise FaceSecureError("Name the look, e.g. glasses or niqab.")
        emb, photo, n, problems, check = self._embed_frames(frames)
        uid, other, score, _ = self.best_match(emb)
        if uid and uid != user["id"] and score >= MATCH_THRESHOLD:
            raise FaceSecureError(f"These photos match '{other}', not '{user['name']}'.")
        own = max((float(np.dot(ref, emb)) for u, _, ref, _ in self.known() if u == user["id"]), default=0.0)
        if own < LOOK_MIN_SIMILARITY:
            problems.append(f"these photos look very different from {user['name']}'s other looks (similarity "
                            f"{own:.2f}); anyone added here will be let in as {user['name']}")
        override = self._require_confirmation(problems, confirmed)
        self.store.add_look(user["id"], label, emb, n, photo)
        self._invalidate()
        self.log("look added", user["id"], user["name"],
                 note=f"{label} ({n} photos, similarity to existing looks {own:.2f}; {check}{override})")
        return {"id": user["id"], "name": user["name"], "label": label, "samples": n,
                "similarity_to_existing": round(own, 3)}

    def remove_look(self, user_id, index):
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        label = self.store.remove_look(user["id"], int(index))
        self._invalidate()
        self.log("look removed", user["id"], user["name"], note=label)

    def delete_user(self, user_id):
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        self.store.delete_user(user["id"])
        self._invalidate()
        self.log("deleted", user["id"], user["name"])
        return user

    # ---- attendance --------------------------------------------------------
    def correct_attendance(self, user_id, day, field, time_value, note):
        """Admin fix for a missed check-in / check-out. Kept as a separate, audited record."""
        user = self.find_user(user_id)
        if not user:
            raise FaceSecureError("User not found.")
        if field not in ("check_in", "check_out"):
            raise FaceSecureError("Choose check-in or check-out.")
        try:
            date.fromisoformat(day)
        except ValueError:
            raise FaceSecureError("Date must look like 2026-11-17.")
        if day > now_local()[:10]:
            raise FaceSecureError("Cannot correct a future date.")
        time_value = check_time(time_value)
        if not (note or "").strip():
            raise FaceSecureError("Add a short reason (for example 'Forgot to look at camera').")
        self.store.add_correction({"date": day, "user_id": user["id"], "name": user["name"], "field": field,
                                   "time": time_value, "note": note.strip(), "recorded_at": now_local()})
        label = "check-in" if field == "check_in" else "check-out"
        self.log("corrected", user["id"], user["name"], note=f"{day} {label} set to {time_value[:5]}: {note.strip()}")

    def attendance(self, day):
        """One row per employee for a date, built from the day's 'granted' sightings:
        check-in = first sighting, check-out = last sighting at least MIN_STAY_MINUTES later.
        Admin corrections override either value."""
        d = date.fromisoformat(day)
        rows = self.store.read_day(day)
        seen = defaultdict(list)
        granted_secs = []
        for r in rows:
            if r["event"] == "granted" and r["user_id"]:
                seen[r["user_id"]].append(r["time"])
                granted_secs.append(to_minutes(r["time"]) * 60 + int(r["time"][6:8]))
        # unknown face recognised at almost the same moment as an employee = possible tailgating
        tailgating = sum(1 for r in rows if r["event"] == "denied" and any(
            abs(to_minutes(r["time"]) * 60 + int(r["time"][6:8]) - g) <= 3 for g in granted_secs))
        fixes = self.store.corrections(day)
        is_workday = d.strftime("%a") in WORK_DAYS
        now = datetime.now()
        is_today, is_past = d == now.date(), d < now.date()
        now_min = now.hour * 60 + now.minute

        out = []
        for u in self.users():
            if u["created_at"][:10] > day:
                continue                                    # not registered yet on that date
            start, end = u.get("shift_start", DEFAULT_SHIFT[0]), u.get("shift_end", DEFAULT_SHIFT[1])
            times = sorted(seen.get(u["id"], []))
            check_in = times[0] if times else None
            check_out = times[-1] if times and to_minutes(times[-1]) - to_minutes(times[0]) >= MIN_STAY_MINUTES else None
            notes = []
            for field in ("check_in", "check_out"):
                fix = fixes.get((u["id"], field))
                if fix:
                    if field == "check_in":
                        check_in = fix["time"]
                    else:
                        check_out = fix["time"]
                    notes.append(f"{'Check-in' if field == 'check_in' else 'Check-out'} corrected by admin: {fix['note']}")

            status = []
            if not check_in:
                if not is_workday:
                    status.append("Day off")
                elif is_past or (is_today and now_min > to_minutes(start) + GRACE_MINUTES):
                    status.append("Absent")
                else:
                    status.append("Not in yet")
            else:
                if to_minutes(check_in) > to_minutes(start) + GRACE_MINUTES:
                    status.append("Late")
                if check_out:
                    if to_minutes(check_out) < to_minutes(end):
                        status.append("Left early")
                elif is_past or (is_today and now_min > to_minutes(end) + 60):
                    status.append("No check-out")
                else:
                    status.append("In office")
                if not status:
                    status.append("On time")
            hours = None
            if check_in and check_out and check_out > check_in:
                hours = round((to_minutes(check_out) - to_minutes(check_in)) / 60, 2)
            out.append({"user_id": u["id"], "name": u["name"], "employee_id": u.get("employee_id", ""),
                        "department": u.get("department", ""), "shift": f"{start}-{end}",
                        "check_in": check_in[:5] if check_in else None,
                        "check_out": check_out[:5] if check_out else None,
                        "hours": hours, "sightings": len(times), "status": status, "notes": notes})
        summary = {"present": sum(1 for r in out if r["check_in"]),
                   "late": sum("Late" in r["status"] for r in out),
                   "absent": sum("Absent" in r["status"] for r in out),
                   "no_checkout": sum("No check-out" in r["status"] for r in out),
                   "tailgating": tailgating, "workday": is_workday}
        return {"date": day, "rows": out, "summary": summary}

    def export_month(self, month):
        """Write data/attendance/attendance_YYYY-MM.csv and return its path."""
        try:
            y, m = map(int, month.split("-"))
            last = calendar.monthrange(y, m)[1]
        except (ValueError, calendar.IllegalMonthError):
            raise FaceSecureError("Month must look like 2026-11.")
        today = date.today()
        rows = []
        for dnum in range(1, last + 1):
            d = date(y, m, dnum)
            if d > today:
                break
            for r in self.attendance(d.isoformat())["rows"]:
                rows.append({"date": d.isoformat(), "day": d.strftime("%a"), "employee_id": r["employee_id"],
                             "name": r["name"], "department": r["department"], "shift": r["shift"],
                             "check_in": r["check_in"] or "", "check_out": r["check_out"] or "",
                             "hours": "" if r["hours"] is None else r["hours"],
                             "status": ", ".join(r["status"]), "notes": " | ".join(r["notes"])})
        path = os.path.join(ATTENDANCE_DIR, f"attendance_{month}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:     # utf-8-sig: Excel-friendly
            w = csv.DictWriter(f, fieldnames=["date", "day", "employee_id", "name", "department", "shift",
                                              "check_in", "check_out", "hours", "status", "notes"])
            w.writeheader()
            w.writerows(rows)
        return path
