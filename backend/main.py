"""
FaceSecure command-line menu (works without the UI).

    python backend/main.py
"""
import os
import queue
import getpass
import threading
from datetime import date

import cv2

from backend import FaceSecure, FaceSecureError, NeedsConfirmation, COOLDOWN_SECONDS, SHIFTS, DEFAULT_SHIFT

SAMPLES = 5
PROMPTS = ["Look straight", "Turn slightly left", "Turn slightly right",
           "Chin slightly up", "Natural smile"]


# ---- optional voice (runs in its own thread so video never freezes) --------
_voice = queue.Queue()


def _voice_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
    except Exception:
        engine = None
    while True:
        text = _voice.get()
        if engine:
            engine.say(text)
            engine.runAndWait()


threading.Thread(target=_voice_worker, daemon=True).start()


def speak(text):
    if _voice.empty():
        _voice.put(text)


# ---- webcam capture --------------------------------------------------------
def open_camera():
    """Open the camera set in FACESECURE_CAMERA, otherwise the first one that gives a picture
    (skips a broken laptop camera and finds e.g. an iPhone running Camo / Iriun / iVCam)."""
    preferred = os.getenv("FACESECURE_CAMERA")
    order = [int(preferred)] if preferred else [0, 1, 2, 3]
    for index in order:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        ok = cap.isOpened() and any(cap.read()[0] for _ in range(5))
        if ok:
            print(f"[INFO] Using camera {index}")
            return cap
        cap.release()
    raise FaceSecureError("No working camera found. Connect one (or start your phone-webcam app) and try again.")


def capture_samples(fs, title):
    cap = open_camera()
    frames = []
    try:
        while len(frames) < SAMPLES:
            prompt = PROMPTS[len(frames) % len(PROMPTS)]
            for i in range(3, 0, -1):
                ok, frame = cap.read()
                if not ok:
                    continue
                show = frame.copy()
                cv2.putText(show, f"{prompt}  {i}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                cv2.putText(show, f"Sample {len(frames) + 1}/{SAMPLES}  (q = cancel)", (30, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(title, show)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    print("[INFO] Cancelled.")
                    return []
            ok, frame = cap.read()
            if not ok:
                continue
            faces = fs.engine.detect(frame)
            if not faces:
                print("[WARNING] No face detected, trying again.")
                continue
            x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
            frames.append(frame)
            print(f"[INFO] Captured sample {len(frames)}/{SAMPLES}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.imshow(title, frame)
            cv2.waitKey(600)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return frames


# ---- admin check -----------------------------------------------------------
_admin_ok = False


def require_admin(fs):
    """Ask for the admin password once per session (creates it the first time)."""
    global _admin_ok
    if _admin_ok:
        return True
    if not fs.admin.is_set():
        print("No admin password yet. Create one (at least 6 characters).")
        pw = getpass.getpass("New admin password: ")
        if pw != getpass.getpass("Confirm password: "):
            print("[ERROR] Passwords don't match.")
            return False
        fs.admin.set_password(pw)
        _admin_ok = True
        return True
    if fs.admin.check(getpass.getpass("Admin password: ")):
        _admin_ok = True
        return True
    print("[ERROR] Wrong password.")
    return False


def admin(action):
    def wrapped(fs):
        if require_admin(fs):
            action(fs)
    return wrapped


def ask_shift(current=None):
    options = list(SHIFTS.items())
    default = current or DEFAULT_SHIFT
    print("Shift:")
    for i, (key, (a, b)) in enumerate(options, 1):
        print(f"  {i}. {a}-{b}")
    print(f"  {len(options) + 1}. Other hours")
    choice = input(f"Choose (Enter = {default[0]}-{default[1]}): ").strip()
    if not choice:
        return default
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1][1]
    return input("Start time (e.g. 08:30): ").strip(), input("End time (e.g. 16:30): ").strip()


def confirm_and_run(action):
    """Run action(confirmed=False); if it needs the admin's confirmation, ask and retry."""
    try:
        return action(False)
    except NeedsConfirmation as e:
        print(f"[WARNING] {e}")
        if input("Continue anyway? (yes/no): ").strip().lower() != "yes":
            print("[INFO] Cancelled.")
            return None
        return action(True)


# ---- menu actions ----------------------------------------------------------
@admin
def register(fs):
    name = input("Full name: ").strip()
    if not name:
        return print("[ERROR] Name cannot be empty.")
    if fs.find_user(name):
        return print(f"[ERROR] '{name}' already exists.")
    employee_id = input("Employee ID (optional): ").strip()
    department = input("Department (optional): ").strip()
    start, end = ask_shift()
    fs._details(employee_id, department, start, end)          # check details before using the camera
    frames = capture_samples(fs, "Register Face")
    if frames:
        user = confirm_and_run(lambda c: fs.register(name, frames, employee_id, department, start, end, c))
        if not user:
            return
        print(f"[SUCCESS] '{user['name']}' registered, shift {user['shift_start']}-{user['shift_end']} "
              f"({user['samples']} samples).")


@admin
def update(fs):
    user = fs.find_user(input("Name or ID to update: "))
    if not user:
        return print("[ERROR] User not found.")
    frames = capture_samples(fs, f"Updating {user['name']}")
    if frames:
        if not confirm_and_run(lambda c: fs.update_face(user["id"], frames, c)):
            return
        print(f"[SUCCESS] Face data for '{user['name']}' updated.")


@admin
def add_look(fs):
    user = fs.find_user(input("Name or ID: "))
    if not user:
        return print("[ERROR] User not found.")
    label = input("Look (e.g. glasses, hijab, niqab, mask): ").strip()
    frames = capture_samples(fs, f"{user['name']} - {label}")
    if frames:
        r = confirm_and_run(lambda c: fs.add_look(user["id"], label, frames, c))
        if not r:
            return
        print(f"[SUCCESS] Added '{r['label']}' look for {r['name']} "
              f"(similarity to existing looks {r['similarity_to_existing']:.2f}).")


@admin
def edit(fs):
    user = fs.find_user(input("Name or ID to edit: "))
    if not user:
        return print("[ERROR] User not found.")
    employee_id = input(f"Employee ID [{user.get('employee_id', '')}]: ").strip() or user.get("employee_id", "")
    department = input(f"Department [{user.get('department', '')}]: ").strip() or user.get("department", "")
    start, end = ask_shift((user.get("shift_start"), user.get("shift_end")) if user.get("shift_start") else None)
    fs.update_details(user["id"], employee_id, department, start, end)
    print(f"[SUCCESS] '{user['name']}' updated.")


@admin
def delete(fs):
    user = fs.find_user(input("Name or ID to delete: "))
    if not user:
        return print("[ERROR] User not found.")
    if input(f"Delete '{user['name']}'? (yes/no): ").strip().lower() == "yes":
        fs.delete_user(user["id"])
        print(f"[SUCCESS] '{user['name']}' removed.")
    else:
        print("[INFO] Cancelled.")


@admin
def list_users(fs):
    users = fs.users()
    if not users:
        return print("No users registered.")
    for u in users:
        print(f"  {u.get('employee_id') or '-':<10} {u['name']:<25} {u.get('department') or '-':<15} "
              f"shift {u.get('shift_start', '?')}-{u.get('shift_end', '?')}")


@admin
def show_logs(fs):
    for e in reversed(fs.logs(20)):
        score = f"{e['score']:.2f}" if e.get("score") is not None else ""
        print(f"  {e['ts'][:19].replace('T', ' ')}  {e['event']:<10} {e.get('user_name') or '':<20} {score:<5} {e.get('note') or ''}")


@admin
def attendance(fs):
    day = input(f"Date (Enter = today {date.today()}): ").strip() or date.today().isoformat()
    report = fs.attendance(day)
    print(f"\nAttendance {day}")
    for r in report["rows"]:
        print(f"  {r['name']:<22} {r['shift']:<12} in {r['check_in'] or '--:--'}  out {r['check_out'] or '--:--'}  "
              f"{', '.join(r['status'])}")
    s = report["summary"]
    print(f"Present {s['present']} | Late {s['late']} | Absent {s['absent']} | No check-out {s['no_checkout']} "
          f"| Unknown face with staff {s['tailgating']}")


@admin
def correct(fs):
    user = fs.find_user(input("Name or ID: "))
    if not user:
        return print("[ERROR] User not found.")
    day = input(f"Date (Enter = today {date.today()}): ").strip() or date.today().isoformat()
    field = "check_out" if input("Correct 1) check-in or 2) check-out? ").strip() == "2" else "check_in"
    fs.correct_attendance(user["id"], day, field, input("Time (e.g. 09:05): ").strip(),
                          input("Reason: ").strip())
    print("[SUCCESS] Correction saved.")


@admin
def export(fs):
    month = input(f"Month (Enter = {date.today():%Y-%m}): ").strip() or f"{date.today():%Y-%m}"
    print(f"[SUCCESS] Saved {fs.export_month(month)}")


def live(fs):
    cap = open_camera()
    print(f"[INFO] Live recognition started (logs once per person every {COOLDOWN_SECONDS:.0f}s). Press 'q' to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        for r in fs.recognize(frame):
            x1, y1, x2, y2 = r["box"]
            color = {"granted": (0, 200, 0), "denied": (0, 0, 230), "spoof": (180, 0, 180)}.get(r["status"], (0, 170, 230))
            label = {"granted": f"{r['name']} {r['score']:.2f}", "denied": "Access Denied",
                     "spoof": "Spoof detected"}.get(r["status"], r.get("hint") or "Checking...")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if r["logged"]:
                speak(f"Access granted to {r['name']}" if r["granted"] else "Access denied")
        cv2.imshow("FaceSecure - Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


MENU = [("Live recognition", live), ("Register employee", register), ("Re-capture face", update),
        ("Add a look (glasses, hijab, niqab, mask)", add_look),
        ("Edit ID / department / shift", edit), ("Delete employee", delete), ("List employees", list_users),
        ("Today's attendance (or any date)", attendance), ("Correct a check-in / check-out", correct),
        ("Export monthly attendance (CSV)", export), ("Recent access logs", show_logs)]


def main():
    fs = FaceSecure()
    while True:
        print("\n" + "=" * 50 + f"\nFACESECURE  ({fs.store.label})\n" + "=" * 50)
        for i, (label, _) in enumerate(MENU, 1):
            print(f"{i}. {label}")
        print("0. Exit")
        choice = input("Choice: ").strip()
        if choice == "0":
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            try:
                MENU[int(choice) - 1][1](fs)
            except FaceSecureError as e:
                print(f"[ERROR] {e}")
        else:
            print("[ERROR] Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
