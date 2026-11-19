# FaceSecure: Facial Recognition Access and Attendance System

FaceSecure is a face recognition system for office entry access and staff attendance, designed for a corporate setting in Pakistan. A camera at the entrance recognises registered employees, grants or denies access, checks that the face is a real person rather than a photo or phone screen, and builds daily attendance (check-in, check-out, late, left early, absent) from the same sightings. It runs fully offline on an ordinary laptop, and every face, log and photo stays on that machine.

## Contents
1. [Overview](#1-overview)
2. [My Role and How the Project Evolved](#2-my-role-and-how-the-project-evolved)
3. [Screenshots](#3-screenshots)
4. [How It Works](#4-how-it-works)
5. [The Methods](#5-the-methods)
6. [Attendance Rules](#6-attendance-rules)
7. [Privacy and Data Handling](#7-privacy-and-data-handling)
8. [Testing and Observations](#8-testing-and-observations)
9. [Research: Recognising Covered Faces (In Progress)](#9-research-recognising-covered-faces-in-progress)
10. [Technology Stack](#10-technology-stack)
11. [Installation and Running](#11-installation-and-running)
12. [Limitations and Future Work](#12-limitations-and-future-work)
13. [License](#13-license)
14. [Contact](#14-contact)

## 1. Overview
Many offices still record attendance on paper registers or with fingerprint machines that everyone has to touch. FaceSecure replaces both with a camera at the door:

| Who | What they get |
|---|---|
| Employees | Recognised as they walk in, with a spoken greeting. Nothing to touch and no card to carry. |
| The admin | Registers and removes employees, sets each person's shift, reviews every entry with a photo, corrects missed check-ins with a recorded reason, and exports the month's attendance to Excel. |
| Security | Every denied face and every spoofing attempt (a photo or phone screen held up to the camera) is logged with a photo. An unknown face entering right behind an employee is flagged as possible tailgating. |

At a glance:
- **Recognition:** InsightFace (SCRFD detector and ArcFace embeddings). A face is accepted when its cosine similarity to a registered profile is at least 0.55.
- **Multiple looks per person:** the same employee can be registered with glasses, hijab, niqab or a mask as separate looks, and the closest look is used.
- **Liveness:** a passive anti-spoofing model checks every face at the door without asking anyone to move. Registration also requires a real head turn.
- **Attendance:** first sighting of the day is check-in, last sighting is check-out, judged against each person's own shift.
- **Storage:** plain JSON and CSV files that open in Excel. No database server and no cloud.

## 2. My Role and How the Project Evolved
**The first version.** FaceSecure began as a command-line program. It could detect faces, register people into a PostgreSQL database and delete them, and it worked reliably from the terminal and VS Code.

**What went wrong.** When I tried to put a clean web interface on top of it, the project broke: the webcam loop, terminal prompts and database code were tangled together, so the interface could not call the recognition code without crashing.

**What I rebuilt, and why.** I rebuilt the project around a simple rule: the face logic lives in one place and the interface in another. I also dropped the database server. For a single office, files are simpler to set up, easier to inspect, and keep biometric data on one machine. Then I extended the scope from "recognise a face" to the problem an office actually has: who came in, when, whether they were late, and whether anyone tried to fool the camera.

| | First version | Current version |
|---|---|---|
| Interface | Command line only | Web interface plus command-line menu, both using the same core |
| Storage | PostgreSQL server | Local JSON and CSV files, one log per day |
| Access control | None | Admin-only registration and records, behind a hashed password |
| Spoofing | A photo on a phone screen was accepted | Passive liveness check at the door; head-turn and liveness checks at registration |
| Attendance | None | Shifts, check-in and check-out, late and left early, corrections, monthly export |

**Problems I found while testing, and how I fixed them:**

| Problem | Why it mattered | Fix |
|---|---|---|
| A photo shown on a phone screen was granted access. | Anyone with a colleague's photo could get in. | Added a passive liveness model at the door, plus head-turn and liveness checks at registration. |
| The same person was logged again every few seconds. | Logs filled with duplicates, and attendance became unreadable. | A person is logged once per arrival. A new arrival needs 60 seconds out of view. |
| Turning the head briefly made a registered person "unknown" and denied. | False alarms and wrong entries in the log. | A face is denied only after 4 unrecognised frames in a row; until then the screen shows "Checking…". |
| Profile photos and entry photos were not saved on Windows. | OpenCV's `imwrite` fails silently when the path contains a character like "–", as this project folder does. | Images are encoded in memory and written as bytes, which works with any path. |
| An admin could accidentally add a different person's face as someone's extra look. | That other person would then be let in under the wrong name. | A new look that is very unlike the person's other looks (similarity below 0.20) needs explicit confirmation, which is logged. |
| Recognition with niqab failed, and glasses gave borderline scores (about 0.57–0.61). | Employees who cover their faces could be wrongly denied. | Extra looks per person, and an ongoing research study (section 9). |

## 3. Screenshots
> **In progress — will be uploaded soon.**

Planned: the live door screen (granted, denied and spoof), guided registration, the People page with looks, the Attendance page and the Activity log. Screenshots will show only consented or blurred faces.

## 4. How It Works
```
Camera ─► SCRFD: find faces + 5 points ─► ArcFace: 512-number embedding ─► compare with every registered look
                                  │                                                     │
                                  └─► liveness model: real person or photo/screen?      ▼
                                                                    granted / denied / spoof / checking…
                                                                                        │
                                                   daily log (CSV) + photo ◄────────────┤
                                                   attendance (check-in, check-out) ◄───┘
```

1. **Detection:** SCRFD finds every face in the frame, along with five points (eyes, nose tip, mouth corners).
2. **Recognition:** ArcFace lines the face up using those five points and turns it into an embedding, a list of 512 numbers that describes the face.
3. **Registration:** the admin takes 5 guided photos (straight, left, right and so on). Their embeddings are averaged into one look. A face already registered under another name is rejected.
4. **Matching:** a live face is compared with every look of every employee using cosine similarity. The best match at 0.55 or above means access granted.
5. **Liveness:** a recognised face must also score at least 0.8 on the liveness model for 2 frames in a row. A face that looks like a photo or screen for 3 frames in a row is logged as a spoof attempt, with a photo.
6. **Logging:** one log entry per arrival, with a small photo of the face, written to that day's CSV file.
7. **Attendance:** each day's log is turned into check-in, check-out and status for every employee (section 6).

**Folder structure**
```
backend/    backend.py        face models, matching, liveness, attendance, file storage
            main.py           command-line menu: register / update / delete / live / logs / attendance
frontend/   app.py            web server for the demo (FastAPI + Uvicorn)
            FaceSecure.html   the whole UI (Live · Register · People · Attendance · Activity)
data/       created automatically on first run; never uploaded (see section 7)
requirements.txt
```

## 5. The Methods

### How a face becomes a match
ArcFace does not store photos and compare pixels. It turns each face into 512 numbers, trained so that two photos of the same person produce similar numbers and photos of different people do not. Cosine similarity measures how closely two of these lists point in the same direction: 1 means identical and 0 means unrelated. The threshold of 0.55 is a trade-off. Raising it lets fewer strangers in but turns away more employees on a bad-lighting day; lowering it does the opposite. For a door, letting in a stranger is the worse mistake, so the threshold leans towards caution.

### Why each person can have several looks
My first approach averaged all of a person's photos into one profile. That works for small changes, but to the model a face with a niqab and the same face uncovered are very different. Averaging them produces a profile that matches neither well. Instead, each look (glasses, hijab, niqab, mask) is stored separately, and a live face is compared with all of them and matched on the closest one. The cost is a small risk: every extra look is one more way a stranger could match, which is why a look that does not resemble the person's other looks needs confirmation from the admin.

### Two different liveness checks, because people behave differently at registration and at the door
A photo or phone screen is flat, and a real head is not. At **registration**, the admin asks the person to turn their head; the nose moves relative to the eyes in a way a flat photo cannot imitate, and the app measures that movement. At the **door**, nobody turns around for a camera; they just walk in. So the door uses a passive liveness model instead: a small neural network, trained to tell real faces from prints and screens, that looks at an 80×80 crop of the face and estimates the chance that it is a real person. Requiring 2 live frames in a row for access, and 3 fake frames in a row for a spoof alert, stops a single blurry frame from deciding either way.

### Why registration warns instead of blocking
The registration checks can be wrong: poor lighting can lower the liveness score of a real person. Blocking would leave the admin stuck. Instead, a failed check shows a clear warning, and the admin can only continue by confirming. The scores and the override are written to the log, so the decision is visible later rather than silently ignored.

### Logging arrivals, not frames
A camera sees each person many times per second. Logging every recognition would produce thousands of rows per person per day. The app logs a person once when they arrive and counts them as arriving again only after 60 seconds out of view. Unknown faces are treated more carefully: a registered person turning their head can briefly look unknown, so a face is only denied after 4 unrecognised frames in a row.

### Keeping the record honest
Attendance can be corrected, because people do forget to look at the camera. But a correction never edits history silently: it needs a reason, and it is stored in its own file and in the day's log with the time it was made. The original sightings are never changed.

## 6. Attendance Rules
- **Shift** chosen at registration: 9am–5pm (default), 9am–6pm, 10am–6pm, or other hours. It can be edited later.
- **Check-in** is the first time the person is recognised that day. **Check-out** is the last time, if at least 30 minutes after check-in.
- **Late** means check-in more than 15 minutes after the shift starts. **Left early** means check-out before the shift ends.
- **No check-out** means seen in the morning but never again, for example someone who left in a hurry. **Absent** means not seen at all on a working day (Monday to Friday). Weekends show as *Day off*, because staff use their keys for extra days.
- **Unknown face with staff:** an unregistered face seen within 3 seconds of a granted employee is flagged as someone possibly following staff in. Its photo is on the Activity page.
- **Corrections:** the admin can set a missed check-in or check-out on the Attendance page. A reason is required, and every correction is saved in `data/attendance/corrections.csv` and the daily log.
- **Export:** *Export month (CSV)* writes `data/attendance/attendance_YYYY-MM.csv`, which opens in Excel.

Times use the computer's clock, so set the PC to Pakistan time (PKT) where it is deployed.

## 7. Privacy and Data Handling
Face data is biometric data, so the design keeps it to the minimum and on one machine:

- **Local only.** Everything is stored on the laptop running the app. Nothing is sent to a server or cloud service, and after the first model download no internet connection is needed.
- **Embeddings, not photo galleries.** Recognition uses the 512-number embeddings. One profile photo per person is kept for the People page.
- **Automatic deletion.** Entry photos of granted employees are deleted after 30 days, and photos of denied faces and spoof attempts after 90 days. The log rows themselves are kept.
- **Admin only.** Registering, editing, deleting, attendance, corrections and logs all need the admin password. Only a salted hash of the password is stored (PBKDF2-SHA256, 200,000 iterations), and the session cookie is HTTP-only.
- **Never uploaded.** The `data/` folder excludes itself from Git, so no face data, photo, log or password hash can be pushed to GitHub by mistake.

```
data/users/<Name>/profile.json     name, ID, department, shift, dates, and one face embedding per look
data/users/<Name>/photo.jpg        profile photo
data/logs/2026-11-17.csv           one file per day: date, time, event, name, user_id, similarity, snapshot, note
data/snapshots/2026-11-17/*.jpg    entry photos (granted kept 30 days, denied and spoof 90 days)
data/attendance/                   corrections.csv and monthly exports
data/admin.json                    admin password (salted hash)
```
If a log is open in Excel while the app is running, new events are held in memory and written as soon as the file is closed.

## 8. Testing and Observations
These are functional checks and observations from development, not a formal accuracy evaluation. A measured study is in progress (section 9).

| Check | Result |
|---|---|
| A face shown on a phone screen, before liveness was added | Granted access, which is what prompted the liveness work |
| A simulated screen replay after liveness was added | Live score about 0.04, logged as a spoof on the 3rd frame |
| Registering from a flat photo | Head-turn and liveness warnings shown; registration only possible with a logged admin confirmation |
| Adding a different person's face as someone's extra look | Similarity −0.05 to the person's other looks; held until the admin confirms |
| A registered person turning their head at the door | No longer logged as unknown, thanks to the 4-frame confirmation |
| Wearing glasses | Recognised, with similarity about 0.57–0.61, close to the 0.55 threshold |
| Wearing a hijab | Recognised |
| Wearing a niqab | Not recognised, which is what prompted the research in section 9 |

## 9. Research: Recognising Covered Faces (In Progress)
> **In progress — code and results will be uploaded soon.**

**Research question:** how much do niqab, face masks and glasses reduce the accuracy of a face recognition access system, and can that loss be recovered without collecting anyone's real face?

The question comes directly from the observations above: the system recognised glasses only just above its threshold and did not recognise a niqab at all. In Pakistan, as in many other countries, covered faces at a workplace entrance are an everyday situation rather than an edge case.

**Planned study, in three parts:**
1. **How large is the drop?** Measure ArcFace on faces with increasing coverage, from uncovered to everything below the eyes hidden, plus glasses and headwear.
2. **Does registering an extra look recover it?** Compare registering a person from uncovered photos only with also registering a covered look, using this app's own looks feature. This needs no model training.
3. **Does fine-tuning help?** Fine-tune InsightFace's small recognition model on covered faces from training identities, then test on identities never seen in training, and check whether uncovered faces get worse.

**Measured by:** the true accept rate at a fixed false accept rate of 0.1% (how often the right person gets in when only 1 stranger in 1,000 would), and the same rates at this app's own threshold of 0.55. Training and test sets always use different identities.

**Data and consent.** The study uses **DigiFace-1M** (Microsoft, WACV 2023): synthetic faces rendered from 3D head scans of a small number of people who gave consent, published for non-commercial research. Masks and niqab are drawn onto the rendered faces by code; glasses and headwear come with the dataset. I considered several real-photo datasets and set them aside, either because the people in them did not consent (CelebA-Spoof uses web images of celebrities, and ENiqab-V2's photographs were collected from the web without recorded consent) or because access needs a signed agreement. For the same reason, no images of me or of people around me are used.

**Known limitation, stated up front:** everything is synthetic. The results will show the size and direction of the effect, not exact accuracy on real people wearing real niqabs.

## 10. Technology Stack
| Area | Tools |
|---|---|
| Face detection and recognition | InsightFace `buffalo_l` (SCRFD detector, ArcFace ResNet-50 recognition) |
| Anti-spoofing | InsightFace passive liveness model (80×80 CNN) |
| Inference | ONNX Runtime (CPU), OpenCV, NumPy |
| Web server | FastAPI, Uvicorn |
| Interface | HTML, CSS and JavaScript in one file: browser camera access (`getUserMedia`), canvas overlay, Web Speech API for the greeting |
| Command line | Python menu with `pyttsx3` voice |
| Storage | JSON profiles and daily CSV logs |
| Security | PBKDF2-SHA256 password hashing, HTTP-only session cookie |

## 11. Installation and Running
**Python version:** use Python 3.13. ONNX Runtime does not support Python 3.15 yet.

```
git clone https://github.com/SyedaNimraFatima/FaceSecure-Facial-Recognition-Access-System.git
cd FaceSecure-Facial-Recognition-Access-System
py -V:3.13 -m venv accessdoor
accessdoor\Scripts\activate
pip install -r requirements.txt
```
The first run downloads the face models (about 280 MB) and the liveness model (1.5 MB) to `C:\Users\<you>\.insightface`. After that, everything runs offline.

| What | Command |
|---|---|
| Demo UI | `python frontend\app.py`, which opens <http://localhost:8000> |
| Command-line menu | `python backend\main.py` |

The first time you click *Admin login* (or use an admin option in `main.py`), you create the admin password. Forgot it? Delete `data/admin.json` and set a new one.

**Using an iPhone as the camera**
1. Install a phone-webcam app on both the iPhone and the laptop, for example **Camo** (USB) or **Iriun Webcam** / **iVCam** (Wi-Fi or USB).
2. Start the app on both, then run FaceSecure.
3. In the UI, pick the iPhone from the **📷 camera menu** at the top right. The choice is remembered.
4. `main.py` automatically skips cameras that give no picture. To force one, set `FACESECURE_CAMERA` (for example `set FACESECURE_CAMERA=1`).

**Optional settings** (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `FACESECURE_THRESHOLD` | 0.55 | Similarity needed for access |
| `FACESECURE_COOLDOWN` | 60 | Seconds out of view before a person counts as arriving again |
| `FACESECURE_UNKNOWN_FRAMES` | 4 | Unrecognised frames in a row before a face is denied |
| `FACESECURE_GRACE` | 15 | Minutes after shift start before someone is late |
| `FACESECURE_MIN_STAY` | 30 | Minutes after check-in before a sighting counts as check-out |
| `FACESECURE_SNAPSHOT_GRANTED` | 1 | `0` = photos only for denied entries |
| `FACESECURE_GRANTED_PHOTO_DAYS` | 30 | Days granted-entry photos are kept |
| `FACESECURE_DENIED_PHOTO_DAYS` | 90 | Days denied and spoof photos are kept |
| `FACESECURE_LIVENESS` | 1 | `0` = liveness off |
| `FACESECURE_LIVE_THRESHOLD` | 0.8 | Live score needed at the door |
| `FACESECURE_HEAD_TURN` | 0.15 | Head movement needed at registration |
| `FACESECURE_MODEL` | buffalo_l | `buffalo_s` is faster on slow laptops |

## 12. Limitations and Future Work
**Limitations**
- **Covered faces:** InsightFace's models were trained mostly on uncovered faces, so niqab-covered faces may not be recognised and glasses lower the score. Extra looks help, and section 9 measures how much.
- **Liveness:** the model is a general pretrained baseline. It has not been evaluated on hijab, niqab or masks, may wrongly flag covered faces as spoofs, and a high-quality video replay may still pass.
- **No formal evaluation yet:** the results in section 8 are development checks on a handful of cases, not accuracy measured on a test set.
- **Lighting and angle:** accuracy drops in poor lighting and at strong side angles, so people should be registered in the conditions where the camera will be used.
- **One camera:** a single camera cannot tell entering from leaving, so first and last sighting are used. Separate entry and exit cameras would be more precise.
- **Groups:** a person whose face is fully hidden behind others is not seen at all; only a turnstile or one-at-a-time door prevents that.
- **Shifts** must start and end on the same day (no overnight shifts yet).

**Planned work**
- Complete and publish the covered-faces study (section 9) and add screenshots (section 3).
- Separate entry and exit cameras for exact in and out times.
- Overnight shifts and public-holiday calendars for Pakistan.
- An evaluation of the liveness model on covered faces.

## 13. License
© 2026 Nimra Fatima. All rights reserved.

This code is shared for viewing and evaluation only. It may not be copied, modified, distributed or used, in whole or in part, without my prior written permission. To request permission, email inimrasyed@gmail.com.

The InsightFace pretrained models (`buffalo_l` and the liveness model) are not included in this repository. They are downloaded on first run and are licensed by InsightFace for non-commercial research use only. DigiFace-1M (section 9) is licensed by Microsoft for non-commercial research use.

## 14. Contact
Nimra Fatima · inimrasyed@gmail.com · GitHub: [SyedaNimraFatima](https://github.com/SyedaNimraFatima)
