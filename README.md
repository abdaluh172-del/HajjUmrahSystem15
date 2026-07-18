# Hajj & Umrah Sentiment Analysis — Backend (Flask + real ML model)

This is a **real** working backend: a scikit-learn model (TF-IDF + Multinomial
Naive Bayes, trained on `dataset.py`) served through a Flask REST API, backed
by a SQLite database (`hajj_umrah.db`, created automatically on first run).

It matches the design in the graduation project document (Chapter 1.6
Methodology: TF‑IDF feature extraction + Naive Bayes classification).

## 1. Open in VS Code
Open this `backend` folder in VS Code (`File → Open Folder…`).
Make sure the **Python extension** is installed.

## 2. Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Activate it:
- Windows (PowerShell): `venv\Scripts\Activate.ps1`
- Windows (cmd): `venv\Scripts\activate.bat`
- macOS / Linux: `source venv/bin/activate`

VS Code may prompt "Select Interpreter" — choose the one inside `venv`.

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. (Optional) Retrain the model
A model is trained automatically the first time you run `app.py` if
`model.pkl` doesn't exist yet. To retrain manually (e.g. after editing
`dataset.py`) and see accuracy/precision/recall:

```bash
python train_model.py
```

> The included dataset is intentionally small (~90 labeled examples) for
> demonstration. Accuracy will be limited — for a stronger model, replace
> `TRAIN_DATA` in `dataset.py` with a larger, real labeled dataset (hundreds/
> thousands of comments), which is exactly what your project proposes to
> collect in Graduation Project 2.

## 5. Run the API — and the full website
```bash
python app.py
```
The API starts at **http://localhost:5000** and auto-creates/seeds
`hajj_umrah.db` on first run.

**Open http://localhost:5000 in your browser** — this now serves a complete,
real, working website (login → dashboard → analyze comment → comments →
settings), built with React (via CDN, no npm/build step needed) and wired
directly to this Flask API and the trained ML model. Every comment you
analyze is saved for real in `hajj_umrah.db`.

**Default login** (auto-seeded on first run): `abdullah2222@ghjj.sa` / `A1231234` — display name: **Abdullah Alharbi**. Logged-in sessions are remembered in the browser (localStorage) for 12 hours.

## What's new in v9
- **Login Logs (admin-only):** every sign-in attempt (and every signup) is stored
  permanently in the database — name, email, date, time and status
  (success/failed). A new **Login Logs** page in the admin sidebar shows them
  with search, sorting (newest/oldest/name/email/status) and pagination.
  API: `GET /api/login-logs` (admin token required).
- **Comments Language (admin-only, in Settings):** choose Original / Arabic /
  English — all displayed comments are translated automatically in the browser
  to the chosen language (originals in the database are never modified).
- **Admin-only tools:** Export CSV (Comments), Print / Save PDF (Reports) and
  the whole **Analytics** page are now visible to the admin only. A non-admin
  reaching Analytics, Login Logs or Users sees an **Access Denied** screen.
  `GET /api/comments/export` now requires the admin token on the server too.
- **Comments language (v11 — per account):** in Settings, EVERY signed-in
  account (regular users AND the admin) can choose the comments display
  language — Original / Arabic / English. Displayed comments are translated
  automatically in the browser; originals in the database are never modified.
  The choice is personal — saved on the account's own row via
  `PUT /api/me/comment-lang`, so it never affects other users, is restored
  on every login (any device) and survives restarts/redeploys. Guests don't
  have an account, so they always see originals.

### Permanent data storage (v10 — no Persistent Disk needed)
- **All data — user accounts, the admin account, comments and login logs — is
  stored in an external PostgreSQL database** referenced by the `DATABASE_URL`
  environment variable. Nothing is kept in code variables or app memory.
- Because the database lives **outside the app's filesystem**, closing the
  app, a **Restart** or a **Redeploy** on Render's **free plan** never
  deletes anything. Accounts and comments are removed only when the admin
  deletes them manually from the dashboard.
- **Setup (one time):**
  1. Create a free PostgreSQL database on **Neon** (https://neon.tech) —
     free and permanent. *Do not use Render's own free PostgreSQL: it is
     automatically **deleted after 30 days**.* (Supabase is another option.)
  2. Copy the connection string, e.g.
     `postgresql://user:pass@host/dbname?sslmode=require`
  3. In Render → your service → **Environment** → add
     `DATABASE_URL` = that connection string → Save (the service redeploys).
- On first start with an empty database, the app creates the tables, seeds
  the sample comments and the fixed admin automatically. After that, startup
  **never touches existing data** (a changed admin password/name survives
  every restart and redeploy).
- Without `DATABASE_URL` the app falls back to a local SQLite file — for
  local development only (on Render's free plan that file is wiped on
  every redeploy).
You can also use "Sign up" on the login page to create a new account —
accounts are real rows in the `users` table, hashed with werkzeug's
`generate_password_hash` (not plaintext). "Forgot password" is simulated —
it confirms the flow but does not send a real email (no SMTP server is
configured; wire up Flask-Mail or similar for that).

## 6. Test it
```bash
curl http://localhost:5000/api/health

curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"الازدحام كان شديد جداً وتأخير في كل شيء\"}"

curl http://localhost:5000/api/comments?per_page=5
curl http://localhost:5000/api/dashboard/stats
```
Or open `http://localhost:5000/api/health` directly in a browser, or use
Postman / VS Code's REST Client extension.

## 7. Connect the React frontend
In the React app (`HajjUmrahSystem.jsx`), replace the client-side
`analyzeText()` calls and the in-memory `comments` state with real `fetch`
calls to this API, e.g.:

```js
const res = await fetch("http://localhost:5000/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
});
const result = await res.json();
```

Do the same for `/api/comments` (GET/POST/PUT/DELETE) and
`/api/dashboard/stats`. This turns the current in-browser demo into a
frontend properly talking to a real backend + real ML model + real
database, as required by the project scope.

## Endpoints reference
| Method | Endpoint                 | Purpose                                |
|--------|---------------------------|-----------------------------------------|
| GET    | `/api/health`             | Check server + model status            |
| POST   | `/api/analyze`            | Analyze a comment (no DB write)        |
| GET    | `/api/comments`           | List/search/filter/sort/paginate       |
| POST   | `/api/comments`           | Add + analyze + store a new comment    |
| PUT    | `/api/comments/<id>`      | Edit a comment (re-analyzed)           |
| DELETE | `/api/comments/<id>`      | Delete a comment                       |
| GET    | `/api/comments/export`    | CSV export (`?format=csv`)             |
| GET    | `/api/dashboard/stats`    | Aggregated stats for charts            |

## Not included (needs further work for production)
- **Authentication (JWT) / roles** — currently the API is open; add
  `flask-jwt-extended` and per-route `@jwt_required()` checks.
- **Larger training dataset** — the model is a real, working classifier, but
  its accuracy is limited by the small illustrative dataset provided.
- **Deployment** — this runs Flask's development server; use `gunicorn` +
  a reverse proxy (nginx) for production.

## Open it from your phone (same Wi-Fi network)
The server listens on `0.0.0.0`, so other devices on the **same Wi-Fi** can
reach it too:

1. On the PC, find its local IP: open PowerShell and run `ipconfig`, look
   for **IPv4 Address** (e.g. `192.168.1.23`).
2. Make sure Windows Firewall allows inbound connections on port 5000 for
   Python (Windows may prompt you the first time you run the server — allow
   it for Private networks).
3. On the phone (connected to the same Wi-Fi), open:
   `http://192.168.1.23:5000` (use your PC's actual IP, not this example).

**Security note:** `debug=True` + `host="0.0.0.0"` exposes Werkzeug's
interactive debugger to everyone on your network — fine for a local
demo/graduation project on a trusted home/campus Wi-Fi, but set
`debug=False` (or don't bind `0.0.0.0`) before using this on any network
you don't fully trust.


## Roles & Permissions (Admin / User / Guest)

The system has exactly three roles:

| Capability | Admin (fixed email only) | Registered User | Guest (no account) |
|---|---|---|---|
| Dashboard, Analytics, Reports | Yes | Yes | Yes |
| View comments | Yes | Yes | Yes |
| Run the analyzer | Yes | Yes | Yes (result not saved) |
| Add comments | Yes | Yes | **No — must sign up** |
| Delete / edit comments | **Yes — admin only** | No | No |
| Users page (see registered emails) | **Yes — admin only** | Hidden + blocked | Hidden + blocked |

- **Fixed admin account:** `abdullah2222@ghjj.sa` / `A1231234` — recreated automatically on startup,
  cannot be deleted, demoted, or have its email changed (enforced server-side).
  These credentials are **no longer shown on the login page** — they live only here and in `app.py`.
- **The admin role is exclusive to that one email.** It can never be assigned through signup or the
  users page (`role` accepts only `user`/`guest`), and on every startup any other row that somehow
  has `role='admin'` is demoted to `user`. Admin endpoints double-check both the role **and** the email.
- **Sign up** always creates a regular `user` account (can view and add comments).
- **Continue as Guest:** the login page has a guest button (`POST /api/auth/guest`) — no account, view only.
  `POST /api/comments` requires a valid `user`/`admin` token, so guests get `401` with an Arabic
  "create an account" message until they register.
- **Admin protection:** login returns a signed token (12h expiry). All `/api/users` endpoints and
  `PUT/DELETE /api/comments/<id>` require `Authorization: Bearer <token>` from the fixed admin;
  everything else gets `403`.
- **Migration:** old databases are upgraded automatically on startup — previously registered accounts
  stored as `guest` become `user`, and any stray admin rows are demoted.
- Set a real `SECRET_KEY` environment variable in production.
