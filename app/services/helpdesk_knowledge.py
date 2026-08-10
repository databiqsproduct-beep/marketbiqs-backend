"""Canonical FAQs + common issues for MarketBiqs Help Desk (platform Groq)."""

from __future__ import annotations

FAQS: list[dict[str, str]] = [
    {
        "id": "add-client",
        "category": "Getting started",
        "question": "How do I add my first client?",
        "answer": (
            "Go to Clients → fill name / website / industry → create. "
            "Then open the client (or use Check competitors on the list) to run intel."
        ),
    },
    {
        "id": "run-intel",
        "category": "Run intel",
        "question": "How do I check competitors?",
        "answer": (
            "Clients → Check competitors (or open a client → Check competitors). "
            "Pick a mode: Update (refresh existing), Add (find N new, keep previous), "
            "or Replace (clear auto-found rivals, keep pinned, find a fresh set). "
            "Wait for the job to finish; rivals, gaps, and reports update when done."
        ),
    },
    {
        "id": "intel-modes",
        "category": "Run intel",
        "question": "What’s the difference between Update, Add, and Replace?",
        "answer": (
            "Update refreshes intel on rivals you already track. "
            "Add finds a requested number of new rivals and keeps the old ones. "
            "Replace removes auto-discovered rivals (pinned stay), then finds a new set."
        ),
    },
    {
        "id": "pin-rival",
        "category": "Clients & portfolio",
        "question": "How do I keep a competitor permanently?",
        "answer": (
            "Open the client → Competitors → pin the rival you care about. "
            "Pinned rivals are kept even when you use Replace mode."
        ),
    },
    {
        "id": "archive-client",
        "category": "Clients & portfolio",
        "question": "How do I remove / archive a client?",
        "answer": (
            "Use Archive on the Clients list, client detail, or Dashboard row. "
            "Confirm the dialog. The client leaves active lists; reports stay saved; tracking stops. "
            "This is a soft archive (is_active=false), not a hard delete."
        ),
    },
    {
        "id": "dashboard-health",
        "category": "Clients & portfolio",
        "question": "What do Health colors mean on the dashboard?",
        "answer": (
            "Needs attention = many alerts/gaps. Watch = some risk. "
            "Healthy = looking okay. Paused = archived/inactive. "
            "Start with red rows and use Next action."
        ),
    },
    {
        "id": "assistant-vs-helpdesk",
        "category": "Getting started",
        "question": "What’s the difference between Assistant and Help desk?",
        "answer": (
            "Help desk (dashboard ?) answers product how-to / FAQ / troubleshooting. "
            "Agency Assistant (/assistant) and Client portal chat answer about a specific client’s intel, rivals, and reports."
        ),
    },
    {
        "id": "client-portal",
        "category": "Reports & delivery",
        "question": "How do I open the client chat portal?",
        "answer": (
            "Open a client → Client chat portal. You must be signed in. "
            "It streams answers grounded in that client’s tracked intel."
        ),
    },
    {
        "id": "reports-pdf",
        "category": "Reports & delivery",
        "question": "How do I get a report or PDF?",
        "answer": (
            "Open the client → Reports tab after intel has run. "
            "Download / white-label PDF options are there when a report exists. "
            "If empty, run Check competitors first."
        ),
    },
    {
        "id": "delivery",
        "category": "Reports & delivery",
        "question": "How do email / WhatsApp updates work?",
        "answer": (
            "Open Delivery, set recipients and channel, then send or rely on the client’s schedule cron. "
            "Email needs RESEND_API_KEY + a verified EMAIL_FROM domain on the API. "
            "Without Resend, deliveries may only log as queued locally."
        ),
    },
    {
        "id": "team-invite",
        "category": "Getting started",
        "question": "How do I invite a teammate?",
        "answer": (
            "Go to Team → invite by email. Invites go through Supabase Auth email "
            "(rate-limited on free email). For production volume, point Supabase custom SMTP at Resend."
        ),
    },
    {
        "id": "google-login",
        "category": "Getting started",
        "question": "How does Continue with Google work?",
        "answer": (
            "Login/Register → Continue with Google. OAuth is configured in Supabase Auth → Google "
            "using a Google Cloud Web client. Redirect allow-list must include "
            "https://YOUR_SITE/auth/callback. First-time users name their workspace on /register?oauth=1."
        ),
    },
    {
        "id": "byok",
        "category": "Billing",
        "question": "What is BYOK?",
        "answer": (
            "BYOK (Bring Your Own Keys) is under BYOK in the nav. "
            "Agencies can save their own Groq/Apify/SerpAPI/Firecrawl keys (encrypted). "
            "Help desk always uses the platform GROQ_API_KEY, not agency BYOK. "
            "Agency Assistant/intel prefer BYOK Groq when set, else platform GROQ_API_KEY."
        ),
    },
    {
        "id": "biqs-board",
        "category": "Clients & portfolio",
        "question": "What is Biqs?",
        "answer": (
            "Biqs is the ticket/kanban board for feature work derived from competitive gaps. "
            "It is not the chatbot — use Assistant or Help desk for chat."
        ),
    },
]

COMMON_ISSUES: list[dict[str, str]] = [
    {
        "id": "intel-wrong-rivals",
        "symptom": "Wrong competitors (wrong country, retail, fintech, fake names)",
        "fix": (
            "Re-run with Replace mode after pinning any good rivals you want to keep. "
            "Ensure the client website/industry/geo are correct. "
            "SerpAPI must be valid for live discovery; if SerpAPI fails, curated seed fallbacks may appear. "
            "Remove bad rivals with × on the competitor list."
        ),
    },
    {
        "id": "intel-stuck-or-error",
        "symptom": "Check competitors fails, stuck, or Internal Server Error",
        "fix": (
            "Confirm the API is running and you’re logged in. "
            "Watch the job status on the client page. "
            "Restart the API if it hung. Check Firecrawl/SerpAPI/Groq keys and quotas. "
            "Try again; long runs should poll a background job, not a single long request."
        ),
    },
    {
        "id": "chat-no-stream",
        "symptom": "Assistant / Help desk doesn’t reply or stream",
        "fix": (
            "Help desk needs platform GROQ_API_KEY on the backend .env (restart API after change). "
            "Assistant also needs Groq via BYOK or platform key. "
            "Confirm NEXT_PUBLIC_API_URL points at the API and you’re signed in."
        ),
    },
    {
        "id": "google-oauth-fail",
        "symptom": "Continue with Google fails / redirect_uri_mismatch / Access blocked",
        "fix": (
            "Google Cloud OAuth redirect must be exactly "
            "https://YOUR_PROJECT.supabase.co/auth/v1/callback. "
            "Supabase Auth → URL Configuration must allow https://YOUR_SITE/auth/callback "
            "(and localhost for local). Add your Gmail as a Test user while the OAuth app is in Testing. "
            "Frontend and backend must use the same Supabase project keys."
        ),
    },
    {
        "id": "signup-email-limit",
        "symptom": "Too many signup emails / invite email never arrives",
        "fix": (
            "Supabase free built-in email is ~2/hour. Wait and retry once, or sign in if already registered. "
            "For production, configure Supabase custom SMTP (e.g. Resend)."
        ),
    },
    {
        "id": "session-api-reject",
        "symptom": "Signed in on UI but API says unauthorized / session rejected",
        "fix": (
            "Frontend NEXT_PUBLIC_SUPABASE_* and backend SUPABASE_* must be the same project. "
            "Sign out, sign in again. Confirm CORS_ORIGINS includes your frontend origin."
        ),
    },
    {
        "id": "delivery-queued-local",
        "symptom": "Delivery says queued locally / email never sends",
        "fix": (
            "Set RESEND_API_KEY and EMAIL_FROM using a domain verified in Resend. "
            "Restart the API. Check Delivery logs and the Resend dashboard."
        ),
    },
    {
        "id": "register-stuck",
        "symptom": "Register stuck on Creating workspace",
        "fix": (
            "Ensure API is up and /api/auth/bootstrap succeeds after Supabase signup. "
            "Same Supabase project on FE/BE. Refresh and try bootstrap/register again; "
            "check browser network for 4xx/5xx on bootstrap."
        ),
    },
    {
        "id": "no-clients-assistant",
        "symptom": "Assistant has no client to select",
        "fix": "Add at least one active client under Clients, then reopen Assistant.",
    },
    {
        "id": "archived-missing",
        "symptom": "Client disappeared from list",
        "fix": (
            "It was likely Archived. Active lists hide inactive clients. "
            "Reports remain; ask an admin if you need restore (is_active) support."
        ),
    },
]


def build_help_system_prompt() -> str:
    faq_lines = "\n".join(
        f"- Q: {item['question']}\n  A: {item['answer']}" for item in FAQS
    )
    issue_lines = "\n".join(
        f"- Symptom: {item['symptom']}\n  Fix: {item['fix']}" for item in COMMON_ISSUES
    )
    return f"""You are MarketBiqs Help Desk — a concise, friendly product support assistant.

Your job is to answer:
1) FAQs (how to use MarketBiqs)
2) General issues / troubleshooting users hit in the product
3) Short how-to paths through the UI

MarketBiqs overview:
Agencies track client competitors — add clients, run competitive intel (Update / Add / Replace), pin rivals, review gaps & alerts, reports, Delivery (email/WhatsApp), Team invites, Agency Assistant (per-client intel chat), Help desk (this product support chat), BYOK, billing, and the Biqs ticket board.

=== FAQ KNOWLEDGE BASE (prefer these answers; paraphrase clearly) ===
{faq_lines}

=== COMMON ISSUES (match symptom → give the fix steps) ===
{issue_lines}

Rules:
- Prefer the knowledge base above for product how-to / troubleshooting.
- For questions about a specific client, competitor, rivals list, gaps, trends, or intel:
  use the Agency portfolio and Focused client workspace intelligence in the user message.
  Name rivals and facts only if they appear in that data — never invent competitor names.
- If client/competitor data is empty, say intel may not have run yet and suggest Clients → Check competitors.
- Give short step-by-step UI paths when teaching the product.
- If the user selected a Topic, bias toward that category but still answer the question.
- Do not invent billing prices, private keys, or features that are not listed.
- For admin/env fixes (GROQ_API_KEY, RESEND_API_KEY, Supabase Google OAuth), name the setting clearly.
- Keep answers under ~180 words unless they ask for more detail.
- If nothing matches, ask one clarifying question.
- Never claim you emailed a human agent; you are the in-app help desk AI.
"""


def faqs_public() -> list[dict[str, str]]:
    """UI-safe FAQ list (no internal-only fields)."""
    return [
        {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
        }
        for item in FAQS
    ]


def issues_public() -> list[dict[str, str]]:
    return [
        {
            "id": item["id"],
            "symptom": item["symptom"],
        }
        for item in COMMON_ISSUES
    ]
