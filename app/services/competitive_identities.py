"""Domain, URL, and Brand Identity normalization and noise filtering.

Provides brand extraction, display name sanitization, fake/noise detection,
and identity deduplication keys.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.core.serp_noise import SERP_NOISE_DOMAINS, is_serp_noise_domain

_is_serp_noise_domain = is_serp_noise_domain


# ---------------------------------------------------------------------------
# Basic primitive coercions
# ---------------------------------------------------------------------------

def _clip(value: str | None, max_len: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _level_label(value: str | None, default: str = "medium", *, max_len: int = 40) -> str:
    """Normalize AI severity labels; allow short free text up to max_len."""
    raw = _as_str(value, default).strip()
    lowered = raw.lower()
    if lowered in {"low", "medium", "high", "critical"}:
        return lowered
    if not raw:
        return default
    return _clip(raw, max_len)


_WEAK_COMPARISON_MARKERS = (
    "none",
    "n/a",
    "does not have a similar feature",
    "do not have a similar feature",
    "both companies have similar features",
    "continue to enhance and promote",
    "continue to monitor and improve",
    "no similar feature",
    "similar features",
)


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _as_int(value: Any, default: int | None = None) -> int | None:
    """AI returns story points as 5, '5', '5 points', or '3-5' — keep the first integer."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", _as_str(value))
    return int(match.group()) if match else default


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _is_generic_text(value: Any) -> bool:
    text = _as_str(value).strip().lower()
    if len(text) < 12:
        return True
    return any(marker in text for marker in _WEAK_COMPARISON_MARKERS)


# ---------------------------------------------------------------------------
# URL and Domain Utilities
# ---------------------------------------------------------------------------

def _domain_of(url: str) -> str:
    raw = _as_str(url).strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_website(url: str | None) -> str | None:
    """Store absolute https URLs only; drop junk that cannot open in a browser."""
    raw = _as_str(url).strip()
    if not raw:
        return None
    # Reject placeholders / obvious non-URLs
    lowered = raw.lower()
    if lowered in {"n/a", "na", "none", "null", "-", "tbd", "unknown"}:
        return None
    if " " in raw or "\n" in raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    host = _domain_of(raw)
    if not host or "." not in host:
        return None
    # Reject bare TLDs / IP-less junk hosts
    if host.count(".") < 1 or host.endswith("."):
        return None
    # Strip ad/tracking query + fragments — UTMs confuse geo profiling
    raw = raw.split("#", 1)[0]
    if "?" in raw:
        base, _, qs = raw.partition("?")
        keep: list[str] = []
        for part in qs.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in {
                "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "device", "placement",
                "gad_source", "gad_campaignid",
            }:
                continue
            if part:
                keep.append(part)
        raw = f"{base}?{'&'.join(keep)}" if keep else base
    return raw.rstrip("/")


# ---------------------------------------------------------------------------
# Identity Keys and Self-Rival Detection
# ---------------------------------------------------------------------------

_GENERIC_RIVAL_NAME_TAILS = (
    "pizzas", "pizza", "burgers", "burger", "shawarmas", "shawarma",
    "restaurants", "restaurant", "limited", "ltd", "inc", "corp",
    "company", "pakistan", "pk",
)


def _rival_name_key(name: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", _as_str(name).lower())
    while compact:
        stripped = False
        for tail in _GENERIC_RIVAL_NAME_TAILS:
            if compact.endswith(tail) and len(compact) - len(tail) >= 4:
                compact = compact[: -len(tail)]
                stripped = True
                break
        if not stripped:
            break
    return compact


def _rival_host_key(website: str | None) -> str:
    host = _domain_of(website or "")
    if not host:
        return ""
    skip = {"com", "net", "org", "pk", "biz", "co", "uk", "ae", "io", "dev", "info", "app", "www"}
    labels = [part for part in host.split(".") if part and part not in skip]
    core = labels[0] if labels else host.split(".")[0]
    return _rival_name_key(core)


def _rival_keys(name: str, website: str | None = None) -> set[str]:
    keys = set()
    name_key = _rival_name_key(name)
    if name_key:
        keys.add(name_key)
    host_key = _rival_host_key(website)
    if host_key:
        keys.add(host_key)
    return keys


def _is_self_rival(
    client_name: str,
    rival_name: str,
    *,
    website: str | None = None,
    client_website: str | None = None,
) -> bool:
    """True when rival is the client itself (or a review/title about the client)."""
    client = _as_str(client_name).strip()
    rival = _as_str(rival_name).strip()
    if not client or not rival:
        return False
    c_key = _rival_name_key(client)
    r_key = _rival_name_key(rival)
    if c_key and r_key:
        if c_key == r_key:
            return True
        if len(c_key) >= 6 and (c_key in r_key or (len(r_key) >= 6 and r_key in c_key)):
            return True
    c_tokens = [t for t in re.split(r"[^a-z0-9]+", client.lower()) if len(t) >= 3]
    rival_l = rival.lower()
    if c_tokens and all(tok in rival_l for tok in c_tokens):
        return True
    ch = _domain_of(client_website or "")
    rh = _domain_of(website or "")
    if ch and rh:
        if ch == rh:
            return True
        ch_parts = ch.split(".")
        rh_parts = rh.split(".")
        ch_base = ".".join(ch_parts[-3:]) if len(ch_parts) >= 3 and ch_parts[-2] in {"edu", "com", "org", "gov", "net", "ac", "co"} else (".".join(ch_parts[-2:]) if len(ch_parts) >= 2 else ch)
        rh_base = ".".join(rh_parts[-3:]) if len(rh_parts) >= 3 and rh_parts[-2] in {"edu", "com", "org", "gov", "net", "ac", "co"} else (".".join(rh_parts[-2:]) if len(rh_parts) >= 2 else rh)
        if ch_base and rh_base and ch_base == rh_base:
            return True
    _GENERIC_SELF_WORDS = {
        "shawarma", "shwarma", "pizza", "pizzas", "burger", "burgers", "cafe", "cafes",
        "restaurant", "restaurants", "bakery", "bakeshop", "bakes", "grill", "bar",
        "lounge", "kitchen", "diner", "bistro", "eatery", "food", "foods", "sweets",
        "fashion", "apparel", "clothing", "wear", "boutique", "textiles", "collection",
        "agency", "consulting", "solutions", "tech", "technologies", "software", "systems",
        "services", "studio", "media", "digital", "labs", "group", "holdings", "global",
        "law", "legal", "firm", "associates", "partners", "health", "healthcare", "care",
        "clinic", "dental", "pharma", "pharmacy", "medical", "hospital", "academy", "school",
    }
    distinctive_tokens = [t for t in c_tokens if t not in _GENERIC_SELF_WORDS and len(t) >= 4]
    if rh and distinctive_tokens:
        rh_core = rh.split(".")[0]
        if any(tok in rh_core for tok in distinctive_tokens) or (len(c_tokens) >= 2 and all(tok in rh_core for tok in c_tokens)):
            return True
    return False


def _find_matching_competitor(rows: list, name: str, website: str | None = None):
    keys = _rival_keys(name, website)
    if not keys:
        return None
    for row in rows:
        row_name = getattr(row, "name", None) or (row.get("name") if isinstance(row, dict) else "")
        row_web = getattr(row, "website", None) if not isinstance(row, dict) else row.get("website")
        if _rival_keys(_as_str(row_name), row_web) & keys:
            return row
    return None


# ---------------------------------------------------------------------------
# Display Name Cleaning and Brand Extraction
# ---------------------------------------------------------------------------

_RECIPE_OR_DISH_TITLE_MARKERS = (
    "recipe", "homemade", "street style", "how to make", "how to cook",
    "ingredients", "step by step", "cooking tutorial", "oven baked",
    "crispy fried", "easy chicken", "best chicken", "complete shawarma menu",
)
_RECIPE_STYLE_TOKENS = (
    "authentic", "street", "style", "homemade", "pakistani", "lebanese", "turkish",
    "arabic", "spicy", "crispy", "juicy", "delicious", "traditional", "classic",
    "chicken", "beef", "mutton", "lamb", "garlic", "loaded", "stuffed",
)
_BARE_DISH_NAME_RE = re.compile(
    r"^(chicken|beef|mutton|lamb|spicy|garlic)?\s*"
    r"(shawarma|pizza|burger|biryani|karahi|broast|pasta|noodles|wrap|roll)"
    r"(\s+(platter|wrap|roll|sandwich|meal|combo|special))?$",
    re.I,
)


def _clean_rival_display_name(name: str) -> str:
    """Strip SEO taglines & slogans: 'Hot & Fresh from Caprinos ...' → 'Caprinos'."""
    raw = _as_str(name).strip()
    if not raw:
        return raw
    # Strip trailing ellipses / dots / snippet truncation
    cleaned = re.sub(r"\s*(\.{2,}|…)\s*$", "", raw).strip()
    cleaned = re.split(r"\s*[|:–—]\s+", cleaned, maxsplit=1)[0].strip()
    # Strip leading marketing slogan hooks / preambles: "Hot & Fresh from Caprinos" -> "Caprinos"
    cleaned = re.sub(
        r"^(hot\s*(&|and)?\s*fresh|fresh\s*(&|and)?\s*hot|freshly\s+baked|order\s+online|order\s+now|welcome\s+to|discover|experience|enjoy|savor|taste\s+the\s+best\s+of|the\s+best)\s+(from|at|by|of|to)?\s*",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    cleaned = re.sub(r"^(from|at|by)\s+", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(
        r"\s+[–—-]\s+(unique|authentic|fresh|order|best|home|menu|delivery|official|shop|online)\b.*$",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    cleaned = re.sub(
        r"\s+(official\s+(clothing\s+|online\s+)?store|official\s+website|online\s+store(\s+in\s+pakistan)?|clothing\s+store(\s+in\s+pakistan)?|clothing\s+brand(\s+in\s+pakistan)?|flagship\s+store)$",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    cleaned = re.sub(r"\s+ready[\s-]+to[\s-]+wear$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*(\.{2,}|…)\s*$", "", cleaned).strip()
    return cleaned or raw


def _looks_like_recipe_or_menu_item_name(name: str) -> bool:
    """True for dish/recipe titles like 'Authentic Pakistani Street Style Chicken Shawarma'."""
    raw = _as_str(name).strip()
    if not raw:
        return True
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    words = key.split()
    if any(m in key for m in _RECIPE_OR_DISH_TITLE_MARKERS):
        return True
    if _BARE_DISH_NAME_RE.match(key):
        return True
    if len(words) >= 5 and any(
        d in key for d in ("shawarma", "pizza", "burger", "biryani", "karahi", "broast", "pasta")
    ):
        style_hits = sum(1 for tok in _RECIPE_STYLE_TOKENS if tok in words)
        if style_hits >= 3:
            return True
    if len(words) >= 6 and key.startswith(
        ("authentic ", "delicious ", "homemade ", "easy ", "best homemade ", "ultimate ")
    ):
        return True
    return False


_GENERIC_RIVAL_NAMES = {
    "techcorp", "tech corp", "tech-corp",
    "softcorp", "soft corp",
    "softsolutions", "soft solutions", "soft-solutions",
    "paktech", "pak tech", "paktech solutions", "pak tech solutions",
    "axonsoft", "axon soft",
    "techsoft", "tech soft",
    "infotech solutions", "info tech solutions",
    "global tech", "smart tech", "future tech", "nextgen tech", "next gen tech",
    "software solutions", "it solutions", "tech solutions", "digital solutions",
    "software house", "it company", "tech company", "software company",
    "abc tech", "xyz tech", "test company", "demo company",
    "about us", "about", "contact us", "contact", "home", "homepage", "our story",
    "who we are", "privacy policy", "terms", "terms of service", "terms & conditions",
    "locations", "our locations", "branches", "outlets", "menu", "our menu", "careers",
    "reviews", "customer reviews", "login", "sign in", "faq", "faqs", "order online", "online ordering",
}
_GENERIC_NAME_RE = re.compile(
    r"^(tech|soft|pak|info|digital|global|smart|future|nextgen|next\s*gen|axon)"
    r"[\s\-]?(corp|soft|tech|solutions|systems|company|house)$",
    re.I,
)
_FAKE_FOOD_NUMBER_NAME_RE = re.compile(
    r"^(pizza|burger|cafe|café|shawarma|broast|biryani|karahi)\s*[\-]?\s*"
    r"(\d{1,4}|2\s*go|to\s*go|express|hub|zone|point|spot|king|queen)$",
    re.I,
)
_FAKE_FOOD_WORD_NAME_RE = re.compile(
    r"^(four\s*twenty\s*four|hot\s*and\s*spicy|pizza\s*mania|pizza\s*house)$",
    re.I,
)


def _is_generic_or_fake_rival_name(name: str) -> bool:
    """Block LLM placeholder brands like TechCorp / Soft Solutions / PakTech Solutions."""
    raw = _as_str(name).strip()
    if not raw:
        return True
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", key)
    if _looks_like_recipe_or_menu_item_name(raw):
        return True
    # Generic e-commerce navigation, CTAs, and collection titles
    if re.search(
        r"^(shop|buy|order|discover|explore|browse|view)\s+(latest|online|now|all|new|collection|dresses|women|men|pret|luxury)\b",
        key,
    ):
        return True
    if re.search(
        r"^(women\'?s\s+|men\'?s\s+)?((luxury|luxe)\s+pret|pret(\s+wear)?|ready[\s-]+to[\s-]+wear|unstitched|formal\s+wear|casual\s+wear|party\s+wear|bridal\s+wear|lawn(\s+collection)?|eastern\s+wear|western\s+wear)(\s+.*)?$|"
        r"^(pakistani\s+)?(women\'?s?\s+)?(clothing|fashion|apparel|dresses)\s+(store|brand|shop|boutique|collection)(\s+.*)?$|"
        r"^(ready[\s-]+to[\s-]+wear|pret|unstitched)(\s+(pret|luxury|lawn))?\s+(collection|dresses|suits|wear)(\s+.*)?$|"
        r"^(pakistani\s+)?(women|men|ladies|girls|boys|kids)(\s+(fashion|clothing|apparel|wear))?$|"
        r"^(clothing|dresses|apparel|suits|fabrics|textiles)(\s+.*)?$",
        key,
    ):
        return True
    if _looks_like_content_or_cpg_noise(raw):
        return True
    if re.search(r"\b(definition\s*(&|and)?\s*meaning|dictionary|thesaurus|synonyms?|antonyms?|etymology|pronunciation)\b", key):
        return True
    if re.match(r"^(top|best|leading|\d+)\s+.*(in|near|of|for)\s+.*", key):
        return True
    if re.match(r"^(top|best|leading|\d+)\s+(lawyers?|doctors?|restaurants?|companies|agencies|places|firms?)\b", key):
        return True
    if key in _GENERIC_RIVAL_NAMES or compact in {re.sub(r"[^a-z0-9]+", "", n) for n in _GENERIC_RIVAL_NAMES}:
        return True

    if _GENERIC_NAME_RE.match(key):
        return True
    if _FAKE_FOOD_NUMBER_NAME_RE.match(key):
        return True
    if _FAKE_FOOD_WORD_NAME_RE.match(key):
        return True
    if re.fullmatch(r"(pizza|burger|cafe|shawarma)\d{1,4}", compact):
        return True
    if compact in {"fourtwentyfour", "420pizza", "hotandspicy"}:
        return True
    if len(compact) < 2:
        return True
    if len(compact) in (2, 3, 4) and compact in {
        "tech", "soft", "corp", "demo", "test", "null", "none", "fake", "temp", "abcd", "xyz", "qwe", "asdf"
    }:
        return True
    if re.match(r"^(tech|soft|pak|it|info|digital|global|smart|web)\s+solutions$", key):
        return True
    if re.search(
        r"\b(software|it|digital|web)\s+(development|developers?|services|solutions|company|house|agency|companies)\b",
        key,
    ) and re.search(
        r"\b(india|pakistan|uae|usa|uk|singapore|bangladesh|remote|offshore|global|saudi|riyadh|dubai)\b",
        key,
    ):
        return True
    if re.match(
        r"^(software|it|digital)\s+(development|developers?|services|solutions)\s+(company|companies|firm|house)(\s+\w+)?$",
        key,
    ):
        return True
    # Listicle / directory / index titles used as "company" names
    if re.search(
        r"\b(companies|brands|manufacturers|stores|shops|suppliers|exporters|vendors|retailers|businesses|agencies|firms|dealers)\s+(list|directory|index|catalog|catalogue|guide|database)\b|"
        r"\b(list|directory|catalog|catalogue|guide|database|index)\s+(of|for|from)\b|"
        r"\b(companies|brands|manufacturers|stores|shops|dealers|agencies|firms)\s+(from|in|near|across)\s+\w+|"
        r"\b(top|best|leading|\d+)\s+.*(companies|brands|manufacturers|stores|shops|exporters|suppliers|agencies|firms|outlets)\b|"
        r"\b(top\s*\d+|best\s*\d+|\d+\s*best|\d+\s*top)\b|"
        r"\b(manufacturers?\s+list|companies?\s+list|suppliers?\s+list|brands?\s+list|exporters?\s+list|directory\s+list)\b",
        key,
    ):
        return True
    if re.search(r"\b(association|council|chamber|federation|bureau|department|ministry)\b", key):
        return True
    if re.search(r"\b(university\s+rankings?|world\s+university\s+rankings?|best\s+universities|top\s+universities|school\s+rankings?)\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+universities\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+schools\b", key):
        return True
    if re.search(r"\b(private|public|recognised|recognized|affiliated|top|best|leading|all)\s+colleges\b", key):
        return True
    if re.search(r"\buniversities\s+(in|of|ranking|rankings|recognised|recognized)\b", key):
        return True
    if re.search(r"\b(higher\s+education\s+commission|hec|higher\s+education\s+department)\b", key):
        return True
    if re.search(r"\b(imarc\s*group|mordor\s*intelligence|grand\s*view\s*research|allied\s*market\s*research|fortune\s*business|market\s*research|business\s+insights)\b", key):
        return True
    if re.fullmatch(
        r"^(data\s+analytics|digital\s+engineering|business\s+intelligence|artificial\s+intelligence|machine\s+learning|software\s+development|web\s+development|cloud\s+services|it\s+services|data\s+science|data\s+analysis)(\s+(services|solutions|tools|consulting|dashboards?|reporting|practice|group|team))?$",
        key,
    ):
        return True
    if re.search(
        r"\b(digital\s+engineering\s+and\s+operational\s+technology\s+services|business\s+intelligence\s+reporting\s+tools|bi\s+tools\s+download|marketing\s+analytics\s+dashboard\s+setup|certified\s+data\s+analyst|data\s+analytics\s+mastery|big\s+data\s+analytics)\b",
        key,
    ):
        return True
    if re.search(r"\b(consulting|services|developers|freelancers|training|course|jobs|hiring)\s+in\s+", key):
        return True
    if re.search(r"\b(freelancers?|upwork|fiverr|toptal|freelancer|guru\.com|ibisworld|statista|capterra|g2\s+crowd|trustpilot)\b", key):
        return True
    if re.match(r"^(about|contact|login|signup|home|privacy|terms|overview|history\s+of|admissions\s+at)\s+", key):
        return True
    return False


def _looks_like_invented_food_domain(name: str, website: str | None) -> bool:
    """Catch AI pairing 'Pizza 24' with pizza24.pk — not a real peer proof."""
    host = _domain_of(website or "")
    if not host:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", _as_str(name).lower())
    if not compact:
        return False
    host_core = re.sub(r"[^a-z0-9]+", "", host.split(".")[0].lower())
    if not host_core:
        return False
    if _FAKE_FOOD_NUMBER_NAME_RE.match(re.sub(r"\s+", " ", _as_str(name).lower()).strip()):
        return True
    if re.fullmatch(r"(pizza|burger|cafe|shawarma)\d{1,4}", compact) and host_core == compact:
        return True
    return False


_PARKED_SITE_MARKERS = (
    "domain for sale", "buy this domain", "this domain is for sale",
    "parked domain", "parkingcrew", "sedoparking", "godaddy parking",
    "coming soon", "under construction", "website coming soon",
    "account suspended", "default web page", "apache2 ubuntu default",
)
_SOFTWARE_PEER_SITE_MARKERS = (
    "software", "development", "digital agency", "web development", "mobile app",
    "it services", "custom software", "product engineering", "outsourcing",
    "app development", "devops", "saas", "solutions for",
)


def _site_looks_parked_or_empty(site_md: str) -> bool:
    text = _as_str(site_md).lower().strip()
    if len(text) < 80:
        return True
    return any(m in text for m in _PARKED_SITE_MARKERS)


def _site_supports_software_peer(site_md: str) -> bool:
    text = _as_str(site_md).lower()
    if not text:
        return False
    return sum(1 for m in _SOFTWARE_PEER_SITE_MARKERS if m in text) >= 2


# ---------------------------------------------------------------------------
# Global Blocklists
# ---------------------------------------------------------------------------

_GLOBAL_RIVAL_BLOCKLIST = {
    "accenture", "ibm", "ibm watson", "watson", "microsoft", "microsoft ai", "microsoft azure",
    "azure", "google", "google cloud", "google cloud ai", "google ai", "dialogflow", "amazon",
    "aws", "amazon web services", "oracle", "oracle ai", "sap", "sap leonardo", "deloitte",
    "pwc", "ey", "ernst & young", "kpmg", "cognizant", "infosys", "capgemini", "tcs",
    "tata consultancy", "wipro", "meta", "openai", "anthropic", "salesforce", "adobe",
    "nvidia", "mckinsey", "bain", "bcg", "boston consulting", "slalom",
    "manychat", "converse.ai", "inbenta",
    "xbox", "playstation", "nintendo", "sony",
    "general motors", "gm", "siemens", "philips", "canon", "paypal", "palantir", "hyperkin",
}

_GLOBAL_DOMAIN_BLOCKLIST = {
    "accenture.com", "ibm.com", "microsoft.com", "azure.microsoft.com", "google.com",
    "cloud.google.com", "dialogflow.cloud.google.com", "amazon.com", "aws.amazon.com",
    "oracle.com", "sap.com", "deloitte.com", "pwc.com", "ey.com", "kpmg.com",
    "cognizant.com", "infosys.com", "capgemini.com", "tcs.com", "wipro.com",
    "openai.com", "anthropic.com", "salesforce.com", "adobe.com", "nvidia.com",
    "mckinsey.com", "bain.com", "bcg.com", "slalom.com",
    "manychat.com", "converse.ai", "inbenta.com",
    "xbox.com", "playstation.com", "nintendo.com",
    "gm.com", "siemens.com", "philips.com", "canon.com", "paypal.com", "palantir.com",
    "hyperkin.com", "hyperkinstore.com",
    "shawarmajunction.com",
    "shawarma-house.com",
}


def _is_global_megarival(name: str, website: str | None = None) -> bool:
    n = _as_str(name).strip().lower()
    if not n:
        return False
    if n in _GLOBAL_RIVAL_BLOCKLIST:
        return True
    for blocked in _GLOBAL_RIVAL_BLOCKLIST:
        if len(blocked) < 4:
            continue
        if blocked == n or n.startswith(blocked + " ") or n.endswith(" " + blocked) or f" {blocked} " in f" {n} ":
            return True
        if blocked in n and blocked not in {"ai", "aws", "ibm", "sap", "ey", "tcs", "bcg"}:
            return True
    host = _domain_of(website or "")
    if host:
        for blocked in _GLOBAL_DOMAIN_BLOCKLIST:
            if host == blocked or host.endswith("." + blocked):
                return True
    return False


# ---------------------------------------------------------------------------
# Blog, Directory, and Content Noise Filtering
# ---------------------------------------------------------------------------

_CONTENT_OR_CPG_HOST_MARKERS = (
    "travelblog", "foodblog", "blog", "magazine", "recipes", "photos",
    "grocery", "review", "directory", "news", "press", "media", "portal", "wiki",
)
_CPG_PATH_MARKERS = ("/product/", "/products/", "/shop/", "/sku/")
_DIRECTORY_OR_LISTICLE_PATH_MARKERS = (
    "/companies/", "/company/", "/agencies/", "/agency/", "/top-", "/best-",
    "/10-top-", "/software-companies", "/it-companies", "/list-of-", "/directory/",
    "/blog/", "/blogs/", "/post/", "/posts/", "/article/", "/articles/", "/news/",
    "/story/", "/stories/", "/review/", "/reviews/", "/vs/", "/compare/",
    "/comparison/", "/list/", "/lists/", "/guides/", "/guide/", "/insights/",
    "/trends/", "/category/", "/tag/", "/author/", "/feed/", "/press-release/",
    "/how-to-", "/tutorials/", "/case-studies/", "/ranking/", "/rankings/",
    "/market-report/",
)
_CITY_IN_TITLE_RE = re.compile(
    r"\b(in|near)\s+(lahore|karachi|islamabad|rawalpindi|peshawar|multan|faisalabad|pakistan|dubai|riyadh|london|uk|usa)\b",
    re.I,
)
_BLOG_OR_ARTICLE_TITLE_PATTERNS = re.compile(
    r"(\btop\s+\d+\b|\bbest\s+\d+\b|\b\d+\s+(top|best|leading)\b|"
    r"\b(top|best|leading|\d+)\s+(pakistani\s+)?(universities|schools|colleges|companies|places|restaurants|cafes|software|softwares|tools|agencies)\b|"
    r"\b(included\s+in|featured?\s+in|rankings?|review|reviews|guide|how\s+to|complete\s+guide|versus|comparison)\b)",
    re.I,
)


def _is_blog_or_article_url(url: str, title: str = "") -> bool:
    """True if URL or title represents a blog post, article, news story, or listicle directory."""
    if not url:
        return False
    if _is_serp_noise_domain(url):
        return True
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
        if any(p in path for p in _DIRECTORY_OR_LISTICLE_PATH_MARKERS):
            return True
        if re.search(r"/\d{4}/\d{2}/", path) or re.search(r"/(how-to|best|top|guide)-", path):
            return True
        if path.count("-") >= 3 or re.search(r"/\d+-[a-z0-9-]+", path):
            return True
        if any(k in host for k in ("regionalstudies", "pakistantoday", "dailytimes", "tribune", "dawn", "brecorder", "thenews", "dailypakistan", "propakistani")):
            return True
    except Exception:
        pass
    if title and _BLOG_OR_ARTICLE_TITLE_PATTERNS.search(title.strip()):
        return True
    return False


def _looks_like_content_or_cpg_noise(name: str, website: str | None = None) -> bool:
    raw = _as_str(name).strip()
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    if not key:
        return True
    if _BLOG_OR_ARTICLE_TITLE_PATTERNS.search(raw):
        return True
    if re.search(r"\bphotos?\b|\bgallery\b|\bimages?\b", key):
        return True
    if _CITY_IN_TITLE_RE.search(key):
        return True
    if any(tok in key for tok in (
        " food blog", "travel blog", "best places", "things to eat", "top 10", "top 5",
        "best software", "complete guide", "alternatives to", "vs ", " versus "
    )):
        return True

    website = _as_str(website).strip().lower()
    if not website:
        return False
    if _is_blog_or_article_url(website, name):
        return True
    host = _domain_of(website)
    if any(m in host for m in _CONTENT_OR_CPG_HOST_MARKERS):
        return True
    path = urlparse(website if "://" in website else f"https://{website}").path.lower()
    if any(m in path for m in _CPG_PATH_MARKERS):
        return True
    return False


def _looks_like_marketing_slogan_name(name: str) -> bool:
    key = re.sub(r"\s+", " ", _as_str(name).lower()).strip()
    if not key or len(key) < 18:
        return False
    if re.search(r"\b(order it today|order now|baked to perfection|made fresh[,—-])\b", key):
        return True
    if key.count(" ") >= 8 and re.search(r"\b(from classic to|has it all|check .+ menu)\b", key):
        return True
    return False


def _brand_guess_from_host(website: str | None) -> str:
    """creative-sols.com → Creative Sols; skip generic hosts."""
    host = _domain_of(website or "")
    if not host:
        return ""
    labels = [p for p in host.split(".") if p and p not in {"com", "net", "org", "gov", "edu", "pk", "co", "uk", "ae", "sa", "in", "us"}]
    core = ""
    for p in labels:
        if p.lower() not in {"www", "m", "pk", "en", "us", "uk", "ae", "shop", "store", "app", "api", "cdn"}:
            core = p
            break
    if not core and labels:
        core = labels[0]
    skip = {
        "www", "app", "apps", "blog", "shop", "store", "orders", "order", "menu",
        "india", "pakistan", "saudi", "uae", "software", "company", "services", "pk",
    }
    if not core or core in skip or len(core) < 3:
        return ""
    core_clean = re.sub(r"^(hey|get|try|my|the|go)[\-_]?", "", core, flags=re.I)
    core_clean = re.sub(r"(shop|store|official|online|pk|clothing|apparel)$", "", core_clean, flags=re.I)
    core_use = core_clean if len(core_clean) >= 3 else core
    parts = re.split(r"[\-_]+", core_use)
    nice = " ".join(p.capitalize() for p in parts if p)
    return nice.strip()


def _clean_brand_from_title(title: str, link: str) -> str:
    """Extract clean company/brand name from page title or domain."""
    host_guess = _brand_guess_from_host(link)
    if not title:
        return host_guess
    title_clean = re.sub(
        r"^(about\s*(us)?|contact\s*(us)?|home(page)?|our\s*story|who\s*we\s*are|menu|our\s*menu|locations|outlets|branches|careers|jobs|reviews|customer\s*reviews|faq|faqs|login|sign\s*in|order\s*online)\s*[-|–—:]\s*",
        "",
        title,
        flags=re.I,
    ).strip()
    title_clean = re.sub(
        r"\s*[-|–—:]\s*(about\s*(us)?|contact\s*(us)?|home(page)?|our\s*story|who\s*we\s*are|menu|our\s*menu|locations|outlets|branches|careers|jobs|reviews|customer\s*reviews|faq|faqs|login|sign\s*in|order\s*online)$",
        "",
        title_clean,
        flags=re.I,
    ).strip()
    parts = [p.strip() for p in re.split(r"\s+[|–—]\s+|\s+-\s+|:\s+", title_clean) if p.strip()]
    if not parts:
        return host_guess

    host_core = re.sub(r"[^a-z0-9]+", "", host_guess.lower())
    host_raw = re.sub(r"[^a-z0-9]+", "", _domain_of(link).split(".")[0].lower())

    # 1. Prefer a title part that shares tokens with the domain host
    for p in parts:
        cleaned = _clean_rival_display_name(p)
        if not cleaned or _is_generic_or_fake_rival_name(cleaned):
            continue
        compact_p = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
        if not compact_p:
            continue
        if compact_p == host_core or compact_p == host_raw:
            return cleaned
        if host_core and len(host_core) >= 3 and (host_core in compact_p or compact_p in host_core):
            if len(cleaned.split()) > 3 and host_guess and len(host_guess) >= 3:
                return host_guess
            return cleaned
        if host_raw and len(host_raw) >= 3 and (host_raw in compact_p or compact_p in host_raw):
            if len(cleaned.split()) > 3 and host_guess and len(host_guess) >= 3:
                return host_guess
            return cleaned

    # 2. Pick the first non-generic, non-slogan part
    for p in parts:
        cleaned = _clean_rival_display_name(p)
        if (
            cleaned
            and len(cleaned) <= 45
            and not _is_generic_or_fake_rival_name(cleaned)
            and not re.match(r"^(order|get\s+the|find|buy|shop|best|welcome|we\s+are|the\s+\d+|experience|savor|discover|authentic|taste|delicious|enjoy|crafting|serving)\b", cleaned, flags=re.I)
        ):
            return cleaned

    # 3. Fall back to clean domain name
    if host_guess and not _is_generic_or_fake_rival_name(host_guess):
        return host_guess
    first_clean = _clean_rival_display_name(parts[0])
    if first_clean and not _is_generic_or_fake_rival_name(first_clean):
        return first_clean
    return host_guess or first_clean

__all__ = [
    "_is_serp_noise_domain",
    "_clip",
    "_level_label",
    "_WEAK_COMPARISON_MARKERS",
    "_as_str",
    "_as_int",
    "_as_list",
    "_is_generic_text",
    "_domain_of",
    "_normalize_website",
    "_GENERIC_RIVAL_NAME_TAILS",
    "_rival_name_key",
    "_rival_host_key",
    "_rival_keys",
    "_is_self_rival",
    "_find_matching_competitor",
    "_RECIPE_OR_DISH_TITLE_MARKERS",
    "_RECIPE_STYLE_TOKENS",
    "_BARE_DISH_NAME_RE",
    "_clean_rival_display_name",
    "_looks_like_recipe_or_menu_item_name",
    "_GENERIC_RIVAL_NAMES",
    "_GENERIC_NAME_RE",
    "_FAKE_FOOD_NUMBER_NAME_RE",
    "_FAKE_FOOD_WORD_NAME_RE",
    "_is_generic_or_fake_rival_name",
    "_looks_like_invented_food_domain",
    "_PARKED_SITE_MARKERS",
    "_SOFTWARE_PEER_SITE_MARKERS",
    "_site_looks_parked_or_empty",
    "_site_supports_software_peer",
    "_GLOBAL_RIVAL_BLOCKLIST",
    "_GLOBAL_DOMAIN_BLOCKLIST",
    "_is_global_megarival",
    "_CONTENT_OR_CPG_HOST_MARKERS",
    "_CPG_PATH_MARKERS",
    "_DIRECTORY_OR_LISTICLE_PATH_MARKERS",
    "_CITY_IN_TITLE_RE",
    "_BLOG_OR_ARTICLE_TITLE_PATTERNS",
    "_is_blog_or_article_url",
    "_looks_like_content_or_cpg_noise",
    "_looks_like_marketing_slogan_name",
    "_brand_guess_from_host",
    "_clean_brand_from_title"
]
