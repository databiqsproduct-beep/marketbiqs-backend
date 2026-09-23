"""Geographic scrutiny, market scoping, and locality gating for competitive analysis."""

import re
from typing import Any

from app.core.geo_constants import (
    BRAND_PLACE_TO_COUNTRY,
    COUNTRY_ALIASES,
    COUNTRY_SERP_GL,
    COUNTRY_TLDS,
    normalize_country_key,
)
from app.core.serp_noise import is_serp_noise_domain
from app.models import ClientBrand
from app.services.competitive_identities import (
    _as_str,
    _domain_of,
    _is_generic_or_fake_rival_name,
    _is_global_megarival,
    _is_self_rival,
)

_COUNTRY_ALIASES = COUNTRY_ALIASES
_COUNTRY_TLDS = COUNTRY_TLDS
_COUNTRY_SERP_GL = COUNTRY_SERP_GL
_BRAND_PLACE_TO_COUNTRY = BRAND_PLACE_TO_COUNTRY
_normalize_country_key = normalize_country_key

# Home market for well-known brands when AI invents expansion markets (e.g. Cheezious → Riyadh)
_KNOWN_BRAND_HOME_MARKET: dict[str, str] = {
    "cheezious": "Pakistan",
    "howdy": "Pakistan",
    "optp": "Pakistan",
    "ranchers": "Pakistan",
    "johnny & jugnu": "Pakistan",
    "johnny and jugnu": "Pakistan",
    "sultan shawarma": "Pakistan",
    "meet me in paris": "Pakistan",
    "fine pizza": "Pakistan",
    "broadway pizza": "Pakistan",
    "pizza max": "Pakistan",
    "california pizza": "Pakistan",
    "systems limited": "Pakistan",
    "systems ltd": "Pakistan",
    "netsol": "Pakistan",
    "netsol technologies": "Pakistan",
}


def _known_brand_home_market(name: str, website: str | None = None) -> str:
    """Canonical home market for known brands — beats AI expansion-market guesses."""
    n = _as_str(name).lower().strip()
    if not n:
        return ""
    # Prefer longer keys first (systems limited before bare systems)
    for brand, market in sorted(_KNOWN_BRAND_HOME_MARKET.items(), key=lambda kv: -len(kv[0])):
        if brand == n or brand in n or n == brand:
            return market
    # Bare "systems" is ambiguous — only map when website looks like Systems Ltd PK
    host = _domain_of(website or "")
    if n in {"systems", "system"} and (
        "systemsltd" in host or host.endswith(".pk") or host.endswith(".com.pk")
    ):
        return "Pakistan"
    if host.endswith(".pk") or host.endswith(".com.pk"):
        return "Pakistan"
    return ""


def _serp_location_for_market(market: str) -> str | None:
    """SerpAPI `location` prefers a clean country/city string, not a free-form AI phrase."""
    key = _normalize_country_key(market)
    if not key:
        raw = _as_str(market).strip()
        return raw or None
    # Prefer the canonical country name so gl + location stay aligned
    pretty = {
        "pakistan": "Pakistan",
        "india": "India",
        "singapore": "Singapore",
        "uae": "United Arab Emirates",
        "saudi arabia": "Saudi Arabia",
        "united states": "United States",
        "united kingdom": "United Kingdom",
        "canada": "Canada",
        "australia": "Australia",
        "germany": "Germany",
        "bangladesh": "Bangladesh",
    }
    return pretty.get(key, key.title())


def _market_aliases(market: str) -> set[str]:
    key = _normalize_country_key(market)
    aliases = set(_COUNTRY_ALIASES.get(key, set()))
    raw = _as_str(market).lower().strip()
    if raw:
        aliases.add(raw)
        for tok in re.split(r"[^a-z0-9]+", raw):
            if len(tok) >= 3:
                aliases.add(tok)
    if key:
        aliases.add(key)
    return {a for a in aliases if a}


def _blob_mentions_any(blob: str, terms: set[str]) -> bool:
    if not blob or not terms:
        return False
    # Word-boundary-ish: prefer whole-word for short tokens (pk, sg, in, uk, ae)
    for term in terms:
        if len(term) <= 2:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", blob):
                return True
        elif term in blob:
            return True
    return False


def _host_matches_tlds(host: str, tlds: set[str]) -> bool:
    if not host:
        return False
    for tld in tlds:
        suffix = tld[1:] if tld.startswith(".") else tld
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _mentions_target_market(blob: str, website: str | None, market: str) -> bool:
    aliases = _market_aliases(market)
    if _blob_mentions_any(blob, aliases):
        return True
    key = _normalize_country_key(market)
    host = _domain_of(website or "")
    if key and _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
        return True
    return False


def _mentions_conflicting_country(
    blob: str,
    website: str | None,
    market: str,
    client_name: str = "",
) -> bool:
    """True when text/site clearly points at a different known country than the required market."""
    target = _normalize_country_key(market)
    if not target:
        return False
    text = _as_str(blob).lower()
    if client_name:
        # Strip client name and its tokens so phrases like "competitor of Comsats Lahore" do not inject Pakistan into an India check
        c_clean = _as_str(client_name).lower().strip()
        if c_clean:
            text = text.replace(c_clean, " ")
            for tok in re.split(r"[^a-z0-9]+", c_clean):
                if len(tok) >= 3:
                    text = re.sub(rf"\b{re.escape(tok)}\b", " ", text)
    host = _domain_of(website or "")
    found: set[str] = set()
    for key, aliases in _COUNTRY_ALIASES.items():
        if key == target:
            continue
        if _blob_mentions_any(text, aliases):
            found.add(key)
        if _host_matches_tlds(host, _COUNTRY_TLDS.get(key, set())):
            found.add(key)
    return bool(found)


# AI often invents these when a food brand name contains a foreign city (e.g. Paris).
_FOREIGN_CAFE_HALLUCINATION_RE = re.compile(
    r"\b("
    r"le petit\b|le french\b|la maison\b|les petits?\b|"
    r"paris\s+(cafe|bakery|bistro|restaurant)|"
    r"cafe\s+paris\b|"
    r"french\s+(cafe|bakery|bistro|restaurant)|"
    r"petit paris|maison bakery|maison cafe"
    r")\b",
    re.I,
)


def _ascii_fold(text: str) -> str:
    raw = _as_str(text).lower()
    return (
        raw.replace("café", "cafe")
        .replace("café", "cafe")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("á", "a")
        .replace("à", "a")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ï", "i")
    )


def _foreign_places_echoed_from_brand(client_name: str, market: str) -> set[str]:
    """Foreign place tokens present in the client brand that are outside the selected market."""
    market_key = _normalize_country_key(market)
    name_l = _ascii_fold(client_name)
    echoed: set[str] = set()
    for token, country in _BRAND_PLACE_TO_COUNTRY.items():
        if token in name_l and country != market_key:
            echoed.add(token)
    return echoed


def _looks_like_brand_geo_hallucination(
    client_name: str,
    rival_name: str,
    market: str,
    *,
    website: str | None = None,
    source: str = "",
) -> bool:
    """
    Reject AI-invented foreign-theme rivals triggered by place words in the client brand
    (e.g. Meet Me in Paris → fake Paris/French cafés when market is Pakistan).
    """
    market_key = _normalize_country_key(market)
    rival_l = _ascii_fold(rival_name).strip()
    if not rival_l:
        return True
    source_l = _as_str(source).lower()
    host = _domain_of(website or "")
    local_host = bool(host) and (
        (
            bool(market_key)
            and _host_matches_tlds(host, _COUNTRY_TLDS.get(market_key, set()))
        )
        or (
            market_key == "pakistan"
            and any(a in host for a in ("pakistan", "lahore", "karachi"))
        )
        or host.endswith(".pk")
    )

    # Real Lahore brand that shares "Paris" with the client name — keep only with local proof
    if "cafe de paris" in rival_l:
        return not (source_l == "serp" or local_host)

    # Classic French-café hallucination templates — always reject (local and global runs)
    if _FOREIGN_CAFE_HALLUCINATION_RE.search(rival_l):
        return True

    if not market_key or market_key == "france":
        return False

    echoed = _foreign_places_echoed_from_brand(client_name, market)
    if not echoed:
        return False
    # Rival name reuses the foreign place token from the brand (paris/french/…) without local proof
    if any(tok in rival_l for tok in echoed):
        if local_host and source_l == "serp":
            return False
        if source_l == "serp" and local_host:
            return False
        if source_l != "serp":
            return True
        if not local_host:
            return True
    return False


def _rival_fits_run_scope(
    *,
    name: str,
    website: str | None,
    headquarters: str | None = None,
    description: str | None = None,
    why: str | None = None,
    scope: str,
    market: str,
    city: str | None = None,
    client_name: str,
    is_pinned: bool = False,
    strict: bool = True,
) -> bool:
    """Whether a rival should stay tracked for this intel run's local/global filter."""
    if _is_generic_or_fake_rival_name(name):
        return False
    if _is_self_rival(client_name, name, website=website):
        return False
    if is_pinned:
        return True

    scope_l = "global" if str(scope).lower() == "global" else "local"

    if scope_l == "global":
        if _is_global_megarival(name, website):
            return False
        if website and is_serp_noise_domain(website):
            return False
        return True

    # Local scope
    mkt = _as_str(market).strip().lower()
    hq = _as_str(headquarters).strip().lower()
    desc = f"{description or ''} {why or ''}".lower()
    desc_for_geo = desc
    if client_name:
        c_name_clean = re.sub(r"[^a-zA-Z0-9]+", " ", client_name).strip().lower()
        if c_name_clean and len(c_name_clean) >= 3:
            desc_for_geo = re.sub(rf"[^.\n]*\b{re.escape(c_name_clean)}\b[^.\n]*", "", desc_for_geo)
    host = _domain_of(website or "")
    name_l = _as_str(name).lower()
    full_blob = f"{name_l} {desc_for_geo} {host} {hq}"

    mkt_key = _normalize_country_key(mkt)
    hq_key = _normalize_country_key(hq)

    # 1. If explicit headquarters is specified and known, enforce country match
    if hq_key and hq_key not in {"unknown", "not disclosed", "not disclosed on the website", "none", "n/a", "worldwide", "global"}:
        if mkt_key and hq_key != mkt_key:
            return False
        if mkt_key and hq_key == mkt_key and not city:
            return True

    # 2. City-level enforcement if city is specified
    c_clean = _as_str(city).strip().lower()
    if c_clean:
        city_tokens = [t for t in re.split(r"[^a-z0-9]+", c_clean) if len(t) >= 3]
        has_city_match = any(t in full_blob for t in city_tokens) or (c_clean in full_blob)
        
        # Check for conflicting other cities in the same market
        market_cities_map = {
            "pakistan": {"karachi", "islamabad", "rawalpindi", "peshawar", "multan", "faisalabad", "sialkot", "gujranwala", "quetta", "lahore"},
            "united arab emirates": {"dubai", "abu dhabi", "sharjah", "ajman", "al ain"},
            "united kingdom": {"london", "manchester", "birmingham", "leeds", "glasgow", "liverpool", "edinburgh"},
            "united states": {"new york", "los angeles", "chicago", "houston", "austin", "dallas", "miami", "seattle", "boston", "san francisco"},
            "saudi arabia": {"riyadh", "jeddah", "dammam", "khobar", "mecca", "medina"},
        }
        known_cities = market_cities_map.get(mkt_key, set())
        conflicting_cities = {ct for ct in known_cities if ct != c_clean and ct not in city_tokens}
        has_conflicting_city = any(re.search(rf"\b{re.escape(ct)}\b", full_blob) for ct in conflicting_cities)

        if has_conflicting_city and not has_city_match:
            # Candidate is explicitly located in a different city in that country
            return False

        if has_city_match:
            return True

        if strict and not has_city_match:
            # If strict local matching is required and city is unknown/unconfirmed, reject
            return False

    # 3. Check local TLD match
    tlds = _COUNTRY_TLDS.get(mkt_key, set()) if mkt_key else set()
    if host and (_host_matches_tlds(host, tlds) or (mkt_key == "pakistan" and host.endswith(".pk"))):
        return True

    # 4. Check for local presence / country tokens in snippet, why, description, or host
    local_tokens = [mkt] if mkt else []
    if mkt_key == "pakistan":
        local_tokens.extend(["lahore", "karachi", "islamabad", "rawalpindi", "faisalabad", "peshawar", "multan", "pakistan"])
    elif mkt_key == "united arab emirates":
        local_tokens.extend(["dubai", "abu dhabi", "sharjah", "uae"])
    elif mkt_key == "united kingdom":
        local_tokens.extend(["london", "manchester", "birmingham", "uk"])
    elif mkt_key == "saudi arabia":
        local_tokens.extend(["riyadh", "jeddah", "dammam", "saudi"])

    if any(tok in desc for tok in local_tokens if tok) or any(tok in host for tok in local_tokens if tok):
        return True

    # If strict is requested and no local proof found, reject for local scope
    if strict:
        return False
    return True


def _serp_gl_for_market(market: str) -> str | None:
    key = _normalize_country_key(market)
    return _COUNTRY_SERP_GL.get(key)


def _market_area_from_client(client: ClientBrand) -> str:
    notes = _as_str(getattr(client, "notes", ""))
    for line in notes.splitlines():
        if line.lower().startswith("market:"):
            return line.split(":", 1)[1].strip()
    return ""


def _city_from_client(client: ClientBrand) -> str:
    """Extract local target city from client attributes or profile notes."""
    c_attr = getattr(client, "city", None)
    if c_attr and _as_str(c_attr).strip():
        return _as_str(c_attr).strip()
    notes = _as_str(getattr(client, "notes", ""))
    for line in notes.splitlines():
        line_l = line.lower().strip()
        if line_l.startswith("city:"):
            return line.split(":", 1)[1].strip()
    mkt = _market_area_from_client(client)
    if "," in mkt:
        candidate_city = mkt.split(",")[0].strip()
        if len(candidate_city) >= 3:
            return candidate_city
    return ""


def _extract_metadata_from_notes(notes: str | None) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not notes:
        return meta
    for ln in str(notes).splitlines():
        trimmed = ln.strip()
        low = trimmed.lower()
        if low.startswith("market:") or low.startswith("country:"):
            meta["country"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("city:") or low.startswith("region:"):
            meta["city"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("industry:"):
            meta["industry"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("niche:"):
            meta["niche"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("customer type:") or low.startswith("target customer:"):
            meta["customer_type"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("primary offering:") or low.startswith("offering:"):
            meta["primary_offering"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("business model:"):
            meta["business_model"] = trimmed.split(":", 1)[1].strip()
        elif low.startswith("industry category:"):
            meta["industry_category"] = trimmed.split(":", 1)[1].strip().lower()
    return meta

__all__ = [
    "_COUNTRY_ALIASES",
    "_COUNTRY_TLDS",
    "_COUNTRY_SERP_GL",
    "_BRAND_PLACE_TO_COUNTRY",
    "_normalize_country_key",
    "_known_brand_home_market",
    "_serp_location_for_market",
    "_market_aliases",
    "_blob_mentions_any",
    "_host_matches_tlds",
    "_mentions_target_market",
    "_mentions_conflicting_country",
    "_FOREIGN_CAFE_HALLUCINATION_RE",
    "_ascii_fold",
    "_foreign_places_echoed_from_brand",
    "_looks_like_brand_geo_hallucination",
    "_rival_fits_run_scope",
    "_serp_gl_for_market",
    "_market_area_from_client",
    "_city_from_client",
    "_extract_metadata_from_notes"
]
