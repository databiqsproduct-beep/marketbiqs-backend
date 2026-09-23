"""Geographic mappings, country aliases, TLDs, and market helpers."""

from __future__ import annotations

import re

COUNTRY_ALIASES: dict[str, set[str]] = {
    "pakistan": {
        "pakistan", "pakistani", "pk", "pak",
        "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad",
        "multan", "peshawar", "sialkot", "gujranwala", "quetta",
    },
    "india": {
        "india", "indian", "bharat",
        "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad",
        "chennai", "pune", "noida", "gurgaon", "gurugram", "kolkata", "ahmedabad",
    },
    "singapore": {"singapore", "singaporean", "sg"},
    "uae": {
        "uae", "united arab emirates", "dubai", "abu dhabi", "abudhabi", "sharjah",
        "emirates",
    },
    "saudi arabia": {"saudi", "saudi arabia", "ksa", "riyadh", "jeddah", "dammam"},
    "united states": {
        "united states", "usa", "u.s.", "u.s.a", "america", "american",
        "california", "new york", "texas", "silicon valley",
    },
    "united kingdom": {"united kingdom", "uk", "u.k.", "britain", "british", "london", "england"},
    "canada": {"canada", "canadian", "toronto", "vancouver", "montreal", "ontario", "mississauga", "british columbia"},
    "australia": {"australia", "australian", "sydney", "melbourne"},
    "germany": {"germany", "german", "berlin", "munich"},
    "france": {
        "france", "french", "lyon", "marseille", "bordeaux",
    },
    "italy": {"italy", "italian", "rome", "milan", "milano", "florence"},
    "bangladesh": {"bangladesh", "bangladeshi", "dhaka", "chittagong", "sylhet", "rajshahi", "khulna"},
    "china": {"china", "chinese", "beijing", "shanghai", "shenzhen"},
}

COUNTRY_TLDS: dict[str, set[str]] = {
    "pakistan": {".pk"},
    "india": {".in"},
    "singapore": {".sg"},
    "uae": {".ae"},
    "saudi arabia": {".sa"},
    "united kingdom": {".uk", ".co.uk"},
    "germany": {".de"},
    "australia": {".au", ".com.au"},
    "canada": {".ca"},
    "bangladesh": {".bd"},
    "china": {".cn"},
}

COUNTRY_SERP_GL: dict[str, str] = {
    "pakistan": "pk",
    "india": "in",
    "singapore": "sg",
    "uae": "ae",
    "saudi arabia": "sa",
    "united states": "us",
    "united kingdom": "uk",
    "canada": "ca",
    "australia": "au",
    "germany": "de",
    "bangladesh": "bd",
}

BRAND_PLACE_TO_COUNTRY: dict[str, str] = {
    "paris": "france",
    "french": "france",
    "france": "france",
    "london": "united kingdom",
    "britain": "united kingdom",
    "british": "united kingdom",
    "rome": "italy",
    "italian": "italy",
    "italy": "italy",
    "tokyo": "japan",
    "japan": "japan",
    "japanese": "japan",
    "new york": "united states",
    "nyc": "united states",
    "america": "united states",
    "dubai": "uae",
    "istanbul": "turkey",
    "turkish": "turkey",
}


def normalize_country_key(val: str | None) -> str:
    """Normalize freeform country name or code into canonical key."""
    raw = (val or "").strip().lower()
    if not raw:
        return ""
    for canonical, aliases in COUNTRY_ALIASES.items():
        if raw == canonical or raw in aliases:
            return canonical
        if any(re.search(r"\b" + re.escape(alias) + r"\b", raw) for alias in aliases):
            return canonical
    return raw
