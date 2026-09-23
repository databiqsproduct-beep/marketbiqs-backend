"""Industry classification, format/tier compatibility, and vertical isolation for competitive analysis."""

from __future__ import annotations

import re
from typing import Any

from app.models import ClientBrand
from app.services.competitive_identities import (
    _as_str,
    _domain_of,
    _is_global_megarival,
    _rival_name_key,
)

# Peer-fit: reject consumer retail / media / wrong verticals when the client is a B2B software/agency peer
_RETAIL_MARKETPLACE_MARKERS = (
    "ecommerce", "e-commerce", "e commerce", "online shopping", "online store", "online retail",
    "shopping platform", "shopping mall", "marketplace", "cash on delivery", "cash-on-delivery",
    "fashion", "electronics store", "consumer durables", "grocery", "retail store", "retailer",
    "buy online", "add to cart", "shop now", "apparel", "clothing", "footwear", "garments", "textiles",
)

_GOVERNMENT_MARKERS = (
    "government", "govt", "gov.", "ministry", "public sector", "state-owned", "state owned",
    "federal board", "provincial board", "information technology board", "it board",
    "authority", "commission", "regulator", "municipal", "city government",
    "public body", "government of", "gov of", "pitb", "nadra", "fbr", "secp",
    "digital pakistan", "e-government", "egovernment", "smart city authority",
)

# Strong verticals — a rival dominated by one of these is NOT a peer unless the client shares it
_VERTICAL_MARKERS: dict[str, tuple[str, ...]] = {
    "fintech": (
        "fintech", "digital wallet", "mobile wallet", "e-wallet", "ewallet", "payment app",
        "payments", "payment gateway", "money transfer", "remittance", "neobank", "digital bank",
        "banking app", "lendtech", "buy now pay later", "bnpl", "credit card", "debit card",
        "wallet app", "send money", "cash in", "cash out", "iban", "branchless banking",
    ),
    "retail": _RETAIL_MARKETPLACE_MARKERS,
    "automotive": (
        "automotive", "automobile", "cars", "car dealership", "dealerships", "vehicles", "suv",
        "motorcycles", "electric vehicles", "used cars", "auto retail", "car trading",
    ),
    "manufacturing": (
        "manufacturing", "pharmaceutical manufacturing", "process engineer", "digital twin",
        "plant optimization", "factory", "industrial automation",
    ),
    "telecom": ("telecom", "mobile network", "mobile operator", "isp ", "broadband provider", "5g network"),
    "healthcare": (
        "hospital", "hospitals", "clinic", "clinics", "telemedicine", "telehealth", "healthcare",
        "health care", "diagnostic", "pathology", "medical lab", "laboratory", "pharma",
        "pharmacy", "pharmaceutical", "medical device",
    ),
    "higher_education": (
        "university", "universities", "higher ed", "higher education", "degree awarding",
        "undergraduate", "postgraduate", "bachelor", "master's degree", "phd program",
        "institute of technology", "business school", "medical university", "engineering university",
    ),
    "k12_school": (
        "school system", "grammar school", "high school", "primary school", "elementary school",
        "middle school", "k-12", "k12", "preschool", "kindergarten", "school network", "o level school",
    ),
    "college_intermediate": (
        "intermediate college", "junior college", "higher secondary", "fsc college", "ics college", "a level college",
    ),
    "test_prep_academy": (
        "test prep", "entry test", "mdcat", "ecat", "sat prep", "coaching center", "tuition academy",
    ),
    "edtech": ("edtech", "online learning", "e-learning", "school management", "university portal", "tutoring platform"),
    "logistics": ("logistics", "courier", "shipping company", "fleet management", "warehousing", "freight"),
    "real_estate": ("real estate", "property portal", "housing marketplace", "listings platform"),
    "government": _GOVERNMENT_MARKERS,
    "cybersecurity": ("cybersecurity", "endpoint security", "soc ", "threat detection", "penetration testing firm"),
    "data_ai": (
        "competitive intelligence", "market intelligence", "business intelligence", "data analytics",
        "ai agency", "machine learning platform", "data platform", "bi platform", "competitor tracking",
        "market research software", "insights platform", "artificial intelligence", "machine learning",
        "data science", "enterprise ai", "ai solutions", "ai consulting", "analytics consulting",
        "data consulting", "ai-driven",
    ),
    "software_services": (
        "software house", "software development company", "custom software", "it services",
        "digital agency", "web development agency", "product engineering", "dev shop",
        "digital engineering", "outsourcing software", "application development", "software company",
        "it consulting", "technology consulting", "software engineering", "saas",
    ),
    "food_qsr": (
        "fast food", "fast-food", "qsr", "pizza", "burger", "fried chicken", "shawarma",
        "restaurant", "restaurants", "diner", "cafe", "café", "bakery", "ice cream",
        "food chain", "food brand", "quick service", "quick-service", "cloud kitchen",
        "ghost kitchen", "delivery pizza", "pizzas", "burgers", "broast", "biryani",
        "fast casual", "eatery", "eateries",
    ),
    "beauty_personal_care": (
        "beauty salon", "beauty parlour", "beauty parlor", "hair salon", "makeup studio",
        "bridal makeup", "skincare", "skin care", "aesthetic clinic", "cosmetics",
        "dermatology clinic", "hair dressing", "hair stylist", "spa and wellness",
        "beauty lounge", "nail bar", "nail salon", "lash bar", "beauty clinic", "medspa",
    ),
    "fashion_lifestyle": (
        "apparel", "clothing brand", "fashion brand", "boutique", "couture", "pret",
        "fabric", "textile", "designer wear", "footwear", "accessories brand",
    ),
    "talent_marketplace": (
        "talent marketplace", "talent network", "hire developers", "staff augmentation marketplace",
        "freelance developers", "remote engineer marketplace", "vetting engineers",
        "andela", "turing.com", "toptal",
    ),
}

_MODEL_FAMILIES: dict[str, str] = {
    "agency": "b2b_services",
    "services": "b2b_services",
    "consulting": "b2b_services",
    "saas": "b2b_software",
    "product": "b2b_software",
    "software": "b2b_software",
    "b2b": "b2b_software",
    "marketplace": "marketplace",
    "ecommerce": "retail",
    "e-commerce": "retail",
    "retail": "retail",
    "shopping": "retail",
    "fintech": "fintech",
    "payments": "fintech",
    "restaurant": "food",
    "fast food": "food",
    "fast-food": "food",
    "qsr": "food",
    "food": "food",
    "beauty": "consumer_services",
    "salon": "consumer_services",
    "cosmetics": "retail",
    "fashion": "retail",
    "apparel": "retail",
    "other": "other",
}

_SOFTWARE_PEER_TOKENS = (
    "software",
    "agency",
    "technology",
    "tech",
    "saas",
    "it services",
    "it-services",
    "it consulting",
    "it solutions",
    "digital solutions",
    "digital transformation",
    "ai transformation",
    "ai services",
    "cloud consulting",
    "cloud solutions",
    "product engineering",
    "software engineering",
    "digital agency",
    "web development",
    "app development",
)

_FOOD_PEER_TOKENS = (
    "fast food",
    "fast-food",
    "qsr",
    "pizza",
    "burger",
    "restaurant",
    "cafe",
    "café",
    "fried chicken",
    "shawarma",
    "bakery",
    "food chain",
    "food brand",
    "cloud kitchen",
    "biryani",
    "broast",
    "eatery",
)

FoodTier = str  # local_specialty | national_chain | global_franchise
_FOOD_TIER_LOCAL = "local_specialty"
_FOOD_TIER_NATIONAL = "national_chain"
_FOOD_TIER_GLOBAL = "global_franchise"

_LOCAL_SPECIALTY_TOKENS = (
    "shawarma",
    "shwarma",
    "doner",
    "kebab",
    "kabab",
    "bakery",
    "patisserie",
    "pastry",
    "dessert",
    "cake shop",
    "cafe",
    "café",
    "coffee shop",
    "cloud kitchen",
    "ghost kitchen",
    "street food",
    "food truck",
    "juice bar",
    "smoothie",
    "ice cream parlor",
    "fine pizza",
    "gourmet pizza",
    "bistro",
    "diner",
    "eatery",
    "pizzeria",
    "artisan",
)

_NATIONAL_CHAIN_TOKENS = (
    "cheezious",
    "howdy",
    "optp",
    "ranchers",
    "broadway pizza",
    "california pizza",
    "pizza max",
    "burger lab",
    "layers",
    "layers bakeshop",
    "kitchen cuisine",
    "tehzeeb",
    "gourmet",
    "del frio",
    "sweet tooth",
    "rahat",
    "pie in the sky",
    "food chain",
    "restaurant chain",
    "multi-city",
    "nationwide",
)

_GLOBAL_FOOD_FRANCHISE_NAMES = (
    "pizza hut",
    "domino",
    "dominos",
    "papa john",
    "papajohn",
    "kfc",
    "kentucky fried",
    "mcdonald",
    "mcdonalds",
    "hardee",
    "hardees",
    "burger king",
    "subway",
    "starbucks",
    "tim hortons",
    "wendy",
    "popeyes",
    "taco bell",
    "dunkin",
)

_GLOBAL_FOOD_FRANCHISE_DOMAINS = (
    "pizzahut.com",
    "pizzahut.com.pk",
    "dominos.com",
    "dominos.com.pk",
    "papajohns.com",
    "papajohns.com.pk",
    "kfc.com",
    "kfcpakistan.com",
    "mcdonalds.com",
    "mcdonalds.com.pk",
    "hardees.com",
    "hardees.com.pk",
    "bk.com",
    "burgerking.com",
    "subway.com",
    "starbucks.com",
)

_BEAUTY_PEER_TOKENS = (
    "beauty",
    "salon",
    "parlour",
    "parlor",
    "spa",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin care",
    "makeup",
    "make-up",
    "hair",
    "aesthetic",
    "aesthetics",
    "dermatology",
    "hair studio",
    "beauty studio",
    "bridal",
    "grooming",
    "nail",
    "lash",
    "brows",
    "facial",
)

FoodFormat = str
_FOOD_FORMAT_RESTAURANT = "restaurant"
_FOOD_FORMAT_CAFE = "cafe"
_FOOD_FORMAT_BAKERY = "bakery"
_FOOD_FORMAT_BURGER = "burger"
_FOOD_FORMAT_PIZZA = "pizza"
_FOOD_FORMAT_ASIAN = "asian"
_FOOD_FORMAT_SHAWARMA = "shawarma"
_FOOD_FORMAT_GENERAL = "general"

_FURNITURE_HOME_TOKENS = (
    "furniture", "wardrobe", "wardrobes", "kitchen cabinet", "kitchen cabinets",
    "home cucine", "interior design", "sofa", "mattress", "furnishings",
)

PeerScale = str  # boutique | mid_market | enterprise
_PEER_BOUTIQUE = "boutique"
_PEER_MID = "mid_market"
_PEER_ENTERPRISE = "enterprise"

_BOUTIQUE_SCALE_TOKENS = (
    "boutique",
    "studio",
    "freelance",
    "freelancer",
    "indie",
    "startup",
    "small business",
    "small agency",
    "local shop",
    "family owned",
    "family-owned",
    "independent",
    "cloud kitchen",
    "ghost kitchen",
    "single outlet",
    "home based",
    "home-based",
)


def _context_blob(*parts: object) -> str:
    return " ".join(_as_str(p) for p in parts if p).lower()


def _detect_verticals(text: str) -> set[str]:
    blob = _as_str(text).lower()
    if not blob:
        return set()
    found: set[str] = set()
    for vertical, markers in _VERTICAL_MARKERS.items():
        if any(m in blob for m in markers):
            found.add(vertical)
    if re.search(r"\b\w{2,}pay\b", blob) or re.search(r"\b\w*wallet\b", blob) or "paisa" in blob:
        found.add("fintech")
    if "pitb" in blob or (
        bool(re.search(r"\b\w+\s+board\b", blob))
        and any(tok in blob for tok in ("information technology", "it board", "government", "pakistan"))
    ):
        found.add("government")
    if ".gov." in blob or blob.endswith(".gov") or ".gob." in blob:
        found.add("government")
    return found


def _looks_like_beauty_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob:
        return False
    if any(tok in blob for tok in ("systems limited", "netsol", "cheezious", "pizza", "burger", "fast food")):
        return False
    if "beauty_personal_care" in _detect_verticals(blob):
        return True
    return any(tok in blob for tok in _BEAUTY_PEER_TOKENS)


def _looks_like_food_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob:
        return False
    if _looks_like_beauty_client(blob):
        return False
    if any(tok in blob for tok in ("systems limited", "netsol", "software house", "custom software")):
        return False
    if "food_qsr" in _detect_verticals(blob):
        return True
    return any(tok in blob for tok in _FOOD_PEER_TOKENS) or any(
        tok in blob for tok in ("cheezious", "pizza", "burger", "shawarma", "fast food", "restaurant", "bakery")
    )


def _is_global_food_franchise(name: str, website: str | None = None) -> bool:
    """Pizza Hut / Domino's / KFC-scale franchises — not peers for indie local food brands."""
    n = _as_str(name).strip().lower()
    if n:
        compact = re.sub(r"[^a-z0-9]+", "", n)
        for blocked in _GLOBAL_FOOD_FRANCHISE_NAMES:
            blocked_c = re.sub(r"[^a-z0-9]+", "", blocked)
            if blocked in n or (blocked_c and blocked_c in compact):
                return True
    host = _domain_of(website or "")
    if host:
        for blocked in _GLOBAL_FOOD_FRANCHISE_DOMAINS:
            if host == blocked or host.endswith("." + blocked):
                return True
    return False


def _food_tier_from_blob(*parts: object) -> FoodTier:
    """
    Infer food brand scale/format.
    local_specialty = shawarma / bakery / cafe / indie QSR
    national_chain = strong multi-city local chains
    global_franchise = Pizza Hut / KFC / McDonald's class
    """
    blob = _context_blob(*parts)
    if not blob:
        return _FOOD_TIER_LOCAL
    # Prefer name/industry/niche over long website copy (sites often mention giants as comps).
    identity = blob[:420]
    if any(tok in identity for tok in _GLOBAL_FOOD_FRANCHISE_NAMES):
        return _FOOD_TIER_GLOBAL
    # Named national chains first (Cheezious etc. can contain "pizza" without being global)
    if any(tok in identity for tok in _NATIONAL_CHAIN_TOKENS):
        return _FOOD_TIER_NATIONAL
    if any(tok in identity for tok in _LOCAL_SPECIALTY_TOKENS):
        return _FOOD_TIER_LOCAL
    # Unknown pizza/burger brand → local specialty, NOT national (avoids Fine Pizza ≈ Pizza Hut)
    if any(tok in identity for tok in ("pizza", "burger", "fried chicken", "qsr", "fast food", "fast-food")):
        return _FOOD_TIER_LOCAL
    return _FOOD_TIER_LOCAL


def _food_tier_compatible(client_tier: FoodTier, rival_tier: FoodTier) -> bool:
    if client_tier == _FOOD_TIER_GLOBAL:
        return True
    if client_tier == _FOOD_TIER_NATIONAL:
        # National PK chains peer with national/local — not Pizza Hut / KFC global franchises
        return rival_tier in {_FOOD_TIER_NATIONAL, _FOOD_TIER_LOCAL}
    # local_specialty: never keep global franchises
    return rival_tier in {_FOOD_TIER_LOCAL, _FOOD_TIER_NATIONAL}


def _looks_like_furniture_or_home_brand(name: str, *extra: object) -> bool:
    """Bare 'Cucina' / furniture retailers are not restaurant peers."""
    key = _rival_name_key(name)
    blob = _context_blob(name, *extra)
    if key in {"cucina", "homecucine", "minicucine"}:
        return True
    if any(tok in blob for tok in _FURNITURE_HOME_TOKENS):
        # Allow real restaurants that mention "interior" once in marketing copy only if
        # they also look clearly like food venues.
        if any(tok in blob for tok in ("restaurant", "cafe", "café", "dining", "menu", "brunch")):
            return False
        return True
    return False


def _looks_like_fmcg_or_snack_brand(name: str, *extra: object) -> bool:
    """Packaged FMCG food/snack manufacturers are not restaurant/dining peers."""
    blob = _context_blob(name, *extra)
    if not blob:
        return False
    fmcg_tokens = (
        "snack foods", "packaged foods", "chips manufacturer", "crisps manufacturer",
        "confectionery manufacturer", "beverage manufacturer", "biscuit manufacturer",
        "fmcg", "consumer goods",
    )
    return any(t in blob for t in fmcg_tokens)


def _food_local_name_denied(name: str, market: str | None = None) -> bool:
    """Compatibility stub for local food deny lists."""
    return False


def _food_format_from_blob(*parts: object) -> FoodFormat:
    blob = _context_blob(*parts)
    if not blob:
        return _FOOD_FORMAT_GENERAL
    compact = re.sub(r"[^a-z0-9]+", " ", blob).strip()
    first = compact.split(" ")[0] if compact else ""

    # Known brands by category (before weak tokens)
    if first in {"mad", "jugnu"} or ("johnny" in compact and "jugnu" in compact):
        return _FOOD_FORMAT_BURGER
    if first in {"ginsoy", "ginyaki", "xinyaki"}:
        return _FOOD_FORMAT_ASIAN
    if "khan baba" in compact or compact.startswith("khanbaba"):
        return _FOOD_FORMAT_RESTAURANT
    if any(tok in compact for tok in ("layers", "butlers")) or "bakeshop" in compact:
        return _FOOD_FORMAT_BAKERY
    if any(tok in compact for tok in ("espresso lab", "gloria jean", "second cup")):
        return _FOOD_FORMAT_CAFE
    if "meet me in paris" in compact or first == "meet":
        return _FOOD_FORMAT_RESTAURANT
    if (
        first in {"pita", "heypita"}
        or "shawarma stop" in compact
        or "arabic shawarma" in compact
        or "sultan shawarma" in compact
    ):
        return _FOOD_FORMAT_SHAWARMA

    if any(tok in blob for tok in ("shawarma", "shwarma", "doner", "kebab", "kabab")):
        # Desi BBQ / karahi houses sometimes mention kebab — not shawarma shops
        if any(
            tok in blob
            for tok in ("karahi", "qeema", "tandoor", "desi ghee", "khan baba", "bbq restaurant", "barbeque")
        ) and "shawarma" not in blob and "shwarma" not in blob:
            return _FOOD_FORMAT_RESTAURANT
        return _FOOD_FORMAT_SHAWARMA
    if any(tok in blob for tok in ("karahi", "qeema naan", "desi restaurant", "pakistani cuisine", "tawa piece")):
        return _FOOD_FORMAT_RESTAURANT
    if any(
        tok in blob
        for tok in (
            "johnny & jugnu", "johnny and jugnu", "burger lab", "mad burger", "smash burger",
            "burger",
        )
    ):
        return _FOOD_FORMAT_BURGER
    if any(tok in blob for tok in ("pizza", "domino", "pizza hut", "broadway pizza", "cheezious")):
        return _FOOD_FORMAT_PIZZA
    if any(
        tok in blob
        for tok in (
            "ginyaki", "ginsoy", "xinyaki", "chinese", "oriental", "asian", "sushi",
            "noodles", "dumpling", "thai", "japanese",
        )
    ):
        return _FOOD_FORMAT_ASIAN
    # Cake shop / bakery — NOT a restaurant peer
    if any(
        tok in blob
        for tok in (
            "bakery", "bakeshop", "patisserie", "pastry", "dessert shop", "cake shop",
            "cakes", "chocolate shop", "confection",
        )
    ):
        return _FOOD_FORMAT_BAKERY
    # Coffee-led cafe (not full restaurant dining)
    if any(
        tok in blob
        for tok in (
            "coffee shop", "coffeehouse", "espresso bar", "gloria jean", "second cup",
            "espresso lab",
        )
    ) and not any(tok in blob for tok in ("restaurant", "dine-in", "dining", "crepe", "entrée", "entree")):
        return _FOOD_FORMAT_CAFE
    # Full restaurant / casual dining / concept restaurants
    if any(
        tok in blob
        for tok in (
            "restaurant", "restaurants", "casual dining", "fine dining", "dine-in", "dining",
            "crepe", "crêpe", "baguette sandwich", "croque", "french street", "french casual",
            "french restaurant", "bistro", "brasserie", "eatery", "kitchen",
            "meet me in paris", "salt'n pepper", "salt n pepper", "arcadian",
        )
    ):
        return _FOOD_FORMAT_RESTAURANT
    if any(tok in blob for tok in ("cafe", "café", "coffee", "brunch")):
        # Bare "cafe" without restaurant cues → cafe; with paris/crepe already caught above
        return _FOOD_FORMAT_CAFE
    if "paris" in blob and not any(tok in blob for tok in ("pizza", "burger", "software", "cake", "bakery")):
        return _FOOD_FORMAT_RESTAURANT
    return _FOOD_FORMAT_GENERAL


def _food_format_compatible(client_fmt: FoodFormat, rival_fmt: FoodFormat) -> bool:
    """Same food category only — restaurant peers must be restaurants, not cake shops."""
    if not client_fmt or client_fmt == _FOOD_FORMAT_GENERAL:
        return True
    if not rival_fmt or rival_fmt == _FOOD_FORMAT_GENERAL:
        # Unknown rival category: do not treat as a match for strict client categories
        return False
    return client_fmt == rival_fmt


def _client_peer_hint(*parts: object) -> str:
    """Human label for empty-rival errors matching the client's actual industry."""
    if _looks_like_food_client(*parts):
        return _food_rival_peer_hint(*parts)
    if _looks_like_beauty_client(*parts):
        return "beauty-salon / makeup-studio rivals"
    if _looks_like_software_peer_client(*parts):
        return "software-house / digital-agency rivals"
    blob = _context_blob(*parts)
    for w in ("fashion", "clothing", "apparel", "boutique"):
        if w in blob:
            return "fashion / apparel rivals"
    for w in ("healthcare", "clinic", "hospital", "medical"):
        if w in blob:
            return "healthcare / clinic rivals"
    for w in ("real estate", "property"):
        if w in blob:
            return "real estate rivals"
    for w in ("education", "school", "academy", "learning"):
        if w in blob:
            return "education rivals"
    return "same-industry peer rivals"


def _food_rival_peer_hint(*parts: object) -> str:
    """Human label for empty-rival errors — match the client's actual food format."""
    fmt = _food_format_from_blob(*parts)
    return {
        _FOOD_FORMAT_SHAWARMA: "shawarma rivals",
        _FOOD_FORMAT_PIZZA: "pizza rivals",
        _FOOD_FORMAT_BURGER: "burger rivals",
        _FOOD_FORMAT_BAKERY: "bakery rivals",
        _FOOD_FORMAT_CAFE: "cafe rivals",
        _FOOD_FORMAT_ASIAN: "Asian restaurant rivals",
        _FOOD_FORMAT_RESTAURANT: "restaurant rivals",
    }.get(fmt, "restaurant / cafe / bakery rivals")


def _peer_scale_from_blob(*parts: object, name: str = "", website: str | None = None) -> PeerScale:
    """
    Infer comparable market position for any client/rival.
    Used for food, software, and future verticals so discovery stays peer-level.
    """
    blob = _context_blob(*parts)
    blob_lower = blob.lower() if blob else ""
    nm = _as_str(name) or (blob_lower.split(" ")[0] if blob_lower else "")

    if _is_global_megarival(nm or blob_lower[:80], website) or _is_global_food_franchise(nm or blob_lower[:80], website):
        return _PEER_ENTERPRISE

    if "scale: enterprise" in blob_lower:
        return _PEER_ENTERPRISE
    if "scale: boutique" in blob_lower or "scale: startup" in blob_lower:
        return _PEER_BOUTIQUE

    if _looks_like_food_client(blob_lower):
        food_tier = _food_tier_from_blob(*parts)
        if food_tier == _FOOD_TIER_GLOBAL:
            return _PEER_ENTERPRISE
        if food_tier == _FOOD_TIER_NATIONAL:
            return _PEER_MID
        return _PEER_BOUTIQUE

    if any(tok in blob_lower[:420] for tok in _BOUTIQUE_SCALE_TOKENS):
        return _PEER_BOUTIQUE

    if _looks_like_software_peer_client(blob_lower):
        if any(tok in blob_lower for tok in ("consultancy", "consulting firm", "enterprise software", "fortune 500", "publicly traded")):
            return _PEER_ENTERPRISE
        return _PEER_MID

    # Generic future niches (retail, fashion, education, clinics, etc.)
    if any(
        tok in blob_lower
        for tok in (
            "global brand",
            "multinational",
            "fortune 500",
            "worldwide chain",
            "nationwide",
            "retail chain",
            "fashion chain",
            "clothing chain",
            "flagship store",
            "flagship stores",
            "stores across",
            "outlets across",
            "hundreds of stores",
            "conglomerate",
            "market leader",
            "leading retail",
            "household brand",
            "major brand",
            "leading brand",
            "enterprise",
            "leading fashion",
            "major fashion",
            "retail brand",
            "largest fashion",
            "textile giant",
            "retailer",
            "retail network",
        )
    ):
        return _PEER_ENTERPRISE
    if any(tok in blob_lower for tok in _BOUTIQUE_SCALE_TOKENS) or any(
        tok in blob_lower for tok in ("local", "neighborhood", "specialty shop", "single studio")
    ):
        return _PEER_BOUTIQUE
    return _PEER_MID


def _peer_scale_compatible(client_scale: PeerScale, rival_scale: PeerScale) -> bool:
    if client_scale == _PEER_ENTERPRISE:
        return True
    if client_scale == _PEER_MID:
        # Mid-market ≠ global franchise giants (Pizza Hut / McDonald's)
        return rival_scale in {_PEER_BOUTIQUE, _PEER_MID}
    # boutique / indie: never keep enterprise giants as "peers"
    return rival_scale in {_PEER_BOUTIQUE, _PEER_MID}


def _looks_like_software_peer_client(*parts: object) -> bool:
    blob = _context_blob(*parts)
    if not blob:
        return False
    if _looks_like_food_client(blob) or _looks_like_beauty_client(blob):
        return False
    if "cheezious" in blob or "cheese-flavored" in blob:
        return False
    verts = _detect_verticals(blob)
    if "software_services" in verts or "data_ai" in verts:
        return True
    return any(tok in blob for tok in _SOFTWARE_PEER_TOKENS) or any(
        tok in blob for tok in (
            "systems limited", "software house", "software company", "it services",
            "it consulting", "digital solutions", "tech", "nextbridge",
        )
    )


def _detect_education_tier(*parts: object) -> str:
    """Classifies education client/rival into: university | school | college | academy | general."""
    blob = " " + _context_blob(*parts).lower() + " "
    if not blob.strip():
        return "general"

    # 1. School markers (check first if not explicitly a university)
    if re.search(
        r"\b(school|schools|grammar school|high school|primary school|elementary school|middle school|preschool|kindergarten|k-?12|o\s*levels?|matric|beaconhouse|the city school|city school|roots|lgs|lahore grammar|karachi grammar|kgs|froebel|learning alliance|the educators|sicas|bloomfield|gems education|nord anglia|taaleem|choueifat|eton college|exeter)\b",
        blob,
    ):
        if not re.search(
            r"\b(university|universities|degree awarding|bachelor|master\b|master'\''s|phd|undergraduate|postgraduate|lums|nust|fast nuces|giki|iba karachi|aku|comsats|szabist|uet|itu|pu\.edu|qau|harvard|mit|stanford|university of oxford|university of cambridge|kaust|kfupm)\b",
            blob,
        ):
            return "school"

    # 2. University / Higher Education markers
    if re.search(
        r"\b(university|universities|higher ed|higher education|degree awarding|bachelor|master\b|master'\''s|phd|undergraduate|postgraduate|lums|nust|fast nuces|giki|iba karachi|aku|szabist|comsats|pu\.edu|qau|itu|uet|harvard|mit|stanford|oxford university|cambridge university|university of cambridge|university of oxford|kaust|kfupm|aus\.edu|nyuad|fccollege|lahore school of economics)\b",
        blob,
    ):
        return "university"

    # 3. Intermediate / Pre-university College
    if re.search(
        r"\b(intermediate college|junior college|punjab college|pgc|superior college|concordia college|fsc|ics|higher secondary)\b",
        blob,
    ):
        return "college"

    # 4. Test Prep / Coaching Academy / Edtech
    if re.search(
        r"\b(test prep|entry test|mdcat|ecat|sat prep|ielts|coaching center|tuition academy|kips academy|step prep|maqsad|noon academy|nearpeer)\b",
        blob,
    ):
        return "academy"

    if re.search(r"\b(school|schools)\b", blob):
        return "school"
    if re.search(r"\b(college|colleges)\b", blob):
        return "college"
    if re.search(r"\b(academy|academies)\b", blob):
        return "academy"

    return "general"


def _industry_category_from_client(client: ClientBrand) -> str:
    notes = _as_str(getattr(client, "notes", ""))
    for line in notes.splitlines():
        if line.lower().startswith("industry category:"):
            return line.split(":", 1)[1].strip().lower()
    return ""


def _set_industry_category(client: ClientBrand, industry_category: str) -> None:
    industry_category = _as_str(industry_category).strip().lower()
    notes = _as_str(getattr(client, "notes", ""))
    lines = [ln for ln in notes.splitlines() if not ln.lower().startswith("industry category:")]
    if industry_category:
        insert_at = 0
        for i, ln in enumerate(lines):
            if ln.lower().startswith("market:") or ln.lower().startswith("business model:"):
                insert_at = i + 1
        lines.insert(insert_at, f"Industry category: {industry_category}")
    client.notes = "\n".join(lines).strip() or None


def _business_model_from_client(client: ClientBrand) -> str:
    notes = _as_str(getattr(client, "notes", ""))
    for line in notes.splitlines():
        if line.lower().startswith("business model:"):
            return line.split(":", 1)[1].strip()
    return ""


def _detect_industry_category(*parts: object) -> str:
    blob = " " + _context_blob(*parts).lower() + " "
    if not blob.strip():
        return "other"
    # 0. Explicit taxonomy tag passed directly or stored in metadata
    for part in parts:
        if isinstance(part, str):
            p_strip = part.strip().lower()
            if p_strip in {
                "data_ai", "software", "food", "beauty", "education", "university",
                "school", "college", "academy", "fintech", "healthcare", "energy",
                "logistics", "fashion", "hospitality", "fitness", "automotive",
                "real_estate", "telecom", "retail",
            }:
                return p_strip
            if "industry category:" in p_strip:
                for line in p_strip.splitlines():
                    if line.strip().startswith("industry category:"):
                        cand = line.split(":", 1)[1].strip().lower()
                        if cand in {
                            "data_ai", "software", "food", "beauty", "education", "university",
                            "school", "college", "academy", "fintech", "healthcare", "energy",
                            "logistics", "fashion", "hospitality", "fitness", "automotive",
                            "real_estate", "telecom", "retail",
                        }:
                            return cand
        elif hasattr(part, "notes"):
            stored = _industry_category_from_client(part)
            if stored:
                return stored

    # Semantic keyword patterns
    if re.search(r"\b(ai\s+solutions?|artificial\s+intelligence|machine\s+learning|data\s+analytics|enterprise\s+ai|bi\s+dashboards?|data\s+platform|competitive\s+intelligence|analytics\s+consultancy|chatbot|enterprise\s+ai\s+solutions|ai\s+transformation)\b", blob):
        return "data_ai"
    if re.search(r"\b(software\s+house|software\s+development|software\s+agency|custom\s+software|it\s+services|it\s+consulting|it\s+solutions|digital\s+solutions|digital\s+transformation|cloud\s+solutions|saas|web\s+development|digital\s+product\s+firm|nextbridge)\b", blob):
        return "software"
    if _looks_like_beauty_client(blob) or re.search(r"\b(beauty|salon|salons|parlour|parlor|spa|bridal|makeup|skincare|hair\s+styling|cosmetics?)\b", blob):
        return "beauty"
    if _looks_like_food_client(blob) or re.search(r"\b(fast\s*food|qsr|restaurants?|cafes?|caf[eé]|baker(y|ies)|shawarma|burgers?|pizzas?|food\s+chain)\b", blob):
        return "food"
    if re.search(r"\b(schools?|colleges?|universit(y|ies)|higher\s+ed|academ(y|ies)|education(al)?|board\s+of\s+(intermediate|secondary|education)|bise|curriculum|examination\s+board|matric|intermediate)\b", blob):
        edu_tier = _detect_education_tier(blob)
        if edu_tier in {"university", "school", "college", "academy"}:
            return edu_tier
        return "education"
    if re.search(r"\b(government|ministry\s+of|department\s+of|police|judiciary|court|authority|commission|public\s+sector)\b", blob):
        return "government"
    if re.search(r"\b(fintech|banks?|banking|wallets?|payments?|easypaisa|jazzcash|nayapay|sadapay|stripe)\b", blob):
        return "fintech"
    if re.search(r"\b(healthcare|hospitals?|clinics?|telemedicine|diagnostic|pharma|medical)\b", blob):
        return "healthcare"
    if re.search(r"\b(logistics|courier|freight|shipping|warehousing|cargo)\b", blob):
        return "logistics"
    if re.search(r"\b(fashion|apparel|clothing|boutique|couture|pret)\b", blob):
        return "fashion"
    if re.search(r"\b(hotels?|resorts?|hospitality|motels?)\b", blob):
        return "hospitality"
    if re.search(r"\b(gyms?|fitness|workout|wellness)\b", blob):
        return "fitness"
    if re.search(r"\b(real\s+estate|property|housing)\b", blob):
        return "real_estate"
    if re.search(r"\b(solar|energy|renewable|cleantech)\b", blob):
        return "energy"
    if re.search(r"\b(cars?|automotive|automobile|dealership)\b", blob):
        return "automotive"
    if re.search(r"\b(gaming|video\s+games?|consoles?|playstation|xbox|nintendo|esports)\b", blob):
        return "gaming"
    return "other"


def _incompatible_peer(
    *,
    client_model: str = "",
    client_industry: str = "",
    client_niche: str = "",
    rival_model: str = "",
    rival_industry: str = "",
    rival_blob: str = "",
    client_name: str = "",
) -> bool:
    """True when rival is clearly not the same kind of business as the client."""
    client_l = f"{client_name} {client_model} {client_industry} {client_niche}".lower()
    rival_l = f"{rival_model} {rival_industry} {rival_blob}".lower()

    client_cat = _detect_industry_category(client_l)
    rival_cat = _detect_industry_category(rival_l)

    # 1. Consumer vs B2B / Cross-vertical mismatch
    client_is_food = _looks_like_food_client(client_l) or client_cat == "food"
    rival_is_food = _looks_like_food_client(rival_l) or rival_cat == "food"

    client_is_beauty = _looks_like_beauty_client(client_l) or client_cat == "beauty"
    rival_is_beauty = _looks_like_beauty_client(rival_l) or rival_cat == "beauty"

    client_is_software = _looks_like_software_peer_client(client_l) or client_cat in {"software", "data_ai"}
    rival_is_software = _looks_like_software_peer_client(rival_l) or rival_cat in {"software", "data_ai"}

    client_is_gaming = client_cat == "gaming"
    rival_is_gaming = rival_cat == "gaming" or any(tok in rival_l for tok in ("xbox", "playstation", "nintendo", "video games", "gaming console"))

    client_is_edu = client_cat in {"education", "university", "school", "college", "academy"} or bool(
        re.search(r"\b(schools?|colleges?|universit(y|ies)|higher\s+ed|academ(y|ies)|education(al)?|board\s+of\s+(intermediate|secondary|education)|bise)\b", client_l)
    )
    rival_is_edu = rival_cat in {"education", "university", "school", "college", "academy"} or bool(
        re.search(r"\b(schools?|colleges?|universit(y|ies)|higher\s+ed|academ(y|ies)|education(al)?|board\s+of\s+(intermediate|secondary|education)|bise)\b", rival_l)
    )

    client_is_gov = client_cat == "government"
    rival_is_gov = rival_cat == "government" or any(
        tok in rival_l for tok in (".gov", ".mil", "ministry of", "department of", "government of", "public sector", "examination board", "board of education", "board of intermediate")
    )

    # Never allow government / public board noise for commercial clients
    if rival_is_gov and not client_is_gov:
        return True

    # Never allow dictionary / definition / encyclopedic results as competitor peers
    if any(tok in rival_l for tok in ("definition & meaning", "dictionary", "thesaurus", "synonyms and antonyms", "merriam-webster", "wiktionary", "biceps femoris", "pubmed")):
        return True

    # Strict isolation for gaming
    if rival_is_gaming and not client_is_gaming:
        return True
    if client_is_gaming and not rival_is_gaming:
        return True

    # Strict isolation for food: food clients compete with food providers, never education, tech, beauty, etc.
    if client_is_food and not rival_is_food:
        return True
    if rival_is_food and not client_is_food:
        return True

    # Intra-Food Sub-Category & Format Tier Isolation
    if client_is_food and rival_is_food:
        client_shawarma = any(tok in client_l for tok in ("shawarma", "wrap", "falafel", "doner", "gyro", "middle eastern"))
        client_pizza = any(tok in client_l for tok in ("pizza", "pizzeria"))
        client_fine_dining = any(tok in client_l for tok in ("fine dining", "steakhouse", "italian fine", "ristorante", "luxury dining", "gourmet dining"))

        rival_shawarma = any(tok in rival_l for tok in ("shawarma", "wrap", "falafel", "doner", "gyro", "middle eastern"))
        rival_pizza = any(tok in rival_l for tok in ("pizza", "pizzeria", "caprinos", "pizza hut", "domino", "papa john"))
        rival_fine_dining = any(tok in rival_l for tok in ("fine dining", "steakhouse", "italian fine", "ristorante", "cosa nostra", "luxury dining", "gourmet restaurant", "upscale restaurant"))

        # Shawarma specialist vs Fine Dining or pure Pizza chain
        if client_shawarma and not client_fine_dining:
            if rival_fine_dining:
                return True
            if rival_pizza and not rival_shawarma:
                return True

        # Pizza specialist vs pure Shawarma spot
        if client_pizza and not client_shawarma:
            if rival_shawarma and not rival_pizza:
                return True

    # Strict isolation for beauty
    if client_is_beauty and not rival_is_beauty:
        return True
    if rival_is_beauty and not client_is_beauty:
        return True

    # Strict isolation for education
    if client_is_edu and not rival_is_edu:
        return True
    if rival_is_edu and not client_is_edu:
        return True

    client_is_fashion = client_cat == "fashion" or any(tok in client_l for tok in ("apparel", "clothing", "fashion", "luxury", "pret", "textile"))
    rival_is_fashion = rival_cat == "fashion" or any(tok in rival_l for tok in ("apparel", "clothing", "fashion", "luxury", "pret", "textile"))

    # Strict isolation for software
    if client_is_software and (rival_is_food or rival_is_beauty or rival_is_edu or rival_is_fashion):
        return True
    if rival_is_software and (client_is_food or client_is_beauty or client_is_edu or client_is_fashion):
        return True

    if client_is_fashion:
        if rival_is_food or rival_is_software or rival_is_edu or rival_is_gaming or rival_is_gov:
            return True
        if any(tok in rival_l for tok in ("automotive", "motor company", "car manufacturer", "fintech", "payment gateway", "defense contractor")):
            return True
        client_pret = any(tok in client_l for tok in ("pret", "unstitched", "lawn", "kurta", "eastern wear", "shalwar"))
        rival_is_western_denim = any(tok in rival_l for tok in ("denim", "jeans", "western wear", "western casuals", "trucker jacket")) and not any(tok in rival_l for tok in ("pret", "lawn", "kurta", "eastern wear", "shalwar", "unstitched", "festive"))
        if client_pret and rival_is_western_denim:
            return True

    client_is_legal = client_cat == "legal" or any(tok in client_l for tok in ("legal", "law firm", "attorney", "solicitor", "advocate", "m&a"))
    rival_is_legal = rival_cat == "legal" or any(tok in rival_l for tok in ("legal", "law firm", "attorney", "solicitor", "advocate", "defense attorney", "bail bonds"))
    if client_is_legal and rival_is_legal:
        client_corporate = any(tok in client_l for tok in ("corporate", "m&a", "mergers", "private equity", "tax", "securities"))
        rival_criminal = any(tok in rival_l for tok in ("criminal defense", "dui", "misdemeanor", "bail bonds", "traffic ticket", "felony"))
        if client_corporate and rival_criminal:
            return True

    # 2. Specialized AI vs generic body shop / low-end legacy outsourcing
    if client_cat == "data_ai":
        if any(tok in rival_l for tok in ("outsourcing", "staff augmentation", "body shop", "offshore outsourcing", "legacy custom software")):
            return True

    # 3. Known category mismatch
    if client_cat != "other" and rival_cat != "other" and client_cat != rival_cat:
        if {client_cat, rival_cat} <= {"data_ai", "software"}:
            pass
        else:
            return True

    return False

__all__ = [
    "_RETAIL_MARKETPLACE_MARKERS",
    "_GOVERNMENT_MARKERS",
    "_SOFTWARE_PEER_TOKENS",
    "_FOOD_PEER_TOKENS",
    "FoodTier",
    "_FOOD_TIER_LOCAL",
    "_FOOD_TIER_NATIONAL",
    "_FOOD_TIER_GLOBAL",
    "_LOCAL_SPECIALTY_TOKENS",
    "_NATIONAL_CHAIN_TOKENS",
    "_GLOBAL_FOOD_FRANCHISE_NAMES",
    "_GLOBAL_FOOD_FRANCHISE_DOMAINS",
    "_BEAUTY_PEER_TOKENS",
    "FoodFormat",
    "_FOOD_FORMAT_RESTAURANT",
    "_FOOD_FORMAT_CAFE",
    "_FOOD_FORMAT_BAKERY",
    "_FOOD_FORMAT_BURGER",
    "_FOOD_FORMAT_PIZZA",
    "_FOOD_FORMAT_ASIAN",
    "_FOOD_FORMAT_SHAWARMA",
    "_FOOD_FORMAT_GENERAL",
    "_FURNITURE_HOME_TOKENS",
    "PeerScale",
    "_PEER_BOUTIQUE",
    "_PEER_MID",
    "_PEER_ENTERPRISE",
    "_BOUTIQUE_SCALE_TOKENS",
    "_context_blob",
    "_detect_verticals",
    "_looks_like_beauty_client",
    "_looks_like_food_client",
    "_is_global_food_franchise",
    "_food_tier_from_blob",
    "_food_tier_compatible",
    "_looks_like_furniture_or_home_brand",
    "_looks_like_fmcg_or_snack_brand",
    "_food_local_name_denied",
    "_food_format_from_blob",
    "_food_format_compatible",
    "_client_peer_hint",
    "_food_rival_peer_hint",
    "_peer_scale_from_blob",
    "_peer_scale_compatible",
    "_looks_like_software_peer_client",
    "_detect_education_tier",
    "_industry_category_from_client",
    "_set_industry_category",
    "_business_model_from_client",
    "_detect_industry_category",
    "_incompatible_peer"
]
