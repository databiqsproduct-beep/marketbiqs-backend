"""Search engine noise domains: directories, review sites, gig platforms, and aggregators."""

from __future__ import annotations

from urllib.parse import urlparse

SERP_NOISE_DOMAINS: set[str] = {
    # Freelancing & Gig Platforms (NOT B2B Software / Agency / Store Peers)
    "upwork.com",
    "fiverr.com",
    "freelancer.com",
    "toptal.com",
    "guru.com",
    "peopleperhour.com",
    "bebee.com",
    "tracxn.com",
    # Software & SaaS review / directory aggregators
    "g2.com",
    "capterra.com",
    "capterra.ae",
    "capterra.co.uk",
    "getapp.com",
    "softwareadvice.com",
    "trustradius.com",
    "crozdesk.com",
    "saashub.com",
    "alternativeto.net",
    "slashdot.org",
    "producthunt.com",
    "clutch.co",
    "goodfirms.co",
    "sortlist.com",
    "designrush.com",
    "upcity.com",
    "techbehemoths.com",
    "themanifest.com",
    "topdevelopers.co",
    "appfutura.com",
    "extract.co",
    "wadline.com",
    "directory.com",
    "yellowpages.com",
    "yellowpages.com.pk",
    # B2B Trade, Industry & Textile Directories / Portals
    "textileinfomedia.com",
    "fibre2fashion.com",
    "indiamart.com",
    "tradeindia.com",
    "exportersindia.com",
    "businesslist.pk",
    "pakistanbusinessjournal.com",
    "pakistanfirms.com",
    "pakistanbusinessdirectory.pk",
    "pakistanplaces.com",
    "findpk.com",
    "b2bmap.com",
    "parhlo.com",
    "mangobaaz.com",
    "brandsynario.com",
    "propergaanda.com",
    "yelp.com",
    "tripadvisor.com",
    "tripadvisor.co.uk",
    "tripadvisor.com.pk",
    "foursquare.com",
    "wheree.com",
    "jagha.pk",
    "menuprices.pk",
    "foodpanda.pk",
    "foodpanda.com",
    "foodiespakistan.pk",
    "trustpilot.com",
    "sitejabber.com",
    "glassdoor.com",
    "indeed.com",
    "indeed.co.uk",
    "zoominfo.com",
    "crunchbase.com",
    "pitchbook.com",
    "owler.com",
    "cbinsights.com",
    "zippia.com",
    "comparably.com",
    "dnb.com",
    # Social & video platforms
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "vm.tiktok.com",
    "pinterest.com",
    "reddit.com",
    "quora.com",
    "threads.net",
    "vimeo.com",
    "dailymotion.com",
    # Major news & media publications (articles/listicles, not SaaS/agency rivals)
    "forbes.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "venturebeat.com",
    "zdnet.com",
    "cnet.com",
    "businessinsider.com",
    "bloomberg.com",
    "reuters.com",
    "nytimes.com",
    "wsj.com",
    "bbc.com",
    "cnn.com",
    "mashable.com",
    "propakistani.pk",
    "techinasia.com",
    "tribune.com.pk",
    "dawn.com",
    "geo.tv",
    "thenews.com.pk",
    "dailymail.co.uk",
    "theguardian.com",
    "huffpost.com",
    "economist.com",
    "entrepreneur.com",
    "inc.com",
    "fastcompany.com",
    "hackernews.com",
    # Forums, Q&A and community discussions
    "zhihu.com",
    "quora.com",
    "reddit.com",
    # General blog hosting / publishing platforms
    "medium.com",
    "substack.com",
    "dev.to",
    "hashnode.dev",
    "hashnode.com",
    "wordpress.com",
    "wordpress.org",
    "blogspot.com",
    "tumblr.com",
    "ghost.io",
    "beehiiv.com",
    "wixsite.com",
    "weebly.com",
    "pakistantravelblog.com",
    "travelblog.org",
    "magnific.com",
    "google.com",
    "photos.google.com",
    "drive.google.com",
    "docs.google.com",
    "notion.site",
    "gitbook.io",
    "wikipedia.org",
    "en.wikipedia.org",
    "wikihow.com",
    "britannica.com",
    "merriam-webster.com",
    "dictionary.com",
    "thesaurus.com",
    "wiktionary.org",
    "cambridge.org",
    "thefreedictionary.com",
    "wordreference.com",
    "collinsdictionary.com",
    "oxfordlearnersdictionaries.com",
    "idcrawl.com",
    "bsky.app",
    "investopedia.com",
    "hbr.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "cc.gov.pk",
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "sourceforge.net",
    # Education ranking aggregators and directories
    "timeshighereducation.com",
    "topuniversities.com",
    "usnews.com",
    "shanghairanking.com",
    "edurank.org",
    "4icu.org",
    "unirank.org",
    "unirank.com",
    "studyportals.com",
    "bachelorsportal.com",
    "mastersportal.com",
    "phdportal.com",
    "universitiesrankings.com",
    "cwur.org",
    "worldresearchranking.com",
    "webometrics.info",
    # Market research report aggregators and PR wire sites
    "ibisworld.com",
    "idc.com",
    "my.idc.com",
    "gartner.com",
    "forrester.com",
    "imarcgroup.com",
    "mordorintelligence.com",
    "grandviewresearch.com",
    "expertmarketresearch.com",
    "alliedmarketresearch.com",
    "fortunebusinessinsights.com",
    "verifiedmarketresearch.com",
    "marketsandmarkets.com",
    "statista.com",
    "globenewswire.com",
    "prnewswire.com",
    "businesswire.com",
    "custommarketinsights.com",
    "thebrainyinsights.com",
    "coherentmarketinsights.com",
    # Document sharing & slide hosting
    "scribd.com",
    "slideshare.net",
    "issuu.com",
    "docdroid.net",
    # Public examination & government boards
    "biselahore.com",
    "biserwp.edu.pk",
    "bisemultan.edu.pk",
    "bisefsd.edu.pk",
    "bisebwp.edu.pk",
    "bisesahiwal.edu.pk",
    "bisegrw.edu.pk",
    "bisedgkhan.edu.pk",
    "fbise.edu.pk",
    "biek.edu.pk",
    "bsek.edu.pk",
    # Job portals & local hiring
    "rozee.pk",
    "mustakbil.com",
    "bayt.com",
    "naukri.com",
    "monster.com",
    "ziprecruiter.com",
    # Regulatory bodies
    "secp.gov.pk",
    "fbr.gov.pk",
    "pitb.gov.pk",
    "pseb.org.pk",
    "nadra.gov.pk",
    "sbp.org.pk",
}


def domain_of(url: str) -> str:
    """Extract normalized base hostname from URL or string."""
    raw = (url or "").strip().lower()
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


def is_serp_noise_domain(url_or_host: str) -> bool:
    """Check if domain or URL belongs to a directory, aggregator, or search noise domain."""
    host = domain_of(url_or_host) or (url_or_host or "").strip().lower()
    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]

    if host.endswith(".gov") or ".gov." in host or host.endswith(".mil") or ".mil." in host:
        return True
    if host.startswith("bise") or ".bise" in host or "examinationboard" in host or "boardofeducation" in host:
        return True

    if host in SERP_NOISE_DOMAINS:
        return True

    for blocked in SERP_NOISE_DOMAINS:
        if host == blocked or host.endswith("." + blocked):
            return True

    # Generic directory indicators in hostname
    if any(
        host.startswith(prefix)
        for prefix in (
            "directory.",
            "directories.",
            "listings.",
            "yellowpages.",
            "find.",
            "search.",
        )
    ):
        return True

    return False
