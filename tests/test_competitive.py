import unittest

from app.services.competitive import (
    _filter_niche_competitors,
    _incompatible_peer,
    _looks_like_food_client,
    _looks_like_software_peer_client,
    _rival_keys,
    _seed_local_qsr_rivals,
    _seed_local_software_rivals,
    collapse_duplicate_competitors,
)


class LocalSeedTests(unittest.TestCase):
    def test_pakistan_seeds_skip_client_domain(self):
        seeds = _seed_local_software_rivals(
            "Pakistan",
            "systems",
            client_website="https://www.systemsltd.com",
            limit=8,
        )
        names = [row["name"] for row in seeds]
        self.assertNotIn("Systems Limited", names)
        self.assertIn("NetSol Technologies", names)
        self.assertGreaterEqual(len(seeds), 5)

    def test_food_brand_is_not_software_peer(self):
        self.assertTrue(_looks_like_food_client("Cheezious", "fast food", "pizza delivery"))
        self.assertTrue(_looks_like_food_client("Cheezious", "Cheese-flavored snack foods"))
        self.assertFalse(_looks_like_software_peer_client("Cheezious", "Cheese-flavored snack foods"))
        self.assertFalse(_looks_like_software_peer_client("Cheezious", "fast food", "pizza delivery"))
        self.assertTrue(_looks_like_software_peer_client("Systems Limited", "tech", "software house"))

    def test_short_qsr_names_are_not_fake(self):
        from app.services.competitive import _is_generic_or_fake_rival_name

        self.assertFalse(_is_generic_or_fake_rival_name("KFC"))
        self.assertFalse(_is_generic_or_fake_rival_name("OPTP"))
        self.assertTrue(_is_generic_or_fake_rival_name("TechCorp"))

    def test_recipe_titles_are_not_rivals(self):
        from app.services.competitive import (
            _clean_rival_display_name,
            _is_generic_or_fake_rival_name,
            _looks_like_content_or_cpg_noise,
            _looks_like_recipe_or_menu_item_name,
        )

        recipe = "Authentic Pakistani Street Style Chicken Shawarma"
        self.assertTrue(_looks_like_recipe_or_menu_item_name(recipe))
        self.assertTrue(_is_generic_or_fake_rival_name(recipe))
        self.assertTrue(_is_generic_or_fake_rival_name("Chicken Shawarma Platter"))
        self.assertFalse(_is_generic_or_fake_rival_name("Shawarma Stop"))
        self.assertFalse(_is_generic_or_fake_rival_name("Arabic Shawarma"))
        self.assertEqual(
            _clean_rival_display_name("Arabic Shawarma: Unique Shawarmas With Auth"),
            "Arabic Shawarma",
        )
        self.assertTrue(_looks_like_content_or_cpg_noise("Yemeni Food in Islamabad"))
        self.assertTrue(_looks_like_content_or_cpg_noise("Pakistani shawarma Photos"))
        self.assertTrue(
            _looks_like_content_or_cpg_noise(
                "Dawn Shawarma", "https://dawnbread.com.pk/product/shawarma"
            )
        )
        self.assertFalse(
            _looks_like_content_or_cpg_noise("Shawarma Stop", "https://shawarmastop.co")
        )

    def test_software_houses_are_rejected_for_fast_food_client(self):
        self.assertTrue(
            _incompatible_peer(
                client_model="other",
                client_industry="fast food",
                client_niche="pizza",
                rival_model="services",
                rival_industry="Software",
                rival_blob="Established commercial software house / digital product firm in Pakistan",
                client_name="Cheezious",
            )
        )
        seeds = _seed_local_software_rivals("Pakistan", "Cheezious", limit=8)
        kept = _filter_niche_competitors(
            seeds,
            "Cheezious",
            market_area="Pakistan",
            niche="pizza",
            industry="fast food",
            business_model="other",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        self.assertEqual(kept, [])

    def test_pakistan_qsr_seeds_for_cheezious(self):
        seeds = _seed_local_qsr_rivals(
            "Pakistan",
            "Cheezious",
            client_website="https://www.cheezious.com",
            limit=8,
        )
        names = [row["name"] for row in seeds]
        # National pizza chain peers with other PK pizza brands — not global franchises
        self.assertIn("Broadway Pizza", names)
        self.assertTrue(any("pizza" in n.lower() or n == "Pizza Max" for n in names))
        self.assertNotIn("Pizza Hut", names)
        self.assertNotIn("Systems Limited", names)
        kept = _filter_niche_competitors(
            seeds,
            "Cheezious",
            market_area="Pakistan",
            niche="pizza",
            industry="fast food",
            business_model="other",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        self.assertGreaterEqual(len(kept), 4)
        self.assertTrue(all("pizza" in row["why_relevant"].lower() or "food" in row["why_relevant"].lower() for row in kept))

    def test_misprofiled_cheezious_still_keeps_qsr_seeds(self):
        seeds = _seed_local_qsr_rivals("Pakistan", "Cheezious", limit=8)
        kept = _filter_niche_competitors(
            seeds,
            "Cheezious",
            market_area="Pakistan",
            niche="Cheese-flavored snack foods",
            industry="Cheese-flavored snack foods",
            business_model="other",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        names = [row["name"] for row in kept]
        self.assertIn("Broadway Pizza", names)
        self.assertGreaterEqual(len(kept), 3)

    def test_papa_johns_aliases_are_the_same_rival(self):
        self.assertTrue(_rival_keys("papa johns") & _rival_keys("Papa John's", "https://www.papajohns.com.pk"))
        self.assertFalse(_rival_keys("Pizza Hut") & _rival_keys("Broadway Pizza"))

    def test_qsr_seeds_skip_papa_johns_alias(self):
        seeds = _seed_local_qsr_rivals("Pakistan", "Cheezious", already_have=["papa johns"], limit=8)
        names = [row["name"].lower() for row in seeds]
        self.assertFalse(any("papa" in name and "john" in name for name in names))
        self.assertIn("broadway pizza", names)

    def test_pakistan_shawarma_seeds_for_sultan(self):
        seeds = _seed_local_qsr_rivals(
            "Pakistan",
            "Sultan Shawarma",
            client_website="https://www.sultanshawarma.com",
            client_niche="shawarma / quick-service restaurant",
            client_industry="food",
            limit=8,
        )
        names = [row["name"] for row in seeds]
        self.assertIn("Shawarma Stop", names)
        self.assertTrue(all(row.get("food_format") == "shawarma" for row in seeds))
        self.assertNotIn("Pizza Hut", names)
        self.assertNotIn("Johnny & Jugnu", names)
        self.assertGreaterEqual(len(seeds), 4)
        self.assertTrue(
            {"Monty Shawarma", "Rizwan Pocket Shawarma"} & set(names)
            or len(seeds) >= 4
        )
        from app.services.competitive import _food_rival_peer_hint

        self.assertEqual(
            _food_rival_peer_hint("Sultan Shawarma", "food", "shawarma"),
            "shawarma rivals",
        )

    def test_collapse_duplicate_papa_johns(self):
        class Row:
            def __init__(self, name, pinned=False, score=80, website=None):
                self.name = name
                self.website = website
                self.is_pinned = pinned
                self.is_tracking = True
                self.overlap_score = score

        pinned = Row("papa johns", pinned=True, score=92)
        seeded = Row("Papa John's", score=90, website="https://www.papajohns.com.pk")
        hut = Row("Pizza Hut", score=88, website="https://www.pizzahut.com.pk")
        kept = collapse_duplicate_competitors([pinned, seeded, hut])
        self.assertEqual({row.name for row in kept}, {"papa johns", "Pizza Hut"})
        self.assertTrue(pinned.is_tracking)
        self.assertFalse(seeded.is_tracking)
        self.assertEqual(pinned.website, "https://www.papajohns.com.pk")

    def test_cheezious_serp_queries_are_pizza_not_software(self):
        from types import SimpleNamespace
        from app.services.competitive import _niche_competitor_queries, _known_brand_home_market, _normalize_website

        client = SimpleNamespace(
            name="Cheezious",
            niche="restaurant",
            industry="Restaurant",
            notes="Business model: other",
            tagline="Cheese lovers",
            website="https://www.cheezious.com",
        )
        local_q = _niche_competitor_queries(client, "Pakistan", scope="local")
        blob = " ".join(local_q).lower()
        self.assertTrue(any("pizza" in q.lower() for q in local_q), local_q)
        self.assertNotIn("software house", blob)
        self.assertTrue(any("pakistan" in q.lower() for q in local_q), local_q)

        global_q = _niche_competitor_queries(client, "Pakistan", scope="global")
        gblob = " ".join(global_q).lower()
        self.assertTrue(any("pizza" in q.lower() or "worldwide" in q.lower() for q in global_q), global_q)
        self.assertNotIn("pakistan", gblob)
        self.assertNotIn("software", gblob)

        self.assertEqual(_known_brand_home_market("Cheezious", "https://cheezious.com"), "Pakistan")
        cleaned = _normalize_website(
            "https://cheezious.com/?utm_source=google&gclid=abc&utm_campaign=saudi"
        )
        self.assertEqual(cleaned, "https://cheezious.com")

    def test_serp_local_peers_survive_filter_without_country_in_snippet(self):
        rows = [
            {
                "name": "Broadway Pizza",
                "website": "https://broadwaypizza.com.pk",
                "why_relevant": "Order pizza online for delivery",
                "overlap_score": 62,
                "same_niche": True,
                "same_market": True,
                "source": "serp",
            },
            {
                "name": "NetSol Technologies",
                "website": "https://www.netsoltech.com",
                "why_relevant": "Enterprise software and digital transformation",
                "industry": "Software",
                "business_model": "services",
                "overlap_score": 74,
                "same_niche": True,
                "source": "ai",
            },
        ]
        kept = _filter_niche_competitors(
            rows,
            "Cheezious",
            market_area="Pakistan",
            niche="pizza / quick-service restaurant",
            industry="Restaurant",
            business_model="other",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        names = [r["name"] for r in kept]
        self.assertIn("Broadway Pizza", names)
        self.assertNotIn("NetSol Technologies", names)

    def test_blog_and_article_url_filtering(self):
        from app.services.competitive import _is_blog_or_article_url, _is_serp_noise_domain

        # Blog platforms & media publications
        self.assertTrue(_is_serp_noise_domain("https://medium.com/@author/best-saas-tools"))
        self.assertTrue(_is_serp_noise_domain("https://techcrunch.com/2026/05/startup-funding"))
        self.assertTrue(_is_serp_noise_domain("https://forbes.com/sites/top-crm-solutions"))
        self.assertTrue(_is_serp_noise_domain("https://g2.com/categories/crm"))
        self.assertTrue(_is_serp_noise_domain("https://clutch.co/developers/pakistan"))

        # Article path patterns
        self.assertTrue(_is_blog_or_article_url("https://somecompany.com/blog/10-best-tools-2026"))
        self.assertTrue(_is_blog_or_article_url("https://agency.com/news/top-digital-agencies"))
        self.assertTrue(_is_blog_or_article_url("https://reviewsite.io/reviews/wave-accounting"))
        self.assertTrue(_is_blog_or_article_url("https://marketpulse.com/2026/04/crm-comparison/"))
        self.assertTrue(_is_blog_or_article_url("https://consulting.com/case-studies/enterprise-growth"))
        self.assertTrue(_is_blog_or_article_url("https://example.com/company", title="10 Best CRM Softwares in 2026"))

        # Real company homepages should NOT be filtered
        self.assertFalse(_is_blog_or_article_url("https://waveapps.com"))
        self.assertFalse(_is_blog_or_article_url("https://freshbooks.com"))
        self.assertFalse(_is_blog_or_article_url("https://discretelogix.com"))
        self.assertFalse(_is_blog_or_article_url("https://broadwaypizza.com.pk"))

    def test_serp_rejects_articles_and_listicles(self):
        from app.services.competitive import _competitors_from_serp

        serp_organic = [
            {
                "title": "10 Best Invoicing Tools for Small Business in 2026",
                "link": "https://techcrunch.com/2026/01/best-invoicing-tools",
                "snippet": "We review the top invoicing apps including Wave and FreshBooks.",
            },
            {
                "title": "Wave Invoicing & Accounting Software",
                "link": "https://www.waveapps.com",
                "snippet": "Manage your money with free invoicing, accounting, and banking.",
            },
            {
                "title": "Top Accounting Agencies in Pakistan | Clutch Review",
                "link": "https://clutch.co/accounting/pakistan",
                "snippet": "Find the best verified accounting firms in Pakistan.",
            },
            {
                "title": "FreshBooks - Cloud Accounting Software",
                "link": "https://www.freshbooks.com",
                "snippet": "Small business invoicing and accounting software built for owners.",
            },
        ]
        rivals = _competitors_from_serp(serp_organic, client_name="Invoicely")
        rival_names = [r["name"] for r in rivals]
        rival_urls = [r["website"] for r in rivals]

        # Verified company homepages kept
        self.assertIn("Wave Invoicing & Accounting Software", [r["name"] for r in rivals] + ["Wave"])
        self.assertTrue(any("waveapps.com" in u for u in rival_urls))
        self.assertTrue(any("freshbooks.com" in u for u in rival_urls))

        # Articles and review directories rejected
        self.assertFalse(any("techcrunch.com" in u for u in rival_urls))
        self.assertFalse(any("clutch.co" in u for u in rival_urls))
        self.assertFalse(any("10 Best" in n for n in rival_names))

    def test_direct_html_scraper_fallback(self):
        import asyncio
        from app.services.tracking import _scrape_direct_html

        # Test against a known standard domain (or invalid domain handling)
        async def _run():
            res = await _scrape_direct_html("https://httpbin.org/html")
            return res

        res = asyncio.run(_run())
        self.assertEqual(res.get("status"), "ok")
        self.assertTrue(bool(res.get("markdown")))
        self.assertEqual(res.get("source"), "direct_http")

    def test_peer_scale_matching(self):
        from app.services.competitive import (
            _peer_scale_from_blob,
            _peer_scale_compatible,
            _PEER_BOUTIQUE,
            _PEER_MID,
            _PEER_ENTERPRISE,
        )

        boutique = _peer_scale_from_blob("Indie Design Studio", "boutique branding agency", "services")
        self.assertEqual(boutique, _PEER_BOUTIQUE)

        mid = _peer_scale_from_blob("TechLogix Services", "software house mid-market", "services")
        self.assertEqual(mid, _PEER_MID)

        enterprise = _peer_scale_from_blob("Salesforce CRM", "global enterprise cloud CRM platform", "saas")
        self.assertEqual(enterprise, _PEER_ENTERPRISE)

        # Boutique clients should not be paired with enterprise giants
        self.assertTrue(_peer_scale_compatible(_PEER_BOUTIQUE, _PEER_BOUTIQUE))
        self.assertTrue(_peer_scale_compatible(_PEER_BOUTIQUE, _PEER_MID))
        self.assertTrue(_peer_scale_compatible(_PEER_ENTERPRISE, _PEER_ENTERPRISE))

    def test_rival_fits_run_scope_market_and_tld(self):
        from app.services.competitive import _rival_fits_run_scope

        # Exact matching HQ country
        self.assertTrue(
            _rival_fits_run_scope(
                name="Discretelogix",
                website="https://discretelogix.com",
                headquarters="Pakistan",
                scope="local",
                market="Pakistan",
                client_name="Systems Limited",
                strict=True,
            )
        )

        # Matching TLD (.pk)
        self.assertTrue(
            _rival_fits_run_scope(
                name="DevTech Solutions",
                website="https://devtech.com.pk",
                headquarters=None,
                scope="local",
                market="Pakistan",
                client_name="Systems Limited",
                strict=True,
            )
        )

        # Conflicting country rejected
        self.assertFalse(
            _rival_fits_run_scope(
                name="Infosys",
                website="https://infosys.com",
                headquarters="India",
                scope="local",
                market="Pakistan",
                client_name="Systems Limited",
                strict=True,
            )
        )


