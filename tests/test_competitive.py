import unittest

from app.services.competitive import (
    _filter_niche_competitors,
    _incompatible_peer,
    _looks_like_food_client,
    _looks_like_software_peer_client,
    _rival_keys,
    collapse_duplicate_competitors,
)


class LocalSeedTests(unittest.TestCase):
    def test_filter_skips_client_domain(self):
        candidates = [
            {"name": "Systems Limited", "website": "https://www.systemsltd.com", "industry": "Software", "headquarters_country": "Pakistan", "overlap_score": 85.0},
            {"name": "Devsinc", "website": "https://www.devsinc.com", "industry": "Software", "headquarters_country": "Pakistan", "overlap_score": 85.0},
        ]
        kept = _filter_niche_competitors(
            candidates,
            "Systems Limited",
            market_area="Pakistan",
            niche="IT consulting",
            industry="Software",
            business_model="services",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        names = [row["name"] for row in kept]
        self.assertNotIn("Systems Limited", names)
        self.assertIn("Devsinc", names)

    def test_food_brand_is_not_software_peer(self):
        self.assertTrue(_looks_like_food_client("Cheezious", "fast food", "pizza delivery"))
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
        software_candidates = [
            {"name": "Systems Limited", "industry": "Software", "business_model": "services", "headquarters_country": "Pakistan", "why_relevant": "IT consultancy", "overlap_score": 85.0},
            {"name": "NetSol", "industry": "Software", "business_model": "services", "headquarters_country": "Pakistan", "why_relevant": "Software house", "overlap_score": 80.0},
        ]
        kept = _filter_niche_competitors(
            software_candidates,
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

    def test_pakistan_qsr_candidates_for_cheezious(self):
        candidates = [
            {"name": "Broadway Pizza", "website": "https://broadwaypizza.com.pk", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "leading pizza competitor in Pakistan", "overlap_score": 88.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "Pizza Max", "website": "https://pizzamax.com.pk", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "local pizza delivery chain", "overlap_score": 85.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "14th Street Pizza", "website": "https://14thstreetpizza.com", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "pizza restaurant in Pakistan", "overlap_score": 84.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "Fork N Knives Pizza", "website": "https://forknknives.com", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "local pizza outlet chain", "overlap_score": 82.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "Systems Limited", "website": "https://systemsltd.com", "industry": "Software", "why_relevant": "IT consulting", "overlap_score": 80.0, "headquarters_country": "Pakistan"},
        ]
        kept = _filter_niche_competitors(
            candidates,
            "Cheezious",
            market_area="Pakistan",
            niche="pizza",
            industry="fast food",
            business_model="other",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        names = [row["name"] for row in kept]
        self.assertIn("Broadway Pizza", names)
        self.assertIn("Pizza Max", names)
        self.assertNotIn("Systems Limited", names)
        self.assertGreaterEqual(len(kept), 4)

    def test_misprofiled_cheezious_still_keeps_qsr_rivals(self):
        candidates = [
            {"name": "Broadway Pizza", "website": "https://broadwaypizza.com.pk", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "pizza restaurant in Pakistan", "overlap_score": 88.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "Pizza Max", "website": "https://pizzamax.com.pk", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "pizza brand in Pakistan", "overlap_score": 85.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
            {"name": "14th Street Pizza", "website": "https://14thstreetpizza.com", "industry": "Restaurant", "food_format": "pizza", "why_relevant": "pizza chain in Pakistan", "overlap_score": 84.0, "headquarters_country": "Pakistan", "same_market": True, "same_niche": True},
        ]
        kept = _filter_niche_competitors(
            candidates,
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

    def test_pakistan_shawarma_for_sultan(self):
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
        import httpx
        from unittest.mock import patch, AsyncMock
        from app.services.tracking import _scrape_direct_html

        class MockResp:
            status_code = 200
            url = "https://example.com"
            text = "<html><head><title>Test Title</title></head><body><h1>Sample Direct HTML</h1><p>Content for testing direct fallback.</p></body></html>"

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MockResp()
            res = asyncio.run(_scrape_direct_html("https://example.com"))
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

    def test_beauty_brand_detection(self):
        from app.services.competitive import (
            _client_peer_hint,
            _looks_like_beauty_client,
        )

        self.assertTrue(_looks_like_beauty_client("Hifsa Khan", "Beauty & Personal Care", "Beauty salon & makeup studio"))
        self.assertTrue(_looks_like_beauty_client("Depilex Beauty Clinic"))
        self.assertTrue(_looks_like_beauty_client("Nabila Salon", "hair styling"))
        self.assertFalse(_looks_like_beauty_client("Systems Limited", "Software", "IT services"))
        self.assertFalse(_looks_like_beauty_client("Cheezious", "Fast Food", "Pizza"))

        self.assertEqual(
            _client_peer_hint("Hifsa Khan", "Beauty & Personal Care", "Beauty salon"),
            "beauty-salon / makeup-studio rivals",
        )

    def test_short_beauty_brand_names_not_fake(self):
        from app.services.competitive import _is_generic_or_fake_rival_name

        self.assertFalse(_is_generic_or_fake_rival_name("Sabs"))
        self.assertFalse(_is_generic_or_fake_rival_name("MAC"))
        self.assertFalse(_is_generic_or_fake_rival_name("NARS"))
        self.assertFalse(_is_generic_or_fake_rival_name("Zara"))
        self.assertFalse(_is_generic_or_fake_rival_name("Huda"))
        self.assertFalse(_is_generic_or_fake_rival_name("Depilex"))
        self.assertTrue(_is_generic_or_fake_rival_name("TechCorp"))
        self.assertTrue(_is_generic_or_fake_rival_name("Soft Solutions"))

    def test_beauty_incompatible_peer_and_filtering(self):
        from app.services.competitive import _filter_niche_competitors, _incompatible_peer

        # Beauty client vs beauty rivals: compatible
        self.assertFalse(
            _incompatible_peer(
                client_model="services",
                client_industry="Beauty & Personal Care",
                client_niche="Beauty salon & makeup studio",
                rival_model="services",
                rival_industry="Beauty & Personal Care",
                rival_blob="Depilex Beauty Clinic aesthetic skincare and bridal salon",
                client_name="Hifsa Khan",
            )
        )
        self.assertFalse(
            _incompatible_peer(
                client_model="services",
                client_industry="Beauty & Personal Care",
                client_niche="Beauty salon & makeup studio",
                rival_model="services",
                rival_industry="Beauty & Personal Care",
                rival_blob="Kashee's Beauty Parlour bridal makeup and beauty products shop",
                client_name="Hifsa Khan",
            )
        )

        # Beauty client vs software house / restaurant: incompatible
        self.assertTrue(
            _incompatible_peer(
                client_model="services",
                client_industry="Beauty & Personal Care",
                client_niche="Beauty salon & makeup studio",
                rival_model="services",
                rival_industry="Software",
                rival_blob="Systems Limited commercial software house / digital product firm",
                client_name="Hifsa Khan",
            )
        )
        self.assertTrue(
            _incompatible_peer(
                client_model="services",
                client_industry="Beauty & Personal Care",
                client_niche="Beauty salon & makeup studio",
                rival_model="other",
                rival_industry="Fast Food",
                rival_blob="Cheezious pizza and fast food chain",
                client_name="Hifsa Khan",
            )
        )

        # Filter keeps beauty competitors in Pakistan
        raw_candidates = [
            {
                "name": "Depilex Beauty Clinic",
                "website": "https://depilex.com",
                "why_relevant": "Leading beauty salon and aesthetic clinic in Pakistan",
                "industry": "Beauty & Personal Care",
                "business_model": "services",
                "overlap_score": 78,
                "same_niche": True,
                "source": "ai",
            },
            {
                "name": "Kashee's Beauty Parlour",
                "website": "https://kashees.com",
                "why_relevant": "Top bridal makeup salon and aesthetic studio in Pakistan",
                "industry": "Beauty & Personal Care",
                "business_model": "services",
                "overlap_score": 76,
                "same_niche": True,
                "source": "ai",
            },
            {
                "name": "Sabs",
                "website": "https://sabs.com.pk",
                "why_relevant": "Famous luxury beauty salon chain across Pakistan",
                "industry": "Beauty & Personal Care",
                "business_model": "services",
                "overlap_score": 75,
                "same_niche": True,
                "source": "ai",
            },
            {
                "name": "Systems Limited",
                "website": "https://systemsltd.com",
                "why_relevant": "Enterprise software development company in Pakistan",
                "industry": "Software",
                "business_model": "services",
                "overlap_score": 80,
                "same_niche": True,
                "source": "ai",
            },
        ]
        kept = _filter_niche_competitors(
            raw_candidates,
            "Hifsa Khan",
            market_area="Pakistan",
            niche="Beauty salon & makeup studio",
            industry="Beauty & Personal Care",
            business_model="services",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )
        kept_names = [r["name"] for r in kept]
        self.assertIn("Depilex Beauty Clinic", kept_names)
        self.assertIn("Kashee's Beauty Parlour", kept_names)
        self.assertIn("Sabs", kept_names)
        self.assertNotIn("Systems Limited", kept_names)
        self.assertEqual(len(kept), 3)

    def test_beauty_serp_queries(self):
        from types import SimpleNamespace
        from app.services.competitive import _niche_competitor_queries

        client = SimpleNamespace(
            name="Hifsa Khan",
            niche="Beauty salon & makeup studio",
            industry="Beauty & Personal Care",
            notes="Business model: services",
            tagline="Bridal & Beauty",
            website="https://hifsakhan.com",
        )
        local_q = _niche_competitor_queries(client, "Pakistan", scope="local")
        blob = " ".join(local_q).lower()
        self.assertTrue(any("beauty" in q.lower() or "salon" in q.lower() for q in local_q), local_q)
        self.assertNotIn("software house", blob)
        self.assertNotIn("pizza", blob)

    def test_semantic_hybrid_data_ai_classification(self):
        from types import SimpleNamespace
        from app.services.competitive import (
            _detect_industry_category,
            _incompatible_peer,
            _niche_competitor_queries,
        )

        # 1. Semantic hybrid category detection
        cat_databiqs = _detect_industry_category(
            "Databiqs", "Enterprise AI solutions and automation", "Software Agency", "services"
        )
        self.assertEqual(cat_databiqs, "data_ai")

        cat_systems = _detect_industry_category(
            "Systems Limited", "IT services and custom software", "Software House", "services"
        )
        self.assertEqual(cat_systems, "software")

        # 2. Incompatible peer filtering: reject generic software giants for AI consultancies
        self.assertTrue(
            _incompatible_peer(
                client_model="services",
                client_industry="Software Agency",
                client_niche="Enterprise AI solutions and automation",
                rival_model="services",
                rival_industry="Software",
                rival_blob="10Pearls global custom software engineering outsourcing",
                client_name="Databiqs",
            )
        )
        self.assertTrue(
            _incompatible_peer(
                client_model="services",
                client_industry="Software Agency",
                client_niche="Enterprise AI solutions and automation",
                rival_model="services",
                rival_industry="Software",
                rival_blob="Arbisoft legacy custom software development body shop",
                client_name="Databiqs",
            )
        )

        # 3. Compatible peer: keep direct AI peers
        self.assertFalse(
            _incompatible_peer(
                client_model="services",
                client_industry="Software Agency",
                client_niche="Enterprise AI solutions and automation",
                rival_model="services",
                rival_industry="AI Solutions",
                rival_blob="Astraea AI enterprise machine learning and automation consultancy",
                client_name="Databiqs",
            )
        )

        # 4. Search query generation: AI-focused queries generated instead of generic software house listicles
        client = SimpleNamespace(
            name="Databiqs",
            niche="Enterprise AI solutions and automation",
            industry="Software Agency",
            notes="Business model: services",
            tagline="Enterprise AI & Automation",
            website="https://databiqs.com",
        )
        queries = _niche_competitor_queries(client, "Pakistan", scope="local")
        blob = " ".join(queries).lower()
        self.assertTrue(any("ai" in q.lower() or "data" in q.lower() or "analytics" in q.lower() for q in queries), queries)
        self.assertNotIn("top software houses", blob)

        # 5. Layer 1 persistence in client.notes
        from app.services.competitive import (
            _industry_category_from_client,
            _set_industry_category,
        )
        test_client = SimpleNamespace(name="Databiqs", notes=None, niche="Enterprise AI", industry="Software Agency")
        _set_industry_category(test_client, "data_ai")
        self.assertIn("Industry category: data_ai", test_client.notes)
        self.assertEqual(_industry_category_from_client(test_client), "data_ai")
        self.assertEqual(_detect_industry_category(test_client), "data_ai")

    def test_local_market_and_peer_verification(self):
        from app.services.competitive import _filter_niche_competitors, _rival_fits_run_scope

        candidates = [
            {
                "name": "Cubix",
                "website": "https://www.cubix.co",
                "industry": "Software development",
                "business_model": "services",
                "headquarters_country": "Unknown",
                "why_relevant": "Cubix is a software development firm that builds mobile apps, web products, and games.",
                "threat_level": "medium",
                "overlap_score": 75.0,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
            },
            {
                "name": "DataSoft Systems",
                "website": "https://www.datasoft-global.com",
                "industry": "IT consulting",
                "business_model": "services",
                "headquarters_country": "Not disclosed on the website",
                "why_relevant": "DataSoft Systems provides custom software development, enterprise integration worldwide.",
                "threat_level": "medium",
                "overlap_score": 75.0,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
            },
            {
                "name": "Techlogix",
                "website": "https://www.techlogix.com",
                "industry": "IT consulting & services",
                "business_model": "services",
                "headquarters_country": "Pakistan",
                "why_relevant": "Based in Karachi, Pakistan, Techlogix runs a dedicated Data Analytics and AI practice delivering BI dashboards.",
                "threat_level": "high",
                "overlap_score": 88.0,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
            },
            {
                "name": "RedBuffer",
                "website": "https://redbuffer.net",
                "industry": "AI Solutions",
                "business_model": "services",
                "headquarters_country": "Pakistan",
                "why_relevant": "RedBuffer specializes in enterprise AI, machine learning and automation platforms for clients in Pakistan.",
                "threat_level": "high",
                "overlap_score": 90.0,
                "same_niche": True,
                "same_market": True,
                "is_global_platform": False,
            }
        ]

        kept = _filter_niche_competitors(
            candidates,
            "Databiqs",
            market_area="Pakistan",
            niche="Enterprise AI automation and chatbot solutions",
            industry="Software agency",
            business_model="services",
            min_overlap=55.0,
            limit=10,
            require_local_market=True,
        )

        kept_names = [c["name"] for c in kept]
        self.assertNotIn("Cubix", kept_names)
        self.assertNotIn("DataSoft Systems", kept_names)
        self.assertIn("Techlogix", kept_names)
        self.assertIn("RedBuffer", kept_names)

        # Also test _rival_fits_run_scope directly on dead domain / unverified candidate
        fits = _rival_fits_run_scope(
            name="DataSoft Systems",
            website="https://www.datasoft-global.com",
            headquarters="Not disclosed on the website",
            description="Global software services",
            why="Global software services",
            scope="local",
            market="Pakistan",
            client_name="Databiqs",
            strict=True,
        )
        self.assertFalse(fits, "DataSoft Systems with unknown HQ and no local proof should not fit local Pakistan scope")




