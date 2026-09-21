import pytest
import unittest
from unittest.mock import AsyncMock, patch

from app.services.competitive import (
    _rival_fits_run_scope,
    _incompatible_peer,
    _filter_niche_competitors,
    batch_evaluate_competitors,
    generate_search_strategy,
)


class TestUniversalCompetitorEngine(unittest.TestCase):
    """Test suite for Universal Multi-Dimensional Competitive Intelligence Engine."""

    def test_city_level_locality_gating_rejects_conflicting_cities(self):
        """Candidates explicitly based in another city (e.g. Karachi or Islamabad) must be rejected for a Lahore client."""
        # Candidate explicitly in Karachi
        fits_karachi = _rival_fits_run_scope(
            name="Pita Shawarma",
            website="https://pitashawarma.pk",
            headquarters="Karachi, Pakistan",
            description="The best authentic shawarma wraps in Karachi Clifton and SMCHS",
            why="Famous shawarma chain across Karachi",
            scope="local",
            market="Pakistan",
            city="Lahore",
            client_name="Sultan Shawarma",
            strict=True,
        )
        self.assertFalse(fits_karachi, "Competitor based in Karachi must be rejected for a Lahore client")

        # Candidate explicitly in Islamabad / Rawalpindi
        fits_isb = _rival_fits_run_scope(
            name="The Wrap Lab",
            website="https://wraplab.pk",
            headquarters="Islamabad, Pakistan",
            description="Premium artisan wraps and shawarma in F-7 and Beverly Centre Islamabad",
            why="Top wrap brand in Islamabad",
            scope="local",
            market="Pakistan",
            city="Lahore",
            client_name="Sultan Shawarma",
            strict=True,
        )
        self.assertFalse(fits_isb, "Competitor based in Islamabad must be rejected for a Lahore client")

        # Candidate verified in Lahore
        fits_lahore = _rival_fits_run_scope(
            name="Cock 'N' Bull",
            website="https://cocknbull.pk",
            headquarters="Lahore, Pakistan",
            description="Famous shawarma, rolls, and platters in Gulberg and DHA Lahore",
            why="Top shawarma and grill destination in Lahore",
            scope="local",
            market="Pakistan",
            city="Lahore",
            client_name="Sultan Shawarma",
            strict=True,
        )
        self.assertTrue(fits_lahore, "Competitor verified in Lahore must be accepted")

    def test_international_city_locality_gating(self):
        """City locality gating works universally across international markets (UAE, UK, US, etc.)."""
        # Dubai vs Abu Dhabi
        fits_ad = _rival_fits_run_scope(
            name="Al Fanar Restaurant",
            website="https://alfanarrestaurant.com",
            headquarters="Abu Dhabi, UAE",
            description="Traditional Emirati cuisine located in Yas Mall Abu Dhabi",
            why="Local Emirati restaurant in Abu Dhabi",
            scope="local",
            market="United Arab Emirates",
            city="Dubai",
            client_name="Arabian Tea House",
            strict=True,
        )
        self.assertFalse(fits_ad, "Abu Dhabi candidate must be rejected for Dubai client")

        # London vs Manchester
        fits_mcr = _rival_fits_run_scope(
            name="Northern Soul Grilled Cheese",
            website="https://northernsoulmcr.com",
            headquarters="Manchester, UK",
            description="Famous grilled cheese and comfort food in Manchester",
            why="Manchester comfort food",
            scope="local",
            market="United Kingdom",
            city="London",
            client_name="Kappacasein London",
            strict=True,
        )
        self.assertFalse(fits_mcr, "Manchester candidate must be rejected for London client")

    def test_intra_industry_sub_category_isolation(self):
        """Food sub-categories must isolate: shawarma specialists reject pizza chains and fine dining."""
        client_name = "Sultan Shawarma"
        client_industry = "Food & Hospitality"
        client_niche = "Shawarma & Middle Eastern Wraps"

        # 1. Cheezious (Pizza / Fast Food Burger Chain)
        incompat_cheezious = _incompatible_peer(
            client_industry=client_industry,
            client_niche=client_niche,
            rival_blob="Cheezious Cheesy pizza, loaded fries, and burgers with special discount deals and fast home delivery",
            client_name=client_name,
        )
        self.assertTrue(incompat_cheezious, "Cheezious (pizza/burgers) must be incompatible with shawarma client")

        # 2. Caprinos (Pure Pizza Chain)
        incompat_caprinos = _incompatible_peer(
            client_industry=client_industry,
            client_niche=client_niche,
            rival_blob="Caprinos Pizza Order delicious stone-baked pizzas, garlic bread, and dips online",
            client_name=client_name,
        )
        self.assertTrue(incompat_caprinos, "Caprinos (pure pizza) must be incompatible with shawarma client")

        # 3. Cosa Nostra (Fine Dining / Upscale Italian)
        incompat_cosa = _incompatible_peer(
            client_industry=client_industry,
            client_niche=client_niche,
            rival_blob="Cosa Nostra Authentic upscale Italian fine dining, artisanal pasta, wood-fired gourmet cuisine, and gelato bar",
            client_name=client_name,
        )
        self.assertTrue(incompat_cosa, "Cosa Nostra (fine dining Italian) must be incompatible with quick-service shawarma")

        # 4. True Shawarma Peer (Cock 'N' Bull / Syrian Shawarma)
        incompat_syrian = _incompatible_peer(
            client_industry=client_industry,
            client_niche=client_niche,
            rival_blob="Syrian Shawarma Authentic Damascus chicken and beef shawarma with garlic toum and pickles",
            client_name=client_name,
        )
        self.assertFalse(incompat_syrian, "True Syrian shawarma peer must NOT be incompatible")

    def test_universal_cross_industry_sub_category_isolation(self):
        """Sub-category isolation works across non-food industries (Fashion, Legal, SaaS)."""
        # Fashion: Eastern Pret vs Western Denim
        incompat_denim = _incompatible_peer(
            client_industry="Fashion & Apparel",
            client_niche="Eastern Pret & Festive Lawn",
            rival_blob="Levis Official Store Jeans, denim trucker jackets, graphic tees, and western casuals",
            client_name="Khaadi",
        )
        self.assertTrue(incompat_denim, "Western denim brand must be incompatible with Eastern Pret boutique")

        # Legal: Corporate M&A vs Criminal Defense
        incompat_crime = _incompatible_peer(
            client_industry="Legal Services",
            client_niche="Corporate M&A & Private Equity",
            rival_blob="Downtown Criminal Defense DUI, assault, misdemeanor bail bonds and traffic ticket defense attorney",
            client_name="Baker & Partners",
        )
        self.assertTrue(incompat_crime, "Criminal defense attorney must be incompatible with Corporate M&A firm")

    def test_client_referencing_clause_stripped_in_geo_gating(self):
        """Clauses referencing the client brand must not falsely credit the client's city to an out-of-city candidate."""
        # Wrap Lab is based in Islamabad, but the comparative AI explanation mentioned 'compete with Sultan Shawarma in Lahore'
        desc_with_client_leak = (
            "Wrap Lab’s extensive menu and online delivery directly compete with Sultan Shawarma’s "
            "core shawarma offerings and local market focus, threatening market share in Lahore and Islamabad."
        )
        fits_lahore = _rival_fits_run_scope(
            name="Wrap Lab",
            website="https://wraplab.pk",
            headquarters=None,
            description=desc_with_client_leak,
            why=desc_with_client_leak,
            scope="local",
            market="Pakistan",
            city="Lahore",
            client_name="Sultan Shawarma",
            strict=True,
        )
        self.assertFalse(fits_lahore, "Candidate must be rejected when city mention is part of a client-referencing clause")

    def test_self_identity_isolation_with_why_dangerous_token_leaks(self):
        """Food format and category checks must isolate candidate self-identity and not leak from why_dangerous."""
        from app.services.competitive import _food_format_from_blob, _food_format_compatible

        client_fmt = _food_format_from_blob("Sultan Shawarma", "Middle Eastern Shawarma & Wraps", "Food & Hospitality")
        self.assertEqual(client_fmt, "shawarma")

        # Caprinos self-identity
        caprinos_name = "Caprinos"
        caprinos_desc = "Caprinos is a pizza restaurant based in Lahore, Pakistan, specializing in classic and specialty pizzas"
        caprinos_web = "https://caprinos.com.pk"
        # Contaminated AI rationale: mentions 'shawarma' in explanation of why it is NOT a competitor
        caprinos_why = "While both operate in Lahore, Caprinos focuses exclusively on pizza, distinct from shawarma."

        # Self-identity only
        rival_fmt_self = _food_format_from_blob(caprinos_name, caprinos_desc, caprinos_web)
        self.assertEqual(rival_fmt_self, "pizza")
        self.assertFalse(_food_format_compatible(client_fmt, rival_fmt_self))

        # Incompatible peer check using self-identity blob
        incompat = _incompatible_peer(
            client_industry="Food & Hospitality",
            client_niche="Middle Eastern Shawarma & Wraps",
            rival_blob=f"{caprinos_name} {caprinos_desc}",
            client_name="Sultan Shawarma",
        )
        self.assertTrue(incompat, "Caprinos must be detected as an incompatible peer when using candidate self-identity")


class TestUniversalEnginePipelineAsync:
    """Async pipeline evaluation tests for Universal Competitive Intelligence Engine."""

    @pytest.mark.asyncio
    @patch("app.services.competitive.ai_service.structured_json", new_callable=AsyncMock)
    async def test_batch_evaluate_competitors_4d_vector(self, mock_ai):
        """Batch evaluation correctly disqualifies candidates failing the 4D vector criteria."""
        mock_db = AsyncMock()
        mock_ai.return_value = {
            "evaluations": [
                {
                    "id": 0,
                    "is_true_competitor": False,
                    "offering_substitute_score": 15,
                    "geographic_match": True,
                    "format_tier_match": 40,
                    "overlap_score": 25,
                    "disqualification_reason": "Different food category: pizza chain rather than shawarma specialist",
                },
                {
                    "id": 1,
                    "is_true_competitor": False,
                    "offering_substitute_score": 85,
                    "geographic_match": False,
                    "format_tier_match": 80,
                    "overlap_score": 35,
                    "disqualification_reason": "Geographic mismatch: candidate operates only in Karachi, not Lahore",
                },
                {
                    "id": 2,
                    "is_true_competitor": True,
                    "offering_substitute_score": 95,
                    "geographic_match": True,
                    "format_tier_match": 90,
                    "overlap_score": 92,
                    "why_relevant": "Direct quick-service shawarma and roll competitor in Lahore",
                    "disqualification_reason": "",
                },
            ]
        }

        client_profile = {
            "name": "Sultan Shawarma",
            "industry": "Food & Hospitality",
            "niche": "Shawarma & Middle Eastern Wraps",
            "primary_offering": "Chicken & Beef Shawarma",
            "customer_type": "Dine-in and Delivery Food Lovers",
        }

        candidates = [
            {"name": "Cheezious", "website": "https://cheezious.com", "snippet": "Cheesy pizza and burgers in Lahore"},
            {"name": "Pita Shawarma", "website": "https://pitashawarma.pk", "snippet": "Best shawarma in Karachi"},
            {"name": "Cock 'N' Bull", "website": "https://cocknbull.pk", "snippet": "Best shawarma and rolls in Gulberg Lahore"},
        ]

        evaluated = await batch_evaluate_competitors(
            mock_db,
            "agency-1",
            client_profile,
            candidates,
            scope="local",
            country="Pakistan",
            city="Lahore",
        )

        names = [c["name"] for c in evaluated]
        assert "Cheezious" not in names, "Cheezious must be disqualified"
        assert "Pita Shawarma" not in names, "Pita Shawarma (Karachi) must be disqualified"
        assert "Cock 'N' Bull" in names, "Cock 'N' Bull (Lahore Shawarma) must be retained"
        assert evaluated[0]["overlap_score"] == 92.0

    def test_sultan_shawarma_full_filtering_case_study(self):
        """End-to-end simulation of the Sultan Shawarma in Lahore candidate set."""
        raw_candidates = [
            # Wrong category (Pizza / Burgers)
            {
                "name": "Cheezious",
                "website": "https://cheezious.com",
                "why_relevant": "Popular fast food chain known for pizza, loaded fries, and fast delivery deals",
                "overlap_score": 85.0,
                "headquarters": "Lahore, Pakistan",
            },
            # Wrong category (Pizza chain)
            {
                "name": "Caprinos Pizza",
                "website": "https://caprinospizza.pk",
                "why_relevant": "Stone-baked pizza delivery and takeaway brand",
                "overlap_score": 80.0,
                "headquarters": "Lahore, Pakistan",
            },
            # Wrong category & format tier (Fine dining Italian)
            {
                "name": "Cosa Nostra",
                "website": "https://cosanostra.pk",
                "why_relevant": "Fine dining restaurant offering gourmet Italian pasta, artisan pizza, and gelato",
                "overlap_score": 75.0,
                "headquarters": "Lahore, Pakistan",
            },
            # Wrong city (Karachi only)
            {
                "name": "Pita",
                "website": "https://pitapakistan.com",
                "why_relevant": "Authentic shawarma and gyros in Karachi",
                "overlap_score": 90.0,
                "headquarters": "Karachi, Pakistan",
            },
            # Wrong city (Islamabad only)
            {
                "name": "The Wrap Lab",
                "website": "https://wraplab.pk",
                "why_relevant": "Artisan wraps and shawarma in Beverly Centre Islamabad",
                "overlap_score": 88.0,
                "headquarters": "Islamabad, Pakistan",
            },
            # Legitimate Lahore Shawarma Peers
            {
                "name": "Cock 'N' Bull",
                "website": "https://cocknbull.pk",
                "why_relevant": "Direct competitor specializing in famous chicken shawarma and paratha rolls in Lahore",
                "overlap_score": 92.0,
                "headquarters": "Lahore, Pakistan",
            },
            {
                "name": "Syrian Shawarma",
                "website": "https://syrianshawarma.pk",
                "why_relevant": "Authentic Syrian shawarma wraps, platters, and garlic sauce in Lahore",
                "overlap_score": 94.0,
                "headquarters": "Lahore, Pakistan",
            },
            {
                "name": "Shawarma Lounge",
                "website": "https://shawarmalounge.pk",
                "why_relevant": "Specialty shawarma and grill bar in DHA Lahore",
                "overlap_score": 89.0,
                "headquarters": "Lahore, Pakistan",
            },
        ]

        sanitized = _filter_niche_competitors(
            raw_candidates,
            client_name="Sultan Shawarma",
            market_area="Pakistan",
            city="Lahore",
            niche="Shawarma & Middle Eastern Wraps",
            industry="Food & Hospitality",
            business_model="Local Food Business",
            min_overlap=50.0,
            limit=5,
            require_local_market=True,
        )

        sanitized_names = [c["name"] for c in sanitized]

        # Ensure all false positives were rejected
        assert "Cheezious" not in sanitized_names
        assert "Caprinos Pizza" not in sanitized_names
        assert "Cosa Nostra" not in sanitized_names
        assert "Pita" not in sanitized_names
        assert "The Wrap Lab" not in sanitized_names

        # Ensure true Lahore shawarma peers were kept
        assert "Cock 'N' Bull" in sanitized_names
        assert "Syrian Shawarma" in sanitized_names
        assert "Shawarma Lounge" in sanitized_names
        assert len(sanitized_names) == 3
