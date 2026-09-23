import unittest
from unittest.mock import AsyncMock, patch

from app.schemas import NicheDetectionResponse
from app.services.niche_detection import (
    CANONICAL_INDUSTRIES,
    GENERIC_NICHE_REJECTS,
    _clean_site_markdown,
    _extract_page_meta_hints,
    _heuristic_niche_fallback,
    detect_brand_niche,
)


class NicheDetectionUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_clean_site_markdown(self):
        raw = "# Welcome to Sultan Shawarma\n\n![Logo](https://img.com/1.png)\n[Menu](https://site.com/menu)\n<p>Authentic wraps</p>"
        cleaned = _clean_site_markdown(raw)
        self.assertIn("Welcome to Sultan Shawarma", cleaned)
        self.assertIn("Menu", cleaned)
        self.assertNotIn("![Logo]", cleaned)
        self.assertNotIn("<p>", cleaned)

    def test_extract_page_meta_hints(self):
        html_doc = (
            "<html><head>"
            "<title>Sultan Shawarma - Authentic Arabic Wraps in Lahore</title>"
            '<meta name="description" content="Best shawarma wraps and platters made fresh to order.">'
            '<meta property="og:description" content="Juicy shawarma and garlic sauce.">'
            "</head><body></body></html>"
        )
        hints = _extract_page_meta_hints(html_doc)
        self.assertEqual(hints["page_title"], "Sultan Shawarma - Authentic Arabic Wraps in Lahore")
        self.assertEqual(hints["meta_description"], "Best shawarma wraps and platters made fresh to order.")
        self.assertEqual(hints["og_description"], "Juicy shawarma and garlic sauce.")

    def test_heuristic_fallback_shawarma(self):
        res = _heuristic_niche_fallback("Sultan Shawarma", "Authentic shawarma wraps in Lahore")
        self.assertEqual(res.industry, "Food & Hospitality")
        self.assertIn("Shawarma", res.niche)
        self.assertGreaterEqual(res.confidence, 0.90)

    def test_heuristic_fallback_apparel(self):
        res = _heuristic_niche_fallback("Khaadi", "Women's ready to wear pret and unstitched lawn")
        self.assertEqual(res.industry, "Apparel & Fashion")
        self.assertIn("Fashion", res.niche)
        self.assertGreaterEqual(res.confidence, 0.90)

    def test_heuristic_fallback_saas(self):
        res = _heuristic_niche_fallback("Databiqs", "Enterprise cloud AI and data analytics platform")
        self.assertEqual(res.industry, "Software & Technology")
        self.assertIn("SaaS", res.niche)

    def test_heuristic_fallback_dental(self):
        res = _heuristic_niche_fallback("Bright Smiles", "Cosmetic dental clinic and teeth whitening")
        self.assertEqual(res.industry, "Healthcare & Medical")
        self.assertIn("Dentistry", res.niche)

    def test_generic_rejects_list(self):
        self.assertIn("food", GENERIC_NICHE_REJECTS)
        self.assertIn("restaurant", GENERIC_NICHE_REJECTS)
        self.assertIn("retail", GENERIC_NICHE_REJECTS)
        self.assertIn("software", GENERIC_NICHE_REJECTS)

    @patch("app.services.niche_detection.scrape_website", new_callable=AsyncMock)
    @patch("app.services.niche_detection.ai_service.structured_json", new_callable=AsyncMock)
    async def test_detect_brand_niche_success(self, mock_ai, mock_scrape):
        mock_scrape.return_value = {
            "markdown": "# Sultan Shawarma\nServing authentic Middle Eastern chicken and beef shawarmas with garlic sauce in Lahore.",
            "html": "<title>Sultan Shawarma Lahore</title>",
        }
        mock_ai.return_value = {
            "industry": "Food & Hospitality",
            "niche": "Shawarma & Middle Eastern Wraps",
            "primary_offering": "Authentic Middle Eastern shawarmas and fresh wraps.",
            "customer_type": "General Consumer",
            "business_model": "restaurant",
            "confidence": 0.96,
            "evidence": "Homepage states: Serving authentic Middle Eastern chicken and beef shawarmas.",
            "suggested_alternatives": ["Middle Eastern Fast Casual", "Quick Service Wraps"],
        }

        db = AsyncMock()
        res = await detect_brand_niche(
            db,
            "agency-123",
            name="Sultan Shawarma",
            website="https://sultanshawarma.com",
            city="Lahore",
            country="Pakistan",
        )

        self.assertIsInstance(res, NicheDetectionResponse)
        self.assertEqual(res.industry, "Food & Hospitality")
        self.assertEqual(res.niche, "Shawarma & Middle Eastern Wraps")
        self.assertEqual(res.confidence, 0.96)
        self.assertIn("Middle Eastern", res.primary_offering)

    @patch("app.services.niche_detection.scrape_website", new_callable=AsyncMock)
    @patch("app.services.niche_detection.ai_service.structured_json", new_callable=AsyncMock)
    async def test_detect_brand_niche_rejects_generic_niche(self, mock_ai, mock_scrape):
        mock_scrape.return_value = {"markdown": "Shawarma shop menu with wraps and platters."}
        # Model attempts to return a generic "Food"
        mock_ai.return_value = {
            "industry": "Food & Hospitality",
            "niche": "Food",
            "primary_offering": "Food items",
            "customer_type": "General Consumer",
            "business_model": "restaurant",
            "confidence": 0.5,
        }

        db = AsyncMock()
        res = await detect_brand_niche(
            db,
            "agency-123",
            name="Sultan Shawarma",
            website="https://sultanshawarma.com",
        )

        # Must reject generic "Food" and fall back to specific Shawarma niche
        self.assertNotEqual(res.niche.lower(), "food")
        self.assertIn("Shawarma", res.niche)
