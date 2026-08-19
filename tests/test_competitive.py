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
        self.assertIn("Pizza Hut", names)
        self.assertIn("Domino's Pizza", names)
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
        self.assertIn("Pizza Hut", names)
        self.assertGreaterEqual(len(kept), 3)

    def test_papa_johns_aliases_are_the_same_rival(self):
        self.assertTrue(_rival_keys("papa johns") & _rival_keys("Papa John's", "https://www.papajohns.com.pk"))
        self.assertFalse(_rival_keys("Pizza Hut") & _rival_keys("Broadway Pizza"))

    def test_qsr_seeds_skip_papa_johns_alias(self):
        seeds = _seed_local_qsr_rivals("Pakistan", "Cheezious", already_have=["papa johns"], limit=8)
        names = [row["name"].lower() for row in seeds]
        self.assertFalse(any("papa" in name and "john" in name for name in names))
        self.assertIn("pizza hut", names)

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
