import unittest
from datetime import datetime, timezone

from app.models import Agency, PlanType, StripeEvent
from app.services import billing


def agency(plan: PlanType = PlanType.agency) -> Agency:
    return Agency(
        id="agency-test",
        name="Test Agency",
        slug="test-agency",
        plan=plan,
        billing_status="incomplete",
        included_clients=10 if plan == PlanType.agency else 1,
        client_pack_count=0,
        reports_used=4,
        scrape_units_used=12,
        reports_quota=40 if plan == PlanType.agency else 10,
        scrape_quota=5000 if plan == PlanType.agency else 500,
        budget_remaining_cents=0,
        byok_discount_percent=0,
        cancel_at_period_end=False,
    )


class BillingCatalogTests(unittest.TestCase):
    def test_current_plan_prices_and_allowances(self):
        catalog = billing.billing_catalog()
        plans = {plan.id: plan for plan in catalog.plans}
        self.assertEqual(plans["agency"].price_cents, 45000)
        self.assertEqual(plans["agency"].included_clients, 10)
        self.assertEqual(plans["creator"].price_cents, 9900)
        self.assertEqual(plans["creator"].included_clients, 1)
        self.assertEqual(catalog.pack.price_cents, 4900)
        self.assertEqual(catalog.pack.extra_reports, 8)
        self.assertEqual(catalog.pack.extra_scrapes, 800)
        self.assertEqual(catalog.scrape_pack.price_cents, 500)
        self.assertEqual(catalog.scrape_pack.units, 100)
        self.assertIn(200, catalog.scrape_pack.options)
        self.assertIsNotNone(catalog.payg)
        self.assertEqual(catalog.payg.client_cents, 900)
        self.assertEqual(catalog.payg.intel_run_cents, 300)
        self.assertEqual(catalog.payg.report_cents, 200)
        self.assertEqual(catalog.payg.scrape_unit_cents, 5)

    def test_agency_payg_bills_actual_usage_not_fixed_packs(self):
        row = agency()
        row.billing_model = "payg"
        row.stripe_subscription_id = "sub_payg"
        row.reports_used = 2
        row.scrape_units_used = 40
        budget = billing.compute_budget(row, active_clients=2, intel_runs_used=2)
        self.assertEqual(budget.billing_model, "payg")
        self.assertTrue(budget.payg_available)
        self.assertEqual(budget.max_clients, billing.PAYG_CLIENT_CAP)
        self.assertEqual(budget.plan_name, "PAYG")
        # 2 clients × $9 + 2 intel × $3 + 2 reports × $2 + 40 scrapes × $0.05 = $30.00
        self.assertEqual(budget.estimated_monthly_cents, 3000)
        keys = [line.key for line in budget.usage_lines]
        self.assertEqual(keys, ["clients", "intel_runs", "reports", "scrapes"])
        self.assertEqual(budget.usage_lines[0].amount_cents, 1800)
        self.assertEqual(budget.usage_lines[1].amount_cents, 600)
        self.assertEqual(budget.usage_lines[2].amount_cents, 400)
        self.assertEqual(budget.usage_lines[3].amount_cents, 200)

    def test_extra_scrape_lots_round_up_from_usage(self):
        self.assertEqual(billing.extra_scrape_lots_needed(0, 800), 0)
        self.assertEqual(billing.extra_scrape_lots_needed(800, 800), 0)
        self.assertEqual(billing.extra_scrape_lots_needed(801, 800), 1)
        self.assertEqual(billing.extra_scrape_lots_needed(1650, 1600), 1)
        self.assertEqual(billing.extra_scrape_lots_needed(1800, 1600), 2)

    def test_payg_bill_scales_with_intel_reports_and_scrapes(self):
        row = agency()
        row.billing_model = "payg"
        row.stripe_subscription_id = "sub_payg"
        row.client_pack_count = 2
        row.scrape_units_used = 1650
        budget = billing.compute_budget(row, active_clients=2, intel_runs_used=3)
        self.assertEqual(budget.intel_runs_used, 3)
        self.assertEqual(budget.reports_used, 4)
        # 2×900 + 3×300 + 4×200 + 1650×5 = 11750
        self.assertEqual(budget.estimated_monthly_cents, 11750)
        self.assertEqual(budget.scrape_overage_lots, 0)
        self.assertEqual(budget.client_pack_count, 2)

    def test_unpaid_payg_mark_does_not_block_plan_seats(self):
        row = agency()
        row.billing_model = "payg"
        row.included_clients = 0
        row.reports_quota = 0
        self.assertFalse(billing.is_payg(row))
        billing.normalize_billing_model(row)
        self.assertEqual(row.billing_model, "plan")
        self.assertEqual(row.included_clients, 10)
        self.assertEqual(row.reports_quota, 40)

    def test_individual_workspace_cannot_use_payg(self):
        from app.models import WorkspaceMode

        row = agency(PlanType.creator)
        row.workspace_mode = WorkspaceMode.creator
        row.billing_model = "payg"
        self.assertFalse(billing.is_payg(row))
        self.assertFalse(billing.compute_budget(row, active_clients=1).payg_available)

    def test_workspace_budget_catalog_is_plan_scoped(self):
        agency_budget = billing.compute_budget(agency(PlanType.agency), active_clients=1)
        self.assertEqual([plan.id for plan in agency_budget.catalog.plans], ["agency"])
        creator_budget = billing.compute_budget(agency(PlanType.creator), active_clients=1)
        self.assertEqual([plan.id for plan in creator_budget.catalog.plans], ["creator"])

    def test_compute_budget_is_read_only(self):
        row = agency()
        row.reports_quota = 91
        result = billing.compute_budget(row, active_clients=2)
        self.assertEqual(row.reports_quota, 91)
        self.assertEqual(result.reports_quota, 91)
        self.assertEqual(result.estimated_monthly_cents, 45000)

    def test_byok_discount_lowers_billing_estimate(self):
        row = agency()
        row.byok_discount_percent = 10
        budget = billing.compute_budget(row, active_clients=1)
        self.assertEqual(budget.list_price_cents, 45000)
        self.assertEqual(budget.estimated_monthly_cents, 40500)
        self.assertEqual(budget.byok_discount_percent, 10)
        self.assertEqual(billing.discounted_cents(3000, 10), 2700)

    def test_stripe_event_id_is_unique_for_webhook_replay(self):
        self.assertTrue(StripeEvent.__table__.c.event_id.unique)


class CheckoutSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_stripe_never_grants_local_packs(self):
        row = agency()
        old_secret = billing.settings.stripe_secret_key
        billing.settings.stripe_secret_key = ""
        try:
            with self.assertRaisesRegex(ValueError, "Stripe is not configured"):
                await billing.create_checkout_session(
                    row,
                    add_client_packs=2,
                    success_url="http://localhost:3000/billing?session_id={CHECKOUT_SESSION_ID}",
                    cancel_url="http://localhost:3000/billing?checkout=canceled",
                    request_id="request-123",
                )
            self.assertEqual(row.client_pack_count, 0)
            self.assertNotEqual(row.billing_model, "payg")
        finally:
            billing.settings.stripe_secret_key = old_secret


class SubscriptionSyncTests(unittest.TestCase):
    def setUp(self):
        self.old_ids = (
            billing.settings.stripe_agency_price_id,
            billing.settings.stripe_individual_price_id,
            billing.settings.stripe_client_pack_price_id,
            billing.settings.stripe_scrape_pack_price_id,
            billing.settings.stripe_payg_price_id,
        )
        billing.settings.stripe_agency_price_id = "price_agency"
        billing.settings.stripe_individual_price_id = "price_individual"
        billing.settings.stripe_client_pack_price_id = "price_pack"
        billing.settings.stripe_scrape_pack_price_id = "price_scrapes"
        billing.settings.stripe_payg_price_id = "price_payg"

    def tearDown(self):
        (
            billing.settings.stripe_agency_price_id,
            billing.settings.stripe_individual_price_id,
            billing.settings.stripe_client_pack_price_id,
            billing.settings.stripe_scrape_pack_price_id,
            billing.settings.stripe_payg_price_id,
        ) = self.old_ids

    def test_subscription_items_set_plan_and_pack_entitlements(self):
        row = agency()
        subscription = {
            "id": "sub_123",
            "status": "active",
            "current_period_start": 1_700_000_000,
            "current_period_end": 1_702_592_000,
            "cancel_at_period_end": False,
            "items": {
                "data": [
                    {"id": "si_base", "price": {"id": "price_agency"}, "quantity": 1},
                    {"id": "si_pack", "price": {"id": "price_pack"}, "quantity": 3},
                ]
            },
        }
        billing.apply_subscription(row, subscription)
        self.assertEqual(row.stripe_subscription_id, "sub_123")
        self.assertEqual(row.stripe_base_item_id, "si_base")
        self.assertEqual(row.stripe_pack_item_id, "si_pack")
        self.assertEqual(row.client_pack_count, 3)
        self.assertEqual(row.reports_quota, 64)
        self.assertEqual(row.scrape_quota, 7400)

    def test_scrape_only_packs_increase_quota_and_monthly_total(self):
        row = agency()
        billing.apply_subscription(
            row,
            {
                "id": "sub_scrapes",
                "status": "active",
                "items": {
                    "data": [
                        {"id": "si_base", "price": {"id": "price_agency"}, "quantity": 1},
                        {"id": "si_scrapes", "price": {"id": "price_scrapes"}, "quantity": 2},
                    ]
                },
            },
        )
        self.assertEqual(row.scrape_pack_count, 2)
        self.assertEqual(row.scrape_quota, 5200)
        self.assertEqual(row.client_pack_count, 0)
        budget = billing.compute_budget(row, active_clients=1)
        self.assertEqual(budget.extra_scrape_units, 200)
        self.assertEqual(budget.estimated_monthly_cents, 46000)

    def test_new_billing_period_resets_usage_once(self):
        row = agency()
        row.billing_period_start = datetime.fromtimestamp(
            1_600_000_000, timezone.utc
        ).replace(tzinfo=None)
        subscription = {
            "id": "sub_123",
            "status": "active",
            "current_period_start": 1_700_000_000,
            "current_period_end": 1_702_592_000,
            "items": {"data": [{"id": "si_base", "price": {"id": "price_individual"}, "quantity": 1}]},
        }
        billing.apply_subscription(row, subscription)
        self.assertEqual(row.plan, PlanType.creator)
        self.assertEqual(row.reports_used, 0)
        self.assertEqual(row.scrape_units_used, 0)
        self.assertEqual(row.reports_quota, 10)
        row.reports_used = 2
        billing.apply_subscription(row, subscription)
        self.assertEqual(row.reports_used, 2)

    def test_apply_subscription_accepts_stripe_objects_without_get(self):
        class StripeLike:
            def __init__(self, data):
                self._data = data

            def to_dict_recursive(self):
                return self._data

            def __getattr__(self, name):
                raise AttributeError(name)

        row = agency()
        billing.apply_subscription(
            row,
            StripeLike(
                {
                    "id": "sub_obj",
                    "status": "active",
                    "current_period_start": 1_700_000_000,
                    "current_period_end": 1_702_592_000,
                    "cancel_at_period_end": False,
                    "items": {"data": [{"id": "si_base", "price": {"id": "price_agency"}, "quantity": 1}]},
                }
            ),
        )
        self.assertEqual(row.stripe_subscription_id, "sub_obj")
        self.assertEqual(row.billing_status, "active")
        self.assertEqual(row.stripe_base_item_id, "si_base")

    def test_payg_zero_price_subscription_is_usage_model(self):
        row = agency()
        billing.apply_subscription(
            row,
            {
                "id": "sub_payg",
                "status": "active",
                "metadata": {"billing_model": "payg"},
                "current_period_start": 1_700_000_000,
                "current_period_end": 1_702_592_000,
                "items": {"data": [{"id": "si_payg", "price": {"id": "price_payg"}, "quantity": 1}]},
            },
        )
        self.assertEqual(row.billing_model, "payg")
        self.assertEqual(row.client_pack_count, 0)
        self.assertEqual(row.included_clients, billing.PAYG_CLIENT_CAP)
        self.assertEqual(row.reports_quota, billing.PAYG_REPORT_CAP)


if __name__ == "__main__":
    unittest.main()
