# Stripe billing setup

MarketBiqs uses Stripe-hosted Checkout and Customer Portal. Stripe webhooks are
the source of truth for subscription status and PAYG pack quantity.

## Products and recurring prices

Create three monthly recurring Prices in the same Stripe mode (test or live):

1. **MarketBiqs Agency** — USD 450.00/month
2. **MarketBiqs Individual** — USD 99.00/month
3. **MarketBiqs Client Add-on Pack** — USD 49.00/month
4. **MarketBiqs Scrape Units (100)** — USD 5.00/month per 100 units

The add-on pack grants one client slot, eight reports, and 800 scrape units.
Scrape-only add-ons are sold in lots of 100 units.
Set the resulting Price IDs as:

```text
STRIPE_AGENCY_PRICE_ID=price_...
STRIPE_INDIVIDUAL_PRICE_ID=price_...
STRIPE_CLIENT_PACK_PRICE_ID=price_...
STRIPE_SCRAPE_PACK_PRICE_ID=price_...
```

Also set:

```text
STRIPE_SECRET_KEY=sk_...
STRIPE_PUBLISHABLE_KEY=pk_...
STRIPE_WEBHOOK_SECRET=whsec_...
FRONTEND_URL=https://marketbiqsfrontend-production.up.railway.app
```

Never commit real keys.

## Customer Portal

Enable Stripe Customer Portal for:

- payment-method updates;
- invoice history;
- subscription cancellation at period end.

Plan and pack changes are performed inside MarketBiqs so local entitlements can
be synchronized immediately and then confirmed by webhook.

## Webhook

Production endpoint:

```text
https://YOUR_BACKEND/api/billing/webhook
```

Subscribe it to:

- `checkout.session.completed`
- `checkout.session.async_payment_succeeded`
- `checkout.session.async_payment_failed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`

Webhook event IDs are stored uniquely in `stripe_events`, so Stripe retries do
not grant packs twice.

## Local test flow

Run the API, then forward Stripe test events:

```bash
stripe listen --forward-to http://127.0.0.1:8000/api/billing/webhook
```

Copy the printed `whsec_...` into the local backend environment and restart.
Use Stripe's test card `4242 4242 4242 4242`.

Validate:

1. Agency checkout activates at $450/month.
2. Individual checkout activates at $99/month.
3. Changing pack quantity creates a prorated $49/month subscription item.
4. Replaying the same webhook does not change pack quantity twice.
5. Customer Portal opens and cancellation updates MarketBiqs after webhook.
6. Failed invoice changes the workspace billing status to `past_due`.

## Railway cutover

Set all six Stripe variables on the backend service, not the frontend. Test and
live keys/Price IDs must never be mixed. After deploying, send a test webhook
from Stripe Dashboard and confirm a 200 response before accepting payments.
