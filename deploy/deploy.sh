#!/usr/bin/env bash
# Deploys the latest master to this server: pull, install, build, publish,
# restart, health-check. Safe to re-run.
#
#   sudo -u ipl bash /opt/ipl-auction/deploy/deploy.sh            # normal deploy
#   sudo -u ipl bash /opt/ipl-auction/deploy/deploy.sh --force    # even if auctions are live
#
# Settings come from /etc/ipl-auction/deploy.env (written by setup-server.sh):
#   DOMAIN          apex domain, e.g. cricketauction.in
#   CONTACT_EMAIL   shown on the policy pages
#   PAY_URL         Razorpay Payment Page URL        (optional; no button without it)
set -euo pipefail

APP_DIR=/opt/ipl-auction
WEB_ROOT=/var/www/ipl-auction
DEPLOY_ENV=/etc/ipl-auction/deploy.env
SERVICE=ipl-auction
FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

say()  { printf '\n\033[1;31m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mDEPLOY FAILED:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(id -un)" == "ipl" ]] || fail "run as the ipl user: sudo -u ipl bash $0 $*"
[[ -r "$DEPLOY_ENV" ]] || fail "$DEPLOY_ENV missing — run deploy/setup-server.sh first"
# shellcheck source=/dev/null
source "$DEPLOY_ENV"
: "${DOMAIN:?DOMAIN is not set in $DEPLOY_ENV}"
[[ -n "${CONTACT_EMAIL:-}" ]] || fail "set CONTACT_EMAIL in $DEPLOY_ENV (the policy pages need a real address)"

cd "$APP_DIR"

# ── 1. Don't cut live auctions short ────────────────────────────────────
# Lot timers and bot schedules live in the server process, so a restart
# interrupts any auction that is running. Rooms are room:<id> hashes.
say "Checking for live auctions"
live=0
while read -r key; do
    [[ "$key" =~ ^room:[^:]+$ ]] || continue
    status=$(redis-cli HGET "$key" status)
    [[ "$status" == "active" || "$status" == "paused" ]] && live=$((live + 1))
done < <(redis-cli --scan --pattern 'room:*')
if (( live > 0 )); then
    if $FORCE; then
        echo "$live auction(s) in progress — continuing because of --force."
    else
        fail "$live auction(s) in progress. Deploy later, or re-run with --force to interrupt them."
    fi
else
    echo "None."
fi

# ── 2. Code ─────────────────────────────────────────────────────────────
say "Pulling latest code"
before=$(git rev-parse --short HEAD)
git pull --ff-only --quiet
after=$(git rev-parse --short HEAD)
echo "$before -> $after"

say "Installing dependencies"
pnpm install --frozen-lockfile --reporter=silent

# ── 3. Game app (React) ─────────────────────────────────────────────────
# VITE_* values are baked into the bundle at build time.
say "Building the game app"
VITE_SERVER_URL="https://api.$DOMAIN" \
VITE_LANDING_URL="https://$DOMAIN" \
    pnpm --filter web build

# Hashed assets are copied without --delete first, so players who still have
# the previous index.html open can lazy-load the chunks it references. Then
# index.html and the rest are swapped in. Old assets are pruned after 14 days.
mkdir -p "$WEB_ROOT/web/assets"
rsync -a apps/web/dist/assets/ "$WEB_ROOT/web/assets/"
rsync -a --delete --exclude 'assets/' apps/web/dist/ "$WEB_ROOT/web/"
find "$WEB_ROOT/web/assets" -type f -mtime +14 -delete

# ── 4. Landing page (static) ────────────────────────────────────────────
# The committed pages keep their localhost/example.com development values;
# set-domain.mjs rewrites a throwaway copy, which is what gets published.
say "Preparing the landing page"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -r apps/landing/. "$stage/"
node "$stage/tools/set-domain.mjs" "$DOMAIN" \
    --app "play.$DOMAIN" --api "api.$DOMAIN" \
    --email "$CONTACT_EMAIL" \
    ${PAY_URL:+--pay "$PAY_URL"} >/dev/null
rm -rf "$stage/tools"

leftover=$(grep -rIl -e 'localhost' -e 'example\.com' -e 'YOUR-PAYMENT-PAGE' "$stage" || true)
[[ -z "$leftover" ]] || fail "placeholders still present in: $leftover"
if grep -rIq '\[City' "$stage"; then
    echo "WARNING: contact.html / terms.html still contain [City] placeholders — fill them in before applying to Razorpay."
fi
[[ -n "${PAY_URL:-}" ]] || echo "Note: PAY_URL is not set, so the 'Buy me a coffee' button is left out."

rsync -a --delete "$stage/" "$WEB_ROOT/landing/"

# ── 5. Server ───────────────────────────────────────────────────────────
say "Restarting the game server"
sudo /usr/bin/systemctl restart "$SERVICE"

for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:3001/health >/dev/null 2>&1; then
        say "Deployed $after — https://$DOMAIN · https://play.$DOMAIN · https://api.$DOMAIN"
        exit 0
    fi
    sleep 1
done

journalctl -u "$SERVICE" -n 40 --no-pager || true
fail "server did not answer /health within 30s (logs above)"
