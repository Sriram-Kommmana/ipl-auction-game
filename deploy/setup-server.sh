#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server (AWS Lightsail or EC2) for
# Cricket Auction: Node.js, pnpm, Redis, MongoDB, Caddy, firewall, swap,
# the app user, config files, the systemd service, backups — then the first
# deploy. Re-running is safe: finished steps are skipped and existing
# passwords/config are kept.
#
# Usage (as the default admin user, e.g. "ubuntu"):
#   curl -fsSLO https://raw.githubusercontent.com/Sriram-Kommmana/ipl-auction-game/master/deploy/setup-server.sh
#   sudo bash setup-server.sh <domain> <contact-email>
#   e.g. sudo bash setup-server.sh cricketauction.in hello@cricketauction.in
#
# Passwords are generated here and written only to root-readable files in
# /etc/ipl-auction — never printed.
set -euo pipefail

DOMAIN=${1:-}
CONTACT_EMAIL=${2:-}
REPO_URL=${REPO_URL:-https://github.com/Sriram-Kommmana/ipl-auction-game.git}
BRANCH=${BRANCH:-master}

APP_USER=ipl
APP_DIR=/opt/ipl-auction
WEB_ROOT=/var/www/ipl-auction
CONF_DIR=/etc/ipl-auction
NODE_MAJOR=24
PNPM_VERSION=10.33.0
MONGO_SERIES=8.0
DB_NAME=ipl-auction

say()  { printf '\n\033[1;31m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
skip() { echo "    already done — skipping"; }
fail() { printf '\n\033[1;31mSETUP FAILED:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fail "run with sudo"
[[ -n "$DOMAIN" && -n "$CONTACT_EMAIL" ]] || fail "usage: sudo bash setup-server.sh <domain> <contact-email>"
[[ "$DOMAIN" =~ ^[a-z0-9.-]+\.[a-z]{2,}$ ]] || fail "domain must be a bare hostname like cricketauction.in"
# shellcheck source=/dev/null
. /etc/os-release
[[ "${VERSION_CODENAME:-}" == "noble" ]] || fail "this script targets Ubuntu 24.04 (noble); found ${PRETTY_NAME:-unknown}"

export DEBIAN_FRONTEND=noninteractive
mkdir -p "$CONF_DIR"
chmod 755 "$CONF_DIR"

# ── System ───────────────────────────────────────────────────────────────
say "System packages"
apt-get update -q
apt-get upgrade -yq
apt-get install -yq git curl gnupg ca-certificates ufw rsync \
    debian-keyring debian-archive-keyring apt-transport-https

say "Swap (2 GB — covers the web build on small instances)"
if swapon --show | grep -q .; then skip; else
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo 'vm.swappiness=10' > /etc/sysctl.d/99-swappiness.conf
    sysctl -q --system
fi

say "Firewall (SSH, HTTP, HTTPS only)"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw allow 443/udp >/dev/null   # HTTP/3
ufw --force enable >/dev/null
ufw status | sed -n '1,12p'

# ── Node.js + pnpm ───────────────────────────────────────────────────────
say "Node.js $NODE_MAJOR and pnpm $PNPM_VERSION"
if command -v node >/dev/null && [[ "$(node -p 'process.versions.node.split(".")[0]')" == "$NODE_MAJOR" ]]; then skip; else
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -
    apt-get install -yq nodejs
fi
npm install -g --silent "pnpm@$PNPM_VERSION"
echo "    node $(node -v), pnpm $(pnpm -v)"

# ── Redis (live room state) ──────────────────────────────────────────────
say "Redis (localhost only)"
apt-get install -yq redis-server
# Our settings live in their own file, included last so they win over the
# package defaults (and survive package upgrades of redis.conf).
cat > /etc/redis/ipl-auction.conf <<'EOF'
bind 127.0.0.1 -::1
protected-mode yes
supervised systemd
maxmemory 256mb
# Game state must never be silently evicted — fail writes instead.
maxmemory-policy noeviction
EOF
chown redis:redis /etc/redis/ipl-auction.conf
grep -qxF 'include /etc/redis/ipl-auction.conf' /etc/redis/redis.conf \
    || echo 'include /etc/redis/ipl-auction.conf' >> /etc/redis/redis.conf
systemctl enable --now redis-server >/dev/null
systemctl restart redis-server
redis-cli ping

# ── MongoDB (rooms, players, results) ────────────────────────────────────
say "MongoDB $MONGO_SERIES (localhost only, auth on)"
if ! command -v mongod >/dev/null; then
    curl -fsSL "https://pgp.mongodb.com/server-$MONGO_SERIES.asc" \
        | gpg --dearmor --yes -o "/usr/share/keyrings/mongodb-server-$MONGO_SERIES.gpg"
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-$MONGO_SERIES.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/$MONGO_SERIES multiverse" \
        > "/etc/apt/sources.list.d/mongodb-org-$MONGO_SERIES.list"
    apt-get update -q
    apt-get install -yq mongodb-org
fi

# MongoDB 8 refuses to start on Linux 6.19–7.0.13 (SERVER-121912): its
# bundled TCMalloc relies on rseq behaviour those kernels changed, and
# Ubuntu 24.04's rolling cloud kernel is in that range. Letting glibc own
# rseq makes TCMalloc fall back from per-CPU caches — a small throughput
# cost that a game this size never notices. Harmless on fixed kernels;
# remove once MongoDB ships a patched TCMalloc.
mkdir -p /etc/systemd/system/mongod.service.d
cat > /etc/systemd/system/mongod.service.d/kernel-rseq.conf <<'EOF'
[Service]
Environment=GLIBC_TUNABLES=glibc.pthread.rseq=1
EOF
systemctl daemon-reload

write_mongod_conf() {   # write_mongod_conf <auth: true|false>
    cat > /etc/mongod.conf <<EOF
# Managed by deploy/setup-server.sh
storage:
  dbPath: /var/lib/mongodb
  wiredTiger:
    engineConfig:
      cacheSizeGB: 0.25   # small box: leave RAM for Node and Redis
systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
net:
  port: 27017
  bindIp: 127.0.0.1
processManagement:
  timeZoneInfo: /usr/share/zoneinfo
EOF
    if [[ "$1" == true ]]; then
        printf 'security:\n  authorization: enabled\n' >> /etc/mongod.conf
    fi
}

wait_for_mongo() {
    for _ in $(seq 1 30); do
        mongosh --quiet --eval 'db.runCommand({ ping: 1 }).ok' >/dev/null 2>&1 && return 0
        sleep 1
    done
    fail "MongoDB did not start (see /var/log/mongodb/mongod.log)"
}

if [[ -f "$CONF_DIR/mongo-admin.uri" ]]; then
    echo "    users already created"
    write_mongod_conf true
    systemctl enable --now mongod >/dev/null
    systemctl restart mongod
    wait_for_mongo
else
    # Start without auth just long enough to create the two users.
    write_mongod_conf false
    systemctl enable --now mongod >/dev/null
    systemctl restart mongod
    wait_for_mongo

    ADMIN_PASS=$(openssl rand -hex 24)
    APP_PASS=$(openssl rand -hex 24)
    # Passwords go in through the environment, not the command line.
    ADMIN_PASS=$ADMIN_PASS APP_PASS=$APP_PASS DB_NAME=$DB_NAME mongosh --quiet --eval '
        db.getSiblingDB("admin").createUser({ user: "admin", pwd: process.env.ADMIN_PASS, roles: ["root"] });
        db.getSiblingDB(process.env.DB_NAME).createUser({ user: "ipl_app", pwd: process.env.APP_PASS,
            roles: [{ role: "readWrite", db: process.env.DB_NAME }] });
    ' >/dev/null

    umask 077
    echo "mongodb://admin:$ADMIN_PASS@127.0.0.1:27017/?authSource=admin" > "$CONF_DIR/mongo-admin.uri"
    # The server appends the database name (src/constants.js DB_NAME), so the
    # app user lives in that database and the URI ends with a slash.
    echo "$APP_PASS" > "$CONF_DIR/.mongo-app-pass"
    umask 022

    write_mongod_conf true
    systemctl restart mongod
    wait_for_mongo
fi

# ── Caddy (HTTPS + static files + proxy) ─────────────────────────────────
say "Caddy"
if command -v caddy >/dev/null; then skip; else
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list
    chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -q
    apt-get install -yq caddy
fi

# ── App user, code, folders ──────────────────────────────────────────────
say "App user '$APP_USER' and code in $APP_DIR"
id "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "/home/$APP_USER" --shell /bin/bash "$APP_USER"
usermod -aG systemd-journal "$APP_USER"   # deploy.sh shows logs on failure
mkdir -p "$APP_DIR" "$WEB_ROOT/landing" "$WEB_ROOT/web"
chown "$APP_USER:$APP_USER" "$APP_DIR" "$WEB_ROOT" "$WEB_ROOT/landing" "$WEB_ROOT/web"
if [[ -d "$APP_DIR/.git" ]]; then skip; else
    sudo -u "$APP_USER" git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

# ── Config files ─────────────────────────────────────────────────────────
say "Config in $CONF_DIR"
if [[ -f "$CONF_DIR/server.env" ]]; then
    echo "    server.env exists — keeping it"
else
    APP_PASS=$(cat "$CONF_DIR/.mongo-app-pass")
    umask 027
    cat > "$CONF_DIR/server.env" <<EOF
# Game server environment — read by the ipl-auction systemd service.
PORT=3001
REDIS_URL=redis://127.0.0.1:6379
MONGODB_URI=mongodb://ipl_app:$APP_PASS@127.0.0.1:27017/
CLIENT_URL=https://play.$DOMAIN
EOF
    umask 022
    chown root:"$APP_USER" "$CONF_DIR/server.env"
fi
rm -f "$CONF_DIR/.mongo-app-pass"

if [[ -f "$CONF_DIR/deploy.env" ]]; then
    echo "    deploy.env exists — keeping it"
else
    cat > "$CONF_DIR/deploy.env" <<EOF
# Read by deploy/deploy.sh on every deploy.
DOMAIN=$DOMAIN
CONTACT_EMAIL=$CONTACT_EMAIL
# Your Razorpay Payment Page URL. Leave empty to hide the coffee button.
PAY_URL=
EOF
    chown root:"$APP_USER" "$CONF_DIR/deploy.env"
    chmod 640 "$CONF_DIR/deploy.env"
fi

# ── Service, sudo rule, Caddy site, backups ──────────────────────────────
say "systemd service"
install -m 644 "$APP_DIR/deploy/ipl-auction.service" /etc/systemd/system/ipl-auction.service
systemctl daemon-reload
systemctl enable ipl-auction >/dev/null

say "Let '$APP_USER' restart only its own service"
cat > /etc/sudoers.d/ipl-auction <<EOF
$APP_USER ALL=(root) NOPASSWD: /usr/bin/systemctl restart ipl-auction
EOF
chmod 440 /etc/sudoers.d/ipl-auction
visudo -cqf /etc/sudoers.d/ipl-auction || fail "sudoers rule did not validate"

say "Caddy sites for $DOMAIN, play.$DOMAIN, api.$DOMAIN"
sed "s/__DOMAIN__/$DOMAIN/g" "$APP_DIR/deploy/Caddyfile.template" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
systemctl reload caddy || systemctl restart caddy

say "Nightly MongoDB backup (03:30, keeps 14)"
install -m 755 "$APP_DIR/deploy/backup-mongo.sh" /usr/local/bin/ipl-auction-backup
echo '30 3 * * * root /usr/local/bin/ipl-auction-backup' > /etc/cron.d/ipl-auction-backup
chmod 644 /etc/cron.d/ipl-auction-backup

# ── Dependencies, player pool, first deploy ──────────────────────────────
say "Installing dependencies"
sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && pnpm install --frozen-lockfile --reporter=silent"

say "Player pool"
players=$(mongosh --quiet "$(cat "$CONF_DIR/mongo-admin.uri")" \
    --eval "db.getSiblingDB('$DB_NAME').players.countDocuments()")
if [[ "$players" -gt 0 ]]; then
    echo "    $players players already in the database"
else
    sudo -u "$APP_USER" bash -c "set -a; . '$CONF_DIR/server.env'; set +a; cd '$APP_DIR/apps/server' && node src/db/seed.js"
fi

say "First deploy"
sudo -u "$APP_USER" bash "$APP_DIR/deploy/deploy.sh" --force

cat <<EOF

Setup complete.

  Landing  https://$DOMAIN
  Game     https://play.$DOMAIN
  API      https://api.$DOMAIN/health

HTTPS certificates are issued automatically once the DNS records for
$DOMAIN, www.$DOMAIN, play.$DOMAIN and api.$DOMAIN point at this server
(check with: sudo journalctl -u caddy -f).

Future deploys:   sudo -u $APP_USER bash $APP_DIR/deploy/deploy.sh
Razorpay link:    set PAY_URL in $CONF_DIR/deploy.env, then deploy again.
EOF
