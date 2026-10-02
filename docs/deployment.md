# Deployment

Cricket Auction runs on **one Ubuntu 24.04 server on AWS**. Everything lives
on that box: the Node game server, Redis, MongoDB and Caddy, which serves the
static files and handles HTTPS.

```
                  ┌────────────────────── AWS server (Ubuntu 24.04) ──────────────────────┐
 yourdomain.com ──┤ Caddy :443 ─► /var/www/ipl-auction/landing   (static landing page)     │
 play.yourdomain ─┤            ─► /var/www/ipl-auction/web       (React game app, SPA)     │
 api.yourdomain ──┤            ─► 127.0.0.1:3001  Node: Express + Socket.IO + bots         │
                  │                                 ├─► Redis   127.0.0.1:6379  live rooms │
                  │                                 └─► MongoDB 127.0.0.1:27017 results    │
                  └────────────────────────────────────────────────────────────────────────┘
```

Everything in `deploy/`:

| File | What it does |
|---|---|
| `setup-server.sh` | One-time setup of a fresh server, ending with the first deploy |
| `deploy.sh` | Every later deploy: pull, install, build, publish, restart, health check |
| `Caddyfile.template` | HTTPS and routing for the three hostnames |
| `ipl-auction.service` | systemd unit for the game server |
| `backup-mongo.sh` | Nightly `mongodump`, keeps 14 |

---

## 1. Create the server

**Lightsail** is AWS's VPS product with a fixed monthly price, a static IP
and data transfer included. It's the simplest fit. EC2 works too; see the
note at the end of this section.

1. Lightsail console → **Create instance**
   - Region: **Mumbai (ap-south-1)**, closest to Indian players, and a
     real-time auction needs low latency.
   - Platform **Linux/Unix**, blueprint **OS Only → Ubuntu 24.04 LTS**.
   - Plan: **2 GB RAM** at minimum. Node, Redis and MongoDB share it, and
     setup adds 2 GB swap for builds. The 4 GB plan gives more room.
   - Turn on **automatic snapshots**. They're a second backup on top of the
     nightly database dump.
2. **Networking** tab:
   - **Create static IP** and attach it to the instance.
   - IPv4 firewall: keep SSH (22) and HTTP (80), and **add HTTPS (443)**.
     Restrict SSH to your own IP if you can.
3. Download the SSH key, or use the browser SSH button.

> **EC2 instead?** Use a `t3.small` or larger, Ubuntu 24.04, 30 GB gp3, an
> **Elastic IP**, and a security group allowing 22 (your IP), 80 and 443.
> Everything below is the same.

## 2. Point the domain at it

At your DNS provider, create four **A records**, all pointing to the
static IP:

| Name | Type | Value |
|---|---|---|
| `@` (the apex, e.g. `cricketauction.in`) | A | static IP |
| `www` | A | static IP |
| `play` | A | static IP |
| `api` | A | static IP |

Caddy requests the HTTPS certificates automatically once these resolve.
If you use **Cloudflare**, set SSL/TLS mode to **Full (strict)**. Proxied
(orange cloud) and DNS-only both work, and Cloudflare passes WebSockets
through.

## 3. Run the setup script

SSH in as `ubuntu` and run:

```bash
curl -fsSLO https://raw.githubusercontent.com/Sriram-Kommmana/ipl-auction-game/master/deploy/setup-server.sh
sudo bash setup-server.sh yourdomain.com you@yourdomain.com
```

It takes about 5–10 minutes. It installs and configures everything, creates
the `ipl` app user, clones the repo to `/opt/ipl-auction`, seeds the 323
players, and runs the first deploy.

- **Database passwords** are generated on the server and stored only in
  root-readable files under `/etc/ipl-auction/`. They are never printed.
- **Re-running** the script is safe. It skips finished steps and keeps the
  existing passwords and config.
- **If the GitHub repo is private,** the `curl` and `git clone` steps need
  access. Make the repo public, or add a read-only **deploy key** for the
  `ipl` user and pass `REPO_URL=git@github.com:Sriram-Kommmana/ipl-auction-game.git`
  before `sudo` (e.g. `sudo REPO_URL=... bash setup-server.sh ...`).

Check it worked:

```bash
curl https://api.yourdomain.com/health      # {"status":"ok"}
```

Then open `https://yourdomain.com` and `https://play.yourdomain.com`.

## 4. Deploying updates

Push to `master` from your machine, then on the server:

```bash
sudo -u ipl bash /opt/ipl-auction/deploy/deploy.sh
```

The script does the following:

1. **Refuses to run while an auction is live.** Restarting the server
   interrupts running auctions, because lot timers live in the process.
   Add `--force` to deploy anyway.
2. **Updates the code.** It pulls, then runs `pnpm install --frozen-lockfile`.
3. **Builds the game app** with
   `VITE_SERVER_URL=https://api.<domain>` and
   `VITE_LANDING_URL=https://<domain>`. Old hashed assets are kept for 14
   days so players with an open tab don't break.
4. **Publishes the landing page.** It copies `apps/landing`, runs
   `tools/set-domain.mjs` on the copy, and publishes it without `tools/`.
   The deploy stops if any `localhost`, `example.com` or payment placeholder
   would be published.
5. **Restarts the server** and waits for `/health`. If the check fails, it
   prints the last log lines.

## 5. Razorpay "Buy me a coffee"

The button stays hidden until you add your Payment Page link:

```bash
sudo nano /etc/ipl-auction/deploy.env     # PAY_URL=https://rzp.io/rzp/...
sudo -u ipl bash /opt/ipl-auction/deploy/deploy.sh
```

Before applying to Razorpay, fill in the `[City, State]` / `[City]`
placeholders in `apps/landing/contact.html` and `terms.html` and deploy;
the deploy prints a warning until you do. In the Razorpay dashboard, set
the Payment Page's redirect after payment to `https://yourdomain.com/?thanks=1`.

## 6. Day-to-day operations

| Task | Command |
|---|---|
| Server logs (live) | `sudo journalctl -u ipl-auction -f` |
| Restart server | `sudo systemctl restart ipl-auction` |
| Caddy / HTTPS logs | `sudo journalctl -u caddy -f` |
| Service status | `systemctl status ipl-auction redis-server mongod caddy` |
| Run a backup now | `sudo /usr/local/bin/ipl-auction-backup` |
| List backups | `sudo ls -lh /var/backups/ipl-auction` |
| Mongo shell (admin) | `mongosh "$(sudo cat /etc/ipl-auction/mongo-admin.uri)"` |
| Redis shell | `redis-cli` |
| OS security updates | applied automatically by Ubuntu's `unattended-upgrades`; reboot occasionally with `sudo reboot` |

**Restoring a backup:**

```bash
sudo mongorestore --uri "$(sudo cat /etc/ipl-auction/mongo-admin.uri)" \
    --gzip --archive=/var/backups/ipl-auction/<file>.archive.gz --drop
```

**Copying data from the old MongoDB Atlas database (optional):**
development used Atlas, so earlier results and room counts live there.
Setup re-seeds the players, so you only need this if you want the old
results. Run it on the server:

```bash
mongodump --uri "<atlas-uri>/ipl-auction" --gzip --archive=/tmp/atlas.gz
sudo mongorestore --uri "$(sudo cat /etc/ipl-auction/mongo-admin.uri)" --gzip --archive=/tmp/atlas.gz
rm /tmp/atlas.gz
```

**Uptime monitoring:** point a free monitor (for example UptimeRobot) at
`https://api.yourdomain.com/health`.

## Notes and limits

- **One server, one Node process.** Live auctions keep their timers and bot
  schedules in memory, which is why a restart interrupts them and why the
  deploy script checks for live auctions first.
- **RL bots on a small CPU.** If an RL bot's decision ever takes longer than
  the 20 ms budget, that room falls back to the rule bots. It's a safety
  net, not an error. Burstable instances that run out of CPU credits make
  this more likely; the server logs report it.
- **Redis** is localhost-only with `maxmemory 256mb` and `noeviction`.
  Rooms expire on their own after 4 days.
- **MongoDB** is localhost-only with authentication on. The app user can
  only read and write the `ipl-auction` database, and the WiredTiger cache is
  capped at 256 MB.
