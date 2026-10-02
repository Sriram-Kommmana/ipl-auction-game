#!/usr/bin/env bash
# Nightly MongoDB backup (cron: /etc/cron.d/ipl-auction-backup, 03:30 server time).
# Keeps the last 14 dumps in /var/backups/ipl-auction.
#
# Restore one:
#   sudo mongorestore --uri "$(sudo cat /etc/ipl-auction/mongo-admin.uri)" \
#        --gzip --archive=/var/backups/ipl-auction/<file>.archive.gz --drop
set -euo pipefail

DEST=/var/backups/ipl-auction
KEEP=14
URI_FILE=/etc/ipl-auction/mongo-admin.uri

mkdir -p "$DEST"
chmod 700 "$DEST"

file="$DEST/ipl-auction-$(date +%Y%m%d-%H%M%S).archive.gz"
mongodump --uri "$(cat "$URI_FILE")" --db ipl-auction --gzip --archive="$file" --quiet

# rotate: newest $KEEP stay
ls -1t "$DEST"/ipl-auction-*.archive.gz | tail -n +$((KEEP + 1)) | xargs -r rm -f
