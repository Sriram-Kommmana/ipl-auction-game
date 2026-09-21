# Landing site build tools

The landing site is deliberately plain HTML/CSS/JS with no build step — it is
served exactly as it sits in this folder. These scripts generate the few
things that would be tedious or error-prone to maintain by hand. Their output
is committed, so deploying stays a file copy.

Run them from `apps/landing`.

## `set-domain.mjs` — run this before deploying

```
node tools/set-domain.mjs cricketauction.in
```

The repo ships with the placeholder host `cricket-auction.example.com` and
with the two CTA buttons and the stats API still pointing at `localhost`.
This rewrites all of them — canonical tags, `og:url`, `og:image`,
`robots.txt`, `sitemap.xml`, the Enter/Create buttons and `STATS_API` — and
prints the `.env` values you still need to set by hand.

Add `--dry` to see what would change. It is re-runnable: the last applied
values are recorded in `tools/.domain`.

## `build-players-table.mjs` — run after editing the player list

```
node tools/build-players-table.mjs
```

Writes the 323 player rows into the `<tbody>` of `players.html` between
generated markers. `players.js` still owns search / filter / sort at runtime;
the static rows exist so crawlers and JS-less visitors see the real content
on the first pass instead of an empty table.

The player data and the price formatter are read straight out of
`players.js`, so the two cannot drift. **Re-run this whenever `players.js`
changes.**

## `build-icons.mjs` — run after changing `favicon.svg` or `og-card.html`

```
node tools/build-icons.mjs
```

Renders, using headless Chrome or Edge (whichever is installed):

| Output | From | Notes |
| --- | --- | --- |
| `favicon.ico` | `favicon.svg` | 16/32/48, packed by the script |
| `assets/apple-touch-icon.png` | `favicon.svg` | 180px, opaque |
| `assets/icon-192.png`, `icon-512.png` | `favicon.svg` | PWA / Android |
| `assets/icon-maskable-512.png` | `favicon.svg` | extra padding for Android's mask |
| `assets/og-cover.jpg` | `tools/og-card.html` | 1200×630 share card |

Headless Chrome is used rather than an SVG rasteriser because the share card
needs real web-font rendering (Anton, Archivo, JetBrains Mono) and those
fonts are not installed locally — Chrome pulls them from Google Fonts, so
this step needs a network connection.

The share card is re-encoded to JPEG with `ffmpeg`, because the PNG lands
around 340 KB and WhatsApp can decline to build a preview at that size; the
JPEG is ~150 KB with no visible loss.

**Edit `og-card.html`, never the generated image.**

## Images

There is no script for these — the source images change rarely. The hero was
converted with:

```
ffmpeg -i assets/hero-batter.png -c:v libwebp -quality 82 -compression_level 6 assets/hero-batter.webp
ffmpeg -i assets/hero-batter.png -vf scale=480:-1 -c:v libwebp -quality 80 -compression_level 6 assets/hero-batter-480.webp
```

That took it from 1.4 MB to 93 KB (and 50 KB for the mobile variant, which is
what actually gets served below 992px, where the image renders at roughly
157×245 CSS pixels).

The original PNGs are kept as masters for regeneration. They are no longer
referenced by any page, so nothing downloads them.

## A note on this folder

`apps/landing` is the web root, so `/tools/` is publicly reachable once
deployed. It is disallowed in `robots.txt`; if you would rather it not be
served at all, block the path in the reverse proxy.
