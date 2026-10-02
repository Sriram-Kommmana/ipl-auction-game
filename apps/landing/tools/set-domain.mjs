// Points the landing site at a real domain.
//
//   node tools/set-domain.mjs cricketauction.in
//   node tools/set-domain.mjs cricketauction.in --dry
//   node tools/set-domain.mjs cricketauction.in --app play.cricketauction.in
//   node tools/set-domain.mjs cricketauction.in --email hi@cricketauction.in --pay https://rzp.io/rzp/abc
//
// The static pages need absolute URLs in three places that cannot be
// relative — canonical tags, og:image/og:url, and the sitemap — plus the
// two CTA links and the stats API call that still point at localhost. This
// rewrites all of them together so none gets forgotten on deploy day.
//
// deploy/deploy.sh runs it on a throwaway copy of this folder, so the
// committed files keep their localhost/example values for development.
// --email and --pay fill in the policy pages' contact address and the
// footer's Razorpay link; without --pay the coffee button is removed rather
// than shipped pointing at the placeholder.
//
// Re-runnable: the last-applied values are recorded in tools/.domain, so
// changing the domain later works exactly the same way.
import { readFile, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const LANDING = join(here, '..')
const STATE = join(here, '.domain')

// What the repo ships with, before anyone has run this.
const DEFAULTS = {
    site: 'https://cricket-auction.example.com',
    app: 'http://localhost:5173',
    api: 'http://localhost:3001',
    email: 'contact@cricket-auction.example.com',
    pay: 'https://rzp.io/rzp/YOUR-PAYMENT-PAGE'
}

const TARGETS = [
    'index.html', 'players.html', 'robots.txt', 'sitemap.xml',
    'contact.html', 'terms.html', 'privacy.html', 'refund-policy.html'
]

const args = process.argv.slice(2)
const dry = args.includes('--dry')
const positional = args.filter((a) => !a.startsWith('--'))
const flag = (name) => {
    const i = args.indexOf(`--${name}`)
    return i !== -1 ? args[i + 1] : null
}

const domain = positional[0]
if (!domain) {
    console.error(`Usage: node tools/set-domain.mjs <domain> [--app host] [--api host] [--dry]

  <domain>   bare apex, e.g. cricketauction.in (no protocol, no trailing slash)
  --app      host for the game client   (default: play.<domain>)
  --api      host for the API server    (default: api.<domain>)
  --email    contact address shown on the policy pages
  --pay      Razorpay Payment Page URL (without it the coffee button is removed)
  --dry      report what would change, write nothing`)
    process.exit(1)
}

if (/^https?:\/\//.test(domain) || domain.endsWith('/')) {
    console.error(`Pass a bare hostname, not a URL: "${domain.replace(/^https?:\/\//, '').replace(/\/$/, '')}"`)
    process.exit(1)
}

const prev = existsSync(STATE) ? { ...DEFAULTS, ...JSON.parse(await readFile(STATE, 'utf8')) } : DEFAULTS

const next = {
    site: `https://${domain}`,
    app: `https://${flag('app') || `play.${domain}`}`,
    api: `https://${flag('api') || `api.${domain}`}`,
    email: flag('email') || prev.email,
    pay: flag('pay') || prev.pay
}

// The coffee button is a single line in each page; drop it while it still
// points at the placeholder.
const dropPlaceholderPay = (text) => next.pay === DEFAULTS.pay
    ? text.split('\n').filter((line) => !line.includes(DEFAULTS.pay)).join('\n')
    : text

const run = async () => {
    let totalHits = 0

    for (const name of TARGETS) {
        const file = join(LANDING, name)
        const original = await readFile(file, 'utf8')
        let updated = original
        const hits = []

        // email before site: the placeholder address contains the site host
        for (const key of ['email', 'pay', 'site', 'app', 'api']) {
            const from = prev[key]
            const to = next[key]
            if (from === to) continue
            const count = updated.split(from).length - 1
            if (count) {
                updated = updated.split(from).join(to)
                hits.push(`${count}x ${from} -> ${to}`)
            }
        }

        const withoutPay = dropPlaceholderPay(updated)
        if (withoutPay !== updated) {
            updated = withoutPay
            hits.push('removed coffee button (no --pay URL yet)')
        }

        if (!hits.length) {
            console.log(`  --  ${name} (nothing to change)`)
            continue
        }

        totalHits += hits.length
        console.log(`  ${dry ? '~~' : 'ok'}  ${name}`)
        for (const h of hits) console.log(`        ${h}`)
        if (!dry) await writeFile(file, updated, 'utf8')
    }

    if (dry) {
        console.log('\nDRY RUN — nothing written.')
        return
    }

    if (!totalHits) {
        console.log('\nNo occurrences found. Already pointed somewhere else?')
        console.log(`Last applied: ${JSON.stringify(prev)}`)
        return
    }

    await writeFile(STATE, `${JSON.stringify(next, null, 2)}\n`, 'utf8')

    console.log(`
Landing site now points at ${next.site}

Still to set by hand (they live outside this folder):

  web build env       VITE_SERVER_URL=${next.api}
                      VITE_LANDING_URL=${next.site}
                      (baked in at build time — rebuild after changing)

  server env          CLIENT_URL=${next.app}
                      (no trailing slash; compared literally by CORS)

On the production server deploy/deploy.sh sets all of these for you.
`)
}

run().catch((err) => {
    console.error('[set-domain] failed:', err.message)
    process.exit(1)
})
