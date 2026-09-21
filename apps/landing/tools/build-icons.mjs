// Generates every raster icon and the social share image from two sources:
// favicon.svg (the cricket-ball mark) and tools/og-card.html.
//
//   node tools/build-icons.mjs
//
// Rendering is done by headless Chrome/Edge, which is already on this machine
// and gets us real web-font rendering for the OG card — something an SVG
// rasteriser can't do without the fonts installed locally.
//
// Outputs (all committed, so deploying needs no build step):
//   favicon.ico                     16/32/48, packed here
//   assets/apple-touch-icon.png     180, opaque (iOS ignores transparency)
//   assets/icon-192.png             PWA / Android
//   assets/icon-512.png             PWA / Android
//   assets/icon-maskable-512.png    extra padding for Android's mask
//   assets/og-cover.png             1200x630 share card
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'
import { tmpdir } from 'node:os'
import { existsSync } from 'node:fs'

const run = promisify(execFile)
const here = dirname(fileURLToPath(import.meta.url))
const LANDING = join(here, '..')
const ASSETS = join(LANDING, 'assets')

const CHROME_CANDIDATES = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe'
]

const findBrowser = () => {
    const found = CHROME_CANDIDATES.find((p) => existsSync(p))
    if (!found) throw new Error('No Chrome or Edge found for rendering')
    return found
}

const BROWSER = findBrowser()
let workdir

// Chrome writes the screenshot itself; --virtual-time-budget gives webfonts
// and layout time to settle before the shot is taken.
const shoot = async (url, out, width, height) => {
    await run(BROWSER, [
        '--headless',
        '--disable-gpu',
        '--hide-scrollbars',
        '--force-device-scale-factor=1',
        '--default-background-color=00000000',
        `--window-size=${width},${height}`,
        '--virtual-time-budget=6000',
        `--screenshot=${out}`,
        url
    ])
    if (!existsSync(out)) throw new Error(`Chrome produced no file for ${out}`)
}

// Wrap the mark in a page sized exactly to the target so the shot is edge to
// edge. `pad` (0-1) shrinks the mark inside its background for the maskable
// variant, whose outer ~20% can be cropped away by Android.
const renderMark = async (svg, out, size, pad = 0) => {
    const inset = Math.round(size * pad)
    const page = `<!DOCTYPE html><meta charset="utf-8"><style>
        *{margin:0;padding:0}
        html,body{width:${size}px;height:${size}px;overflow:hidden;background:#060607}
        .wrap{width:${size}px;height:${size}px;display:flex;align-items:center;justify-content:center}
        svg{width:${size - inset * 2}px;height:${size - inset * 2}px;display:block}
    </style><div class="wrap">${svg}</div>`
    const file = join(workdir, `mark-${size}-${pad}.html`)
    await writeFile(file, page, 'utf8')
    await shoot(pathToFileURL(file).href, out, size, size)
}

// Minimal ICO container. Every modern browser and Windows itself reads
// PNG-compressed entries, so the frames go in as-is.
const packIco = (pngs) => {
    const header = Buffer.alloc(6)
    header.writeUInt16LE(0, 0) // reserved
    header.writeUInt16LE(1, 2) // 1 = icon
    header.writeUInt16LE(pngs.length, 4)

    let offset = 6 + pngs.length * 16
    const entries = []
    for (const { size, data } of pngs) {
        const e = Buffer.alloc(16)
        e.writeUInt8(size >= 256 ? 0 : size, 0)
        e.writeUInt8(size >= 256 ? 0 : size, 1)
        e.writeUInt8(0, 2) // palette
        e.writeUInt8(0, 3) // reserved
        e.writeUInt16LE(1, 4) // colour planes
        e.writeUInt16LE(32, 6) // bits per pixel
        e.writeUInt32LE(data.length, 8)
        e.writeUInt32LE(offset, 12)
        offset += data.length
        entries.push(e)
    }
    return Buffer.concat([header, ...entries, ...pngs.map((p) => p.data)])
}

const main = async () => {
    workdir = await mkdtemp(join(tmpdir(), 'ipl-icons-'))
    try {
        const svg = await readFile(join(LANDING, 'favicon.svg'), 'utf8')

        const targets = [
            { out: join(ASSETS, 'apple-touch-icon.png'), size: 180, pad: 0 },
            { out: join(ASSETS, 'icon-192.png'), size: 192, pad: 0 },
            { out: join(ASSETS, 'icon-512.png'), size: 512, pad: 0 },
            { out: join(ASSETS, 'icon-maskable-512.png'), size: 512, pad: 0.12 }
        ]
        for (const t of targets) {
            await renderMark(svg, t.out, t.size, t.pad)
            console.log(`  ok  ${t.out.split(/[\\/]/).pop()} (${t.size}px)`)
        }

        // favicon.ico frames
        const icoSizes = [16, 32, 48]
        const frames = []
        for (const size of icoSizes) {
            const tmp = join(workdir, `ico-${size}.png`)
            await renderMark(svg, tmp, size, 0)
            frames.push({ size, data: await readFile(tmp) })
        }
        await writeFile(join(LANDING, 'favicon.ico'), packIco(frames))
        console.log(`  ok  favicon.ico (${icoSizes.join(', ')})`)

        // Social share card. Chrome only writes PNG, and this artwork lands
        // around 340 KB as one — heavy enough that WhatsApp can decline to
        // build a preview. Re-encoded to JPEG it is ~150 KB with no visible
        // loss, so the JPEG is what ships and the PNG is a build artefact.
        const card = pathToFileURL(join(here, 'og-card.html')).href
        const ogPng = join(ASSETS, 'og-cover.png')
        await shoot(card, ogPng, 1200, 630)

        try {
            await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', ogPng, '-q:v', '3', join(ASSETS, 'og-cover.jpg'), '-y'])
            await rm(ogPng, { force: true })
            console.log('  ok  og-cover.jpg (1200x630)')
        } catch {
            console.warn('  !!  ffmpeg not found — keeping og-cover.png; update og:image if you ship it')
        }
    } finally {
        await rm(workdir, { recursive: true, force: true })
    }
}

main().catch((err) => {
    console.error('[build-icons] failed:', err.message)
    process.exit(1)
})
