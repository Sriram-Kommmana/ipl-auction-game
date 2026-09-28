// Phase 2E.0 — frozen-hash guard.
//   node ml/ipl_rl/crossplay/hashes.mjs --record <file>   write the sha256 record (once, before the full run)
//   node ml/ipl_rl/crossplay/hashes.mjs --check <file>    exit 1 on any difference
// Covers the obs/act spec hashes, every frozen source/data/manifest file and
// all 15 exports (policy.json and the training checkpoint beside it). The
// frozen sources are also checked against git HEAD (no uncommitted change).
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { FROZEN_SOURCES, ROOT, frozenHashes, loadExports } from './common.mjs'

const [mode, file] = process.argv.slice(2)
const now = frozenHashes()
loadExports()
const dirty = execFileSync('git', ['status', '--porcelain', '--', ...FROZEN_SOURCES], { cwd: fileURLToPath(ROOT), encoding: 'utf8' }).trim()
if (dirty) {
    console.log(`FROZEN FILES CHANGED vs git HEAD:\n${dirty}`)
    process.exit(1)
}
if (mode === '--record') {
    if (existsSync(file)) throw new Error(`${file} exists — refusing to overwrite the frozen-hash record`)
    writeFileSync(file, JSON.stringify(now, null, 2))
    console.log(`recorded ${Object.keys(now.sources).length} frozen files + ${Object.keys(now.exports).length} exports → ${file}`)
} else if (mode === '--check') {
    const rec = JSON.parse(readFileSync(file, 'utf8'))
    const diffs = []
    if (rec.obsSpecHash !== now.obsSpecHash) diffs.push('obsSpecHash')
    if (rec.actSpecHash !== now.actSpecHash) diffs.push('actSpecHash')
    for (const [f, h] of Object.entries(rec.sources)) if (now.sources[f] !== h) diffs.push(f)
    for (const [k, v] of Object.entries(rec.exports)) if (JSON.stringify(now.exports[k]) !== JSON.stringify(v)) diffs.push(k)
    if (diffs.length) {
        console.log(`HASH MISMATCH: ${diffs.join(', ')}`)
        process.exit(1)
    }
    console.log(`HASHES OK: obs ${now.obsSpecHash} · act ${now.actSpecHash} · ${Object.keys(rec.sources).length} frozen files unchanged (and clean vs git HEAD) · ${Object.keys(rec.exports).length} exports + checkpoints unchanged`)
} else throw new Error('usage: hashes.mjs --record|--check <file>')
