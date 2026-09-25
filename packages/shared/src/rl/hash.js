// Deterministic, dependency-free spec hashing (FNV-1a, 64-bit). Used to stamp
// obs-v2 / act-v2 so a policy trained on one layout can never be loaded
// against another. Not cryptographic — it only has to change whenever the
// canonical spec text changes.

const FNV_OFFSET = 0xcbf29ce484222325n
const FNV_PRIME = 0x100000001b3n
const MASK = 0xffffffffffffffffn

export const fnv1a64 = (text) => {
    let h = FNV_OFFSET
    for (const byte of new TextEncoder().encode(text)) {
        h ^= BigInt(byte)
        h = (h * FNV_PRIME) & MASK
    }
    return h.toString(16).padStart(16, '0')
}

// Stable JSON: object keys sorted, so the hash never depends on insertion order.
export const canonicalJson = (value) => {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
    if (value && typeof value === 'object') {
        return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`
    }
    return JSON.stringify(value)
}

export const specHash = (spec) => fnv1a64(canonicalJson(spec))

// 32-bit seed mixing for independent, reproducible random streams derived
// from one episode seed (e.g. the lineup sampler must not disturb the
// simulator's own random stream).
export const deriveSeed = (seed, stream) => parseInt(fnv1a64(`${seed}:${stream}`).slice(-8), 16) >>> 0
