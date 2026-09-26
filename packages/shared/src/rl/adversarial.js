// Adversarial learner controllers (Phase 2C.2) — deliberately pathological
// policies that act through the canonical act-v2 action space and whatever
// mask the environment applies. They exist to test that the completion
// mechanism protects the legal-XI invariant against bad learned behaviour;
// they are not baselines and are never trained.
//
// "Prefer PASS" means PASS when legal, else the lowest legal bid — so any
// requirement they end up filling is filled because the shield forced it.

import { PASS } from './actionSpec.js'
import { BASELINES } from './evaluate.js'

const STAR = 90
const legalBids = (m) => m.mask.flatMap((ok, a) => (ok && a !== PASS ? [a] : []))
const highest = (m) => legalBids(m).reduce((b, a) => (m.caps[a] > m.caps[b] ? a : b))
const lowest = (m) => legalBids(m).reduce((b, a) => (m.caps[a] < m.caps[b] ? a : b))
const preferPass = (m) => (m.mask[PASS] ? PASS : lowest(m))
const fair = (ep, rng) => BASELINES.fairValue.act(ep, rng)
const isKeeper = (lot) => lot.role === 'WICKET KEEPER'
const fillsSpecific = (plan) => plan.playerImpact.fillsRequirement.some((r) => r !== 'players')

const controller = (fn) => ({ kind: 'action', act: (ep, rng) => fn(ep.pending, ep, rng) })

export const ADVERSARIAL = Object.freeze({
    // Stars at any legal price, fair value otherwise.
    advStarSpender: controller((p, ep, rng) => (p.ctx.lot.rating >= STAR ? highest(p.mask) : fair(ep, rng))),
    // Always the highest legal bid: spends straight down to the planner's floor.
    advSpendDown: controller((p) => highest(p.mask)),
    // Never a keeper unless forced; fair value otherwise.
    advPassKeepers: controller((p, ep, rng) => (isKeeper(p.ctx.lot) ? preferPass(p.mask) : fair(ep, rng))),
    // Never a player who fills a missing keeper / bowling / Indian need unless forced.
    advPassRequirements: controller((p, ep, rng) => (fillsSpecific(p.plan) ? preferPass(p.mask) : fair(ep, rng))),
    // Stars only (highest legal bid), nothing else unless forced.
    advMaxStars: controller((p) => (p.ctx.lot.rating >= STAR ? highest(p.mask) : preferPass(p.mask))),
    // Highest legal bid on everything in the first 30% of the main round, then pass.
    advEarlySpender: controller((p) => (p.ctx.phase === 'main' && p.ctx.progress < 0.3 ? highest(p.mask) : preferPass(p.mask))),
    // Pass the whole main round, then the highest legal bid in the re-auction.
    advWaitForReauction: controller((p) => (p.ctx.phase === 'main' ? preferPass(p.mask) : highest(p.mask)))
})
