// Shared planning layer for the rule bots: what is SAFE and FEASIBLE for a
// team, never what a personality WANTS. Personalities decide how much they
// desire a player; this layer tells them whether buying it keeps a legal XI
// reachable, how scarce each requirement is, and how much money is truly
// spare. (The RL observation keeps its own spend limit in observation.js.)
//
// ── Why counts are enough ─────────────────────────────────────────────────
// Whether a squad can field a legal XI (scoring.js: 11 players, ≥1 keeper,
// ≥5 bowling options, ≤4 overseas) depends only on how many players it has in
// six classes: keeper / bowling option / other × Indian / overseas. Ratings
// only decide WHICH legal XI is best. So "can this team still complete an
// XI, and what is the cheapest way?" is a small exact search over how many
// players of each class to add — always the cheapest ones in each class —
// instead of a search over actual squads.
//
// ── Supply ────────────────────────────────────────────────────────────────
// Players a team can still buy = the rest of the current round (ctx.upcoming)
// plus, during the main round, everyone already unsold (ctx.returning): they
// come back in the re-auction. The player on the block is NOT supply — if it
// isn't bought now it is either gone or only back in the re-auction.
// Prices are base prices: the minimum anyone can pay. That makes completion
// cost a floor, not a forecast; scarcity (below) is how the plan flags when
// that floor is unrealistic.

import { DEFAULT_RULES, XI_SIZE, bidBlocker } from './rules.js'
import { XI_RULES, selectBestXI, xiGain } from './scoring.js'

export const REQUIREMENT_STATUS = Object.freeze({
    SAFE: 'SAFE', // already satisfied by the current squad
    NEED: 'NEED', // missing, plenty of supply
    CRITICAL: 'CRITICAL', // missing, supply barely covers the teams that need it
    IMPOSSIBLE: 'IMPOSSIBLE' // cannot be satisfied any more
})
const { SAFE, NEED, CRITICAL, IMPOSSIBLE } = REQUIREMENT_STATUS
const STATUS_RANK = { SAFE: 0, NEED: 1, CRITICAL: 2, IMPOSSIBLE: 3 }

export const REQUIREMENTS = Object.freeze(['keeper', 'bowling', 'indians', 'players'])

const CLASSES = ['Wi', 'Wo', 'Bi', 'Bo', 'Oi', 'Oo']
const isOverseas = (p) => p.nationality === 'Overseas'
const isBowlingOption = (p) => p.role === 'BOWLER' || p.role === 'ALL ROUNDER'
export const playerClass = (p) =>
    (p.role === 'WICKET KEEPER' ? 'W' : isBowlingOption(p) ? 'B' : 'O') + (isOverseas(p) ? 'o' : 'i')

const zeroCounts = () => ({ Wi: 0, Wo: 0, Bi: 0, Bo: 0, Oi: 0, Oo: 0 })

// Class counts, memoised per squad array (same trick as scoring.xiTotal:
// a squad only changes by push, so the length invalidates the entry).
const countCache = new WeakMap()
export const classCounts = (squad) => {
    if (!squad) return zeroCounts()
    const hit = countCache.get(squad)
    if (hit && hit.length === squad.length) return hit.counts
    const counts = zeroCounts()
    for (const p of squad) if (p) counts[playerClass(p)]++
    countCache.set(squad, { length: squad.length, counts })
    return counts
}

// Most players a legal-as-possible XI can field from these counts — the same
// number selectBestXI fields (11 − emptySlots). Requirements not met leave
// slots empty; at most 4 overseas. Picks keepers/bowlers from Indians first,
// which never hurts.
export const maxXiCount = (c) => {
    let best = 0
    const keepers = c.Wi + c.Wo
    const bowlers = c.Bi + c.Bo
    for (let w = 0; w <= Math.min(XI_RULES.minKeepers, keepers); w++) {
        for (let b = 0; b <= Math.min(XI_RULES.minBowlingOptions, bowlers); b++) {
            const wOs = w && c.Wi === 0 ? 1 : 0
            const bIn = Math.min(b, c.Bi)
            const bOs = b - bIn
            const osUsed = wOs + bOs
            if (osUsed > XI_RULES.maxOverseas) continue
            const indiansLeft = c.Wi + c.Bi + c.Oi - (w - wOs) - bIn
            const overseasLeft = c.Wo + c.Bo + c.Oo - osUsed
            const cap = XI_SIZE - (XI_RULES.minKeepers - w) - (XI_RULES.minBowlingOptions - b)
            const count = Math.min(cap, w + b + indiansLeft + Math.min(overseasLeft, XI_RULES.maxOverseas - osUsed))
            if (count > best) best = count
        }
    }
    return best
}

// ── Team needs (step 2) ───────────────────────────────────────────────────
// How far the current squad is from each XI requirement, in players.
export const teamComposition = (team, rules = DEFAULT_RULES) => {
    const c = classCounts(team.squad)
    const indians = c.Wi + c.Bi + c.Oi
    const overseas = c.Wo + c.Bo + c.Oo
    const overseasSlots = rules.maxOverseas - team.overseasCount
    const xiCount = maxXiCount(c)
    return {
        counts: c,
        squadSize: team.playerCount,
        slotsLeft: rules.maxPlayers - team.playerCount,
        overseasSlots,
        indians,
        overseas,
        keepers: c.Wi + c.Wo,
        bowlingOptions: c.Bi + c.Bo,
        batsmenAndOthers: c.Oi + c.Oo,
        xiCount,
        missing: {
            keeper: c.Wi + c.Wo >= XI_RULES.minKeepers ? 0 : XI_RULES.minKeepers - (c.Wi + c.Wo),
            bowling: Math.max(0, XI_RULES.minBowlingOptions - (c.Bi + c.Bo)),
            // Indians the XI still needs even if every overseas XI place is used
            // (overseas XI places are limited by the 4-in-XI rule and by squad slots left).
            indians: Math.max(0, XI_SIZE - indians - Math.min(XI_RULES.maxOverseas, overseas + Math.max(0, overseasSlots))),
            players: XI_SIZE - xiCount
        }
    }
}

// ── Supply (step 4) ───────────────────────────────────────────────────────
// Per class: base prices ascending (+ prefix sums), and where in the current
// round the first player of that class comes up. Built once per lot and
// shared by every team (cached on the upcoming/returning arrays).
const EMPTY = Object.freeze([])
const supplyCache = new WeakMap()

const buildSupply = (upcoming, returning) => {
    const byClass = Object.fromEntries(CLASSES.map((k) => [k, []]))
    const firstIndex = Object.fromEntries(CLASSES.map((k) => [k, null]))
    upcoming.forEach((p, i) => {
        const k = playerClass(p)
        byClass[k].push(p.basePrice)
        if (firstIndex[k] === null) firstIndex[k] = i
    })
    for (const p of returning) byClass[playerClass(p)].push(p.basePrice)
    const prices = {}
    const prefix = {}
    for (const k of CLASSES) {
        prices[k] = byClass[k].sort((a, b) => a - b)
        prefix[k] = [0]
        for (const price of prices[k]) prefix[k].push(prefix[k].at(-1) + price)
    }
    return { prices, prefix, firstIndex, upcomingCount: upcoming.length, returningCount: returning.length }
}

export const supplyOf = (ctx) => {
    const upcoming = ctx.upcoming || EMPTY
    const returning = ctx.returning || EMPTY
    let inner = supplyCache.get(upcoming)
    if (!inner) supplyCache.set(upcoming, (inner = new WeakMap()))
    let supply = inner.get(returning)
    if (!supply) inner.set(returning, (supply = buildSupply(upcoming, returning)))
    return supply
}

const countAtMost = (sorted, limit) => {
    let lo = 0
    let hi = sorted.length
    while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (sorted[mid] <= limit) lo = mid + 1
        else hi = mid
    }
    return lo
}

// ── Minimum completion cost (steps 6-7) ───────────────────────────────────
// Always the cheapest players of each class, so no player is ever counted
// twice and one player can fill several needs (an Indian bowler covers
// bowling AND Indians).
//
// Cheapest LEGAL XI — exact and fast. A legal XI exists for class totals iff
//   • there is a keeper: an Indian one, or else an overseas one (k = 1 overseas place)
//   • b = max(0, 5 − Indian bowling options) overseas bowling options exist, with k + b ≤ 4
//   • Indians + min(overseas, 4) ≥ 11
// (fill the keeper and bowling places with Indians first, the rest with
// anyone). At most 4 overseas additions can ever play, so the search is 35
// overseas mixes × Indian keeper/bowler counts, with the Indian fillers then
// determined — a few thousand O(1) checks at worst, usually far fewer.
export const legalCompletionCost = (counts, { slots, overseasSlots, supply }) => {
    if (maxXiCount(counts) === XI_SIZE) return { cost: 0, additions: zeroCounts() }
    const { prices, prefix } = supply
    const indiansNow = counts.Wi + counts.Bi + counts.Oi
    const overseasNow = counts.Wo + counts.Bo + counts.Oo
    const osMax = Math.max(0, Math.min(overseasSlots, XI_RULES.maxOverseas, slots))
    let best = Infinity
    let additions = null
    for (let wo = 0; wo <= Math.min(prices.Wo.length, osMax); wo++) {
        for (let bo = 0; bo <= Math.min(prices.Bo.length, osMax - wo); bo++) {
            for (let oo = 0; oo <= Math.min(prices.Oo.length, osMax - wo - bo); oo++) {
                const osCost = prefix.Wo[wo] + prefix.Bo[bo] + prefix.Oo[oo]
                if (osCost >= best) break
                const keepersOs = counts.Wo + wo
                const bowlersOs = counts.Bo + bo
                const usableOs = Math.min(overseasNow + wo + bo + oo, XI_RULES.maxOverseas)
                const indianRoom = Math.min(slots - wo - bo - oo, XI_SIZE)
                for (let wi = 0; wi <= Math.min(prices.Wi.length, indianRoom); wi++) {
                    if (osCost + prefix.Wi[wi] >= best) break
                    const k = counts.Wi + wi >= 1 ? 0 : 1
                    if (k === 1 && keepersOs < 1) continue
                    for (let bi = 0; bi <= Math.min(prices.Bi.length, indianRoom - wi); bi++) {
                        const partial = osCost + prefix.Wi[wi] + prefix.Bi[bi]
                        if (partial >= best) break
                        const b = Math.max(0, XI_RULES.minBowlingOptions - (counts.Bi + bi))
                        if (bowlersOs < b || k + b > XI_RULES.maxOverseas) continue
                        const oi = Math.max(0, XI_SIZE - usableOs - (indiansNow + wi + bi))
                        if (oi > prices.Oi.length || wi + bi + oi > indianRoom) continue
                        const cost = partial + prefix.Oi[oi]
                        if (cost < best) {
                            best = cost
                            additions = { Wi: wi, Wo: wo, Bi: bi, Bo: bo, Oi: oi, Oo: oo }
                        }
                    }
                }
            }
        }
    }
    return { cost: best, additions }
}

// Cheapest way to reach an XI with at most e empty slots, for every e —
// only needed once a legal XI is out of reach (the team then salvages the
// best partial XI it can). Exhaustive over class counts (at most 11
// additions), pruned by `budget`: nothing costing more than the team can pay
// matters. It can take MORE additions than the XI is currently short: a squad
// of four overseas keepers "fields" them, but a legal XI may need those
// overseas places for bowlers instead.
export const completionCosts = (counts, { slots, overseasSlots, supply, budget = Infinity }) => {
    const minCost = new Array(XI_SIZE + 1).fill(Infinity)
    if (maxXiCount(counts) === XI_SIZE) return { minCost: minCost.fill(0) }
    const limit = Math.max(0, Math.min(slots, XI_SIZE))
    const avail = CLASSES.map((k) => supply.prices[k].length)
    const pre = CLASSES.map((k) => supply.prefix[k])
    const base = CLASSES.map((k) => counts[k])
    const add = [0, 0, 0, 0, 0, 0]
    const visit = (i, used, usedOs, cost) => {
        if (i === 6) {
            const empty = XI_SIZE - maxXiCount({
                Wi: base[0] + add[0], Wo: base[1] + add[1], Bi: base[2] + add[2],
                Bo: base[3] + add[3], Oi: base[4] + add[4], Oo: base[5] + add[5]
            })
            if (cost < minCost[empty]) minCost[empty] = cost
            return
        }
        const overseas = i % 2 === 1 // CLASSES alternates Indian / overseas
        const max = Math.min(avail[i], limit - used, overseas ? Math.max(0, overseasSlots - usedOs) : Infinity)
        for (let n = 0; n <= max; n++) {
            const c = cost + pre[i][n]
            if (c > budget) break
            add[i] = n
            visit(i + 1, used + n, usedOs + (overseas ? n : 0), c)
        }
        add[i] = 0
    }
    visit(0, 0, 0, 0)
    // "at most e empty" — carry cheaper solutions from fewer empties upward.
    for (let e = 1; e <= XI_SIZE; e++) minCost[e] = Math.min(minCost[e], minCost[e - 1])
    return { minCost }
}

// Fewest empty XI slots reachable within `budget`.
const reachableEmpty = (minCost, budget) => {
    for (let e = 0; e <= XI_SIZE; e++) if (minCost[e] <= budget) return e
    return XI_SIZE
}

// Step 7 as a standalone answer for a team (no purchase).
export const minimumCostToCompleteXI = (ctx, team = ctx.self) => {
    const rules = ctx.rules || DEFAULT_RULES
    const comp = teamComposition(team, rules)
    const { cost, additions } = legalCompletionCost(comp.counts, {
        slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply: supplyOf(ctx)
    })
    const feasible = cost <= team.purseLeft
    return {
        feasible,
        minimumCost: Number.isFinite(cost) ? cost : null,
        playersNeeded: additions ? Object.values(additions).reduce((s, n) => s + n, 0) : null,
        additions,
        criticalRequirements: REQUIREMENTS.filter((r) => comp.missing[r] > 0)
    }
}

// ── Rival pressure ────────────────────────────────────────────────────────
// How many rivals still lack each requirement — they compete for the same
// supply, so "30 keepers left" means less when 9 teams need one.
const rivalsLacking = (ctx, rules) => {
    const out = { keeper: 0, bowling: 0, indians: 0, players: 0 }
    for (const r of ctx.rivals || EMPTY) {
        const m = teamComposition(r, rules).missing
        for (const k of REQUIREMENTS) if (m[k] > 0) out[k]++
    }
    return out
}

// Players in the supply that could fill each requirement for THIS team
// (affordable at base, overseas only while the team has overseas slots).
const suitableCounts = (supply, purse, overseasSlots) => {
    const n = (k) => (k.endsWith('o') && overseasSlots <= 0 ? 0 : countAtMost(supply.prices[k], purse))
    return {
        keeper: n('Wi') + n('Wo'),
        bowling: n('Bi') + n('Bo'),
        indians: n('Wi') + n('Bi') + n('Oi'),
        players: CLASSES.reduce((s, k) => s + n(k), 0)
    }
}
const requirementClasses = {
    keeper: ['Wi', 'Wo'], bowling: ['Bi', 'Bo'], indians: ['Wi', 'Bi', 'Oi'], players: CLASSES
}
const lotFills = (lot, missing) => {
    const k = playerClass(lot)
    return REQUIREMENTS.filter((r) => missing[r] > 0 && requirementClasses[r].includes(k))
}

const median = (sorted) => (sorted.length ? sorted[sorted.length >> 1] : null)

// Sorted base prices of the players who could fill requirement r (overseas
// ones only if the team can still sign overseas) — per supply, computed once.
const requirementPrices = (supply, r, overseasAllowed) => {
    const key = `${r}:${overseasAllowed}`
    supply.byRequirement ??= {}
    return (supply.byRequirement[key] ??= requirementClasses[r]
        .flatMap((k) => (k.endsWith('o') && !overseasAllowed ? [] : supply.prices[k]))
        .sort((a, b) => a - b))
}

// The player on the block against the rest of the market in his role — the
// same for every team, so computed once per lot.
const lotMarket = (ctx, supply) => {
    const { lot } = ctx
    if (supply.lotMarket?.slNo === lot.slNo) return supply.lotMarket
    let equivalent = 0
    let better = 0
    for (const list of [ctx.upcoming || EMPTY, ctx.returning || EMPTY]) {
        for (const p of list) {
            if (p.role !== lot.role) continue
            if (Math.abs(p.rating - lot.rating) <= 3) equivalent++
            if (p.rating > lot.rating) better++
        }
    }
    const cls = supply.prices[playerClass(lot)]
    supply.lotMarket = { slNo: lot.slNo, equivalent, better, cheaper: countAtMost(cls, lot.basePrice - 1) }
    return supply.lotMarket
}

// ── Bid plan (steps 3, 5, 8-10, 12) ───────────────────────────────────────
// Everything a personality needs to turn its desire into a safe cap.
// `maxSafeBid` is the most the team can pay for the lot while keeping the
// best XI it could otherwise reach — a legal one whenever that is reachable.
// It is NOT a recommendation to bid that much.
export const planBid = (ctx) => {
    const rules = ctx.rules || DEFAULT_RULES
    const { lot, self } = ctx
    const purse = self.purseLeft
    const supply = supplyOf(ctx)
    const comp = teamComposition(self, rules)
    const lotClass = playerClass(lot)
    const lotOverseas = isOverseas(lot)

    // Completion if we let this lot go… (fast exact legal search; the
    // partial-XI search only when a legal XI is already out of reach)
    const skipOpts = { slots: comp.slotsLeft, overseasSlots: comp.overseasSlots, supply }
    const skipLegal = legalCompletionCost(comp.counts, skipOpts)
    let skipEmpty = 0
    let reserveIfSkip = skipLegal.cost
    if (skipLegal.cost > purse) {
        const levels = completionCosts(comp.counts, { ...skipOpts, budget: purse }).minCost
        skipEmpty = reachableEmpty(levels, purse)
        reserveIfSkip = levels[skipEmpty]
    }

    // …and if we buy it (auction rules first: full squad / overseas cap / leader).
    const ruleBlock = bidBlocker({ team: self, lot: { ...lot, currentBidderId: '' }, rules, amount: lot.basePrice })
    let buyCost = () => Infinity
    let buyEmpty = XI_SIZE
    if (!ruleBlock) {
        const after = { ...comp.counts, [lotClass]: comp.counts[lotClass] + 1 }
        const buyOpts = { slots: comp.slotsLeft - 1, overseasSlots: comp.overseasSlots - (lotOverseas ? 1 : 0), supply }
        const buyLegal = legalCompletionCost(after, buyOpts)
        let levels = null
        const allLevels = () => (levels ??= completionCosts(after, { ...buyOpts, budget: purse }).minCost)
        buyCost = (e) => (e === 0 ? buyLegal.cost : allLevels()[e])
        buyEmpty = buyLegal.cost <= purse - lot.basePrice ? 0 : reachableEmpty(allLevels(), purse - lot.basePrice)
    }
    // Buying it (even at base price) would let the team reach FEWER empty
    // slots than it can without it — this lot unlocks a requirement. Then the
    // plan targets that better XI, so the reserve covers the rest of it
    // (buying the last keeper with the last rupee and leaving the 11th place
    // empty would be no gain). Otherwise the target is what's reachable
    // without him, and he may only be bought if that stays reachable.
    const unlocks = buyEmpty < skipEmpty
    const targetEmpty = unlocks ? buyEmpty : skipEmpty
    const reserveIfBought = buyCost(targetEmpty)
    const maxSafeBid = Number.isFinite(reserveIfBought) ? purse - reserveIfBought : 0

    // Requirement criticality and scarcity (with and without this lot).
    const suitable = suitableCounts(supply, purse, comp.overseasSlots)
    const rivals = rivalsLacking(ctx, rules)
    const fills = lotFills(lot, comp.missing)
    const lotSuitableFor = new Set(ruleBlock ? [] : fills)
    // Slot slack: squad places left beyond the signings a legal XI still
    // needs. At zero, one wrong signing makes the XI impossible, so every
    // unmet requirement is CRITICAL however much supply is left.
    const additionsNeeded = skipLegal.additions ? Object.values(skipLegal.additions).reduce((s, n) => s + n, 0) : null
    const noSlotSlack = additionsNeeded !== null && additionsNeeded > 0 && comp.slotsLeft <= additionsNeeded
    const requirements = {}
    for (const r of REQUIREMENTS) {
        const need = comp.missing[r]
        const after = suitable[r]
        const including = after + (lotSuitableFor.has(r) && lot.basePrice <= purse ? 1 : 0)
        // CRITICAL = no slack: what's left after this lot only just covers
        // this team plus every rival who needs the same thing (or less).
        let status
        if (need === 0) status = SAFE
        else if (including < need) status = IMPOSSIBLE
        else if (after <= need + rivals[r] || noSlotSlack) status = CRITICAL
        else status = NEED
        const classes = requirementClasses[r]
        const prices = requirementPrices(supply, r, comp.overseasSlots > 0)
        const next = classes.map((k) => supply.firstIndex[k]).filter((i) => i !== null)
        requirements[r] = {
            status,
            need,
            remainingAfterLot: after,
            realisticRemaining: Math.max(0, after - rivals[r]),
            rivalsNeeding: rivals[r],
            cheapestBase: prices[0] ?? null,
            medianBase: median(prices),
            // Lots until the next suitable player in this round (null = only
            // in the re-auction, or none at all).
            nextInLots: next.length ? Math.min(...next) + 1 : null,
            onlyInReauction: next.length === 0 && prices.length > 0,
            finalOpportunity: lotSuitableFor.has(r) && after < need
        }
    }
    // Whole-XI status: impossible → critical (this lot or scarcity is decisive) → need → safe.
    let completionStatus
    if (targetEmpty > 0) completionStatus = IMPOSSIBLE
    else if (comp.missing.players === 0 && REQUIREMENTS.every((r) => comp.missing[r] === 0)) completionStatus = SAFE
    else if (unlocks || REQUIREMENTS.some((r) => requirements[r].status === CRITICAL)) completionStatus = CRITICAL
    else completionStatus = NEED

    // The current lot against what is still to come in its role.
    const market = lotMarket(ctx, supply)
    const lotContext = {
        phase: ctx.phase || (ctx.progress >= 1 ? 'reauction' : 'main'),
        equivalentRemaining: market.equivalent,
        betterRemaining: market.better,
        cheaperAlternatives: market.cheaper,
        scarce: fills.some((r) => requirements[r].status === CRITICAL),
        finalOpportunity: fills.some((r) => requirements[r].finalOpportunity) || unlocks
    }

    const gain = xiGain(self.squad, lot)
    let reason
    const allowed = !ruleBlock && maxSafeBid >= lot.basePrice
    if (ruleBlock) reason = ruleBlock
    else if (!Number.isFinite(reserveIfBought)) reason = targetEmpty === 0
        ? 'Buying this player would make a legal XI impossible.'
        : 'Buying this player would leave more XI slots empty than passing on him.'
    else if (!allowed) reason = `Can't afford him and still complete the XI (₹${reserveIfBought}L must stay in reserve).`
    else reason = targetEmpty === 0 ? 'Legal XI stays reachable.' : `Best reachable XI keeps ${targetEmpty} empty slot(s).`

    return {
        allowed,
        reason,
        teamNeeds: {
            ...Object.fromEntries(REQUIREMENTS.map((r) => [r, requirements[r].status])),
            completion: completionStatus,
            composition: comp
        },
        requirements,
        scarcity: {
            keepersRemaining: suitable.keeper,
            bowlingOptionsRemaining: suitable.bowling,
            indiansRemaining: suitable.indians,
            playersRemaining: suitable.players,
            returningInReauction: supply.returningCount
        },
        lotContext,
        completion: {
            reachableEmptySlots: targetEmpty,
            feasibleAfterPurchase: Number.isFinite(reserveIfBought) && reserveIfBought + lot.basePrice <= purse,
            minimumCompletionCost: Number.isFinite(reserveIfSkip) ? reserveIfSkip : null,
            minimumCompletionCostAfterPurchase: Number.isFinite(reserveIfBought) ? reserveIfBought : null,
            slotsRequired: comp.missing.players,
            slotsLeft: comp.slotsLeft
        },
        playerImpact: {
            xiGain: gain,
            fillsRequirement: fills,
            criticalRequirement: lotContext.scarce || unlocks,
            unlocksRequirement: unlocks
        },
        budget: {
            purse,
            requiredReserve: Number.isFinite(reserveIfBought) ? reserveIfBought : null,
            reserveIfPassed: Number.isFinite(reserveIfSkip) ? reserveIfSkip : null,
            discretionaryBudget: Number.isFinite(reserveIfSkip) ? Math.max(0, purse - reserveIfSkip) : 0,
            maxSafeBid: Math.max(0, maxSafeBid)
        }
    }
}

// Step 6: "If I buy this player at this price, can I still build a legal XI?"
export const canCompleteXIAfterPurchase = (ctx, price, plan = planBid(ctx)) =>
    plan.completion.minimumCompletionCostAfterPurchase !== null &&
    plan.completion.reachableEmptySlots === 0 &&
    plan.completion.minimumCompletionCostAfterPurchase + price <= ctx.self.purseLeft

// ── Opportunity classification ────────────────────────────────────────────
// What KIND of opportunity the player on the block is for this team — facts,
// not a decision. Personalities turn the category into a price.
//   critical — fills a requirement the XI still lacks (keeper, bowling
//              options, Indians, or an XI place)
//   useful   — lifts the Best XI by ≥ 0.5 points
//   marginal — lifts it by a little (0.05–0.5)
//   depth    — doesn't make the XI but adds real cover, with the XI secure,
//              a slot to spare and money that isn't needed elsewhere
//   none     — anything else
// The 0.05 and 0.5 cut-offs are the ones the rule bots already used ("no XI
// improvement" and "small improvement").
export const XI_GAIN = Object.freeze({ none: 0.05, useful: 0.5 })

// Is `lot` real cover rather than a spare body? Either a backup for an XI
// requirement the squad only just meets, or better than every bench player
// (squad members outside the Best XI) in his role.
const coverFor = (squad, lot, comp) => {
    const cover = []
    const lotIsIndian = !isOverseas(lot)
    if (lot.role === 'WICKET KEEPER' && comp.keepers === XI_RULES.minKeepers) cover.push('keeper')
    if (isBowlingOption(lot) && comp.bowlingOptions === XI_RULES.minBowlingOptions) cover.push('bowling')
    if (lotIsIndian && comp.indians === XI_SIZE - XI_RULES.maxOverseas) cover.push('indians')
    const inXI = new Set(selectBestXI(squad).players.map((p) => p.slNo))
    const bench = squad.filter((p) => !inXI.has(p.slNo) && p.role === lot.role)
    if (bench.every((p) => p.rating < lot.rating)) cover.push('bench')
    return cover
}

export const classifyOpportunity = (ctx, plan = planBid(ctx)) => {
    const { lot, self } = ctx
    const gain = plan.playerImpact.xiGain
    if (!plan.allowed) return { category: 'none', reason: plan.reason, gain }
    // Belonging to a needed class isn't enough — he must actually take an XI
    // place (an overseas player can't fill a gap the 4-overseas limit blocks).
    const fills = plan.playerImpact.fillsRequirement.length > 0 && gain > XI_GAIN.none
    if (fills || plan.playerImpact.unlocksRequirement) {
        return {
            category: 'critical',
            reason: `fills ${plan.playerImpact.fillsRequirement.join(', ') || 'an XI requirement'}`,
            finalOpportunity: plan.lotContext.finalOpportunity,
            gain
        }
    }
    if (gain >= XI_GAIN.useful) return { category: 'useful', reason: `XI +${gain.toFixed(2)}`, gain }
    if (gain > XI_GAIN.none) return { category: 'marginal', reason: `XI +${gain.toFixed(2)}`, gain }
    const secure = plan.teamNeeds.completion === REQUIREMENT_STATUS.SAFE
    const spareSlot = plan.completion.slotsLeft > 1
    const healthy = plan.budget.discretionaryBudget >= 2 * lot.basePrice
    if (secure && spareSlot && healthy) {
        const cover = coverFor(self.squad, lot, plan.teamNeeds.composition)
        if (cover.length) return { category: 'depth', reason: `cover: ${cover.join(', ')}`, cover, gain }
    }
    return { category: 'none', reason: secure ? 'no XI gain and no useful cover' : 'no XI gain', gain }
}

export { STATUS_RANK }
