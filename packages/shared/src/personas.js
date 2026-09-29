// Who sits in the nine bot seats of a solo room.
//
// 4 rule-based personalities (hand-written logic, ruleBots.js) and
// 5 RL personalities, each played by its own trained model (the server's
// bots/models/registry.json). An RL persona's id, vector and fallback are
// training identifiers and stay fixed; its name and blurb describe how its
// model actually bids in the gameplay audit (ml/ipl_rl/gameplay/README.md),
// so they change if the roster does. The spread of personalities follows
// Joglekar et al. 2025 (Journal of Sports Analytics), which found distinct
// clusters of real IPL auction strategies — mainly how concentrated or
// spread out a franchise's spending is.

export const RULE_PERSONAS = Object.freeze({
    moneyball: {
        id: 'moneyball',
        name: 'Moneyball',
        blurb: 'Pays for value, walks away when a cheaper substitute is still to come.'
    },
    starChaser: {
        id: 'starChaser',
        name: 'Star Chaser',
        blurb: 'Spends big and early on marquee players.'
    },
    balancedBuilder: {
        id: 'balancedBuilder',
        name: 'Balanced Builder',
        blurb: 'Budgets by position — fills every role before chasing names.'
    },
    opportunist: {
        id: 'opportunist',
        name: 'The Opportunist',
        blurb: 'Waits early, pounces late when rivals have run low on money.'
    }
})

// vector order: [aggression, bowlingFocus, battingFocus, overseasFocus, starFocus]
export const RL_PERSONAS = Object.freeze({
    aggressor: {
        id: 'aggressor',
        name: 'Aggressor',
        vector: [1, 0, 0, 0, 0.5],
        fallback: 'starChaser',
        blurb: 'Hates losing a bidding war.'
    },
    paceFirst: {
        id: 'paceFirst',
        name: 'Bargain Hunter',
        vector: [0.3, 1, 0, 0, 0],
        fallback: 'balancedBuilder',
        blurb: 'Sits out the early frenzy, then buys stars cheap once rivals run dry.'
    },
    battingFirst: {
        id: 'battingFirst',
        name: 'Fast Starter',
        vector: [0.3, 0, 1, 0, 0],
        fallback: 'balancedBuilder',
        blurb: 'Goes hard from the first lot and loves an all-rounder.'
    },
    overseasSpecialist: {
        id: 'overseasSpecialist',
        name: 'Price Pusher',
        vector: [0.3, 0, 0, 1, 0],
        fallback: 'opportunist',
        blurb: 'Bids on almost every lot — nobody gets a player cheap past it.'
    },
    adaptive: {
        id: 'adaptive',
        name: 'Adaptive',
        vector: [0, 0, 0, 0, 0],
        fallback: 'moneyball',
        blurb: 'No fixed plan — reads the room.'
    }
})

// The nine seats of a solo room: 4 rule bots + 5 RL bots.
export const SOLO_BOT_LINEUP = Object.freeze([
    ...Object.keys(RULE_PERSONAS).map((id) => ({ kind: 'rule', persona: id })),
    ...Object.keys(RL_PERSONAS).map((id) => ({ kind: 'rl', persona: id }))
])

export const botDisplayName = (seat) =>
    seat.kind === 'rule' ? RULE_PERSONAS[seat.persona]?.name : RL_PERSONAS[seat.persona]?.name
