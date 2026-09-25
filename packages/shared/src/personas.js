// Who sits in the nine bot seats of a solo room.
//
// 4 rule-based personalities (hand-written logic, ruleBots.js) and
// 5 RL personalities that share ONE trained policy but receive a different
// persona vector — the same vector that weighted their reward in training
// (see PERSONA_DIMS in observation.js). The spread of personalities follows
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
        name: 'Pace Factory',
        vector: [0.3, 1, 0, 0, 0],
        fallback: 'balancedBuilder',
        blurb: 'Wins matches with the ball.'
    },
    battingFirst: {
        id: 'battingFirst',
        name: 'Run Machine',
        vector: [0.3, 0, 1, 0, 0],
        fallback: 'balancedBuilder',
        blurb: 'Stacks the batting order.'
    },
    overseasSpecialist: {
        id: 'overseasSpecialist',
        name: 'Global Scout',
        vector: [0.3, 0, 0, 1, 0],
        fallback: 'opportunist',
        blurb: 'Hunts the best overseas talent.'
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
