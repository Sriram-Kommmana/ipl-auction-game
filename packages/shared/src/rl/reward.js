// The RL reward (Phase 2A, frozen).
//
//   r_t = ΔBestXI_total / 11 / 10      (unrounded XI total; γ = 1)
//   terminal: −2 if the final Best XI has any empty slot
//
// A squad's Best XI total never decreases as players are added, so with
// γ = 1 the per-step rewards sum exactly to the final XI (/110) — a dense
// signal with no bias toward buying early. Nothing else is rewarded: no
// spending, leftover purse, rank, style or relative-to-rivals terms.

import { XI_SIZE } from '../rules.js'
import { selectBestXI, xiTotal } from '../scoring.js'

export const GAMMA = 1.0
export const LAMBDA_REL = 0 // competitive term: off in v1 (future ablation only)
export const XI_REWARD_SCALE = XI_SIZE * 10
export const EMPTY_SLOT_PENALTY = -2

export const xiPotential = (squad) => xiTotal(squad) / XI_REWARD_SCALE

export const stepReward = (potentialBefore, squadAfter) => xiPotential(squadAfter) - potentialBefore

export const terminalReward = (squad) => (selectBestXI(squad).emptySlots > 0 ? EMPTY_SLOT_PENALTY : 0)

// What the episode's rewards must sum to (parity checks).
export const episodeReturn = (finalSquad) => xiPotential(finalSquad) + terminalReward(finalSquad)
