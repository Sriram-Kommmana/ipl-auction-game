// Public surface of @ipl-auction/shared.
//
// Browser-safe: nothing here touches Node APIs, so the web app imports the
// same scoring and bid rules the server enforces.

export * from './rules.js'
export * from './scoring.js'
export * from './valuation.js'
export * from './pool.js'
export * from './observation.js'
export * from './personas.js'
export * from './ruleBots.js'
export * from './mlp.js'
export * from './rewards.js'
