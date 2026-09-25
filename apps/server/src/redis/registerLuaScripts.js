import redis from './client.js'

// Atomically acquires the expiry lock by flipping timerState to PROCESSING_EXPIRY.
// Returns 1 if lock acquired, 0 if not (already locked, wrong state, or a
// newer timer has replaced the one asking).
// Used by onTimerExpiry, onSkip and solo auto-close so only one of them can
// ever end a lot.
//
// KEYS[1] = room:{roomId}:current
// ARGV[1] = timer token, or '' for "any timer" (skip, auto-close).
//           A timeout only closes the lot if its token is still the current
//           one — a bid that landed just before the deadline issues a new
//           token, so the stale timeout can no longer end the lot.
// ARGV[2] = 'running' to refuse while PAUSED (auto-close must never end a
//           paused lot); '' to allow RUNNING or PAUSED (skip works while paused).
redis.defineCommand('acquireExpiryLock', {
    numberOfKeys: 1,
    lua: `
        local token = ARGV[1] or ''
        local mode = ARGV[2] or ''
        local timerState = redis.call('HGET', KEYS[1], 'timerState')

        if mode == 'running' then
            if timerState ~= 'RUNNING' then return 0 end
        elseif timerState ~= 'RUNNING' and timerState ~= 'PAUSED' then
            return 0
        end

        if token ~= '' and redis.call('HGET', KEYS[1], 'timerToken') ~= token then
            return 0
        end

        redis.call(
            'HSET', KEYS[1],
            'timerState', 'PROCESSING_EXPIRY',
            'timerEndsAt', '',
            'pausedTimeRemaining', '',
            'timerToken', ''
        )
        return 1
    `
})

// Atomically pauses a RUNNING lot. Reading the countdown and writing PAUSED
// in one step means a bid can't land in between (which would leave the
// paused countdown computed from the pre-bid deadline).
// Returns the seconds remaining, or -1 if the lot wasn't RUNNING.
//
// KEYS[1] = room:{roomId}:current
// KEYS[2] = room:{roomId}
// ARGV[1] = now (unix seconds)
redis.defineCommand('pauseTimerAtomic', {
    numberOfKeys: 2,
    lua: `
        if redis.call('HGET', KEYS[1], 'timerState') ~= 'RUNNING' then return -1 end
        local remaining = (tonumber(redis.call('HGET', KEYS[1], 'timerEndsAt')) or 0) - tonumber(ARGV[1])
        if remaining < 0 then remaining = 0 end
        redis.call(
            'HSET', KEYS[1],
            'timerState', 'PAUSED',
            'timerEndsAt', '',
            'pausedTimeRemaining', tostring(remaining),
            'timerToken', ''
        )
        redis.call('HSET', KEYS[2], 'status', 'paused')
        return remaining
    `
})

// KEYS[1] = room:{roomId}:current        → currentBid, currentBidderId, timerState, nationality
// KEYS[2] = room:{roomId}:team:{teamId}  → purseLeft, playerCount, overseasCount
// KEYS[3] = room:{roomId}                → maxPlayers, maxOverseas
// ARGV[1] = teamId
// ARGV[2] = expected bid amount, or '' — bots pass the price they decided
//           to pay; if someone bid first the price has moved and the bid is
//           rejected as stale instead of silently costing an extra increment.
// ARGV[3] = new timerEndsAt (unix seconds)
// ARGV[4] = new timer token
//
// Runs entirely inside Redis as one atomic step — no other command can
// interleave between the read and the write, which is what prevents the
// lost-update race condition two simultaneous bids would otherwise cause.
// Accepting the bid and extending the timer happen in the SAME step, so an
// expiry can never slip in between them and sell the lot at the old price.
//
// The increment ladder must match nextBidAmount in packages/shared/src/rules.js
// (packages/shared/test/rules.test.js checks the two stay in step).
redis.defineCommand('placeBidAtomic', {
    numberOfKeys: 3,
    lua: `
        local currentKey = KEYS[1]
        local teamKey     = KEYS[2]
        local roomKey      = KEYS[3]
        local teamId        = ARGV[1]
        local expectedBid   = ARGV[2] or ''
        local newTimerEndsAt = ARGV[3]
        local newTimerToken  = ARGV[4]

        local timerState = redis.call('HGET', currentKey, 'timerState')
        if timerState ~= 'RUNNING' then
            return cjson.encode({success = false, error = 'Bidding is not currently open.'})
        end

        local currentBidderId = redis.call('HGET', currentKey, 'currentBidderId')
        if currentBidderId == teamId then
            return cjson.encode({success = false, error = 'You are already the highest bidder.'})
        end

        local currentBidRaw = redis.call('HGET', currentKey, 'currentBid')
        if not currentBidRaw then
            return cjson.encode({success = false, error = 'No active auction for this room.'})
        end
        local currentBid = tonumber(currentBidRaw)

        local newBid
        if currentBidderId == '' then
            newBid = currentBid
        else
            local increment
            if currentBid < 200 then
                increment = 10
            elseif currentBid < 300 then
                increment = 20
            else
                increment = 50
            end
            newBid = currentBid + increment
        end

        if expectedBid ~= '' and newBid ~= tonumber(expectedBid) then
            return cjson.encode({success = false, stale = true, error = 'The price moved before your bid landed.'})
        end

        local purseLeftRaw = redis.call('HGET', teamKey, 'purseLeft')
        if not purseLeftRaw then
            return cjson.encode({success = false, error = 'Team not found.'})
        end
        local purseLeft = tonumber(purseLeftRaw)

        if purseLeft < newBid then
            return cjson.encode({success = false, error = 'Insufficient purse for this bid.'})
        end

        local playerCount = tonumber(redis.call('HGET', teamKey, 'playerCount'))
        local maxPlayers   = tonumber(redis.call('HGET', roomKey, 'maxPlayers'))
        if playerCount >= maxPlayers then
            return cjson.encode({success = false, error = 'Your squad is full.'})
        end

        local nationality = redis.call('HGET', currentKey, 'nationality')
        if nationality == 'Overseas' then
            local overseasCount = tonumber(redis.call('HGET', teamKey, 'overseasCount'))
            local maxOverseas    = tonumber(redis.call('HGET', roomKey, 'maxOverseas'))
            if overseasCount >= maxOverseas then
                return cjson.encode({success = false, error = 'Overseas player limit reached.'})
            end
        end

        redis.call(
            'HSET', currentKey,
            'currentBid', tostring(newBid),
            'currentBidderId', teamId,
            'timerEndsAt', newTimerEndsAt,
            'timerToken', newTimerToken
        )

        return cjson.encode({success = true, newBid = newBid})
    `
})