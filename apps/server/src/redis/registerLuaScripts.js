import redis from './client.js'

// Atomically acquires the expiry lock by flipping timerState to PROCESSING_EXPIRY.
// Returns 1 if lock acquired (timerState was RUNNING or PAUSED).
// Returns 0 if not acquired (already locked or wrong state).
// Used by both onTimerExpiry and onSkip to prevent double-advance race condition.
// KEYS[1] = room:{roomId}:current
redis.defineCommand('acquireExpiryLock', {
    numberOfKeys: 1,
    lua: `
        local timerState = redis.call('HGET', KEYS[1], 'timerState')
        if timerState == 'RUNNING' or timerState == 'PAUSED' then
            redis.call(
                'HSET', KEYS[1],
                'timerState', 'PROCESSING_EXPIRY',
                'timerEndsAt', '',
                'pausedTimeRemaining', ''
            )
            return 1
        end
        return 0
    `
})

// KEYS[1] = room:{roomId}:current        → currentBid, currentBidderId, timerState, nationality
// KEYS[2] = room:{roomId}:team:{teamId}  → purseLeft, playerCount, overseasCount
// KEYS[3] = room:{roomId}                → maxPlayers, maxOverseas
// ARGV[1] = teamId
//
// Runs entirely inside Redis as one atomic step — no other command can
// interleave between the read and the write, which is what prevents the
// lost-update race condition two simultaneous bids would otherwise cause.
redis.defineCommand('placeBidAtomic', {
    numberOfKeys: 3,
    lua: `
        local currentKey = KEYS[1]
        local teamKey     = KEYS[2]
        local roomKey      = KEYS[3]
        local teamId        = ARGV[1]

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

        redis.call('HSET', currentKey, 'currentBid', tostring(newBid), 'currentBidderId', teamId)

        return cjson.encode({success = true, newBid = newBid})
    `
})