export const DB_NAME = "ipl-auction"
export const RESULT_DISPLAY_DURATION = 1500

export const FOUR_DAYS_IN_SECONDS = 4 * 24 * 60 * 60

// Seconds a lot stays open after the last bid.
export const MULTIPLAYER_TIMER_SECONDS = 30
// Solo lots also close early once nobody will bid again (see bots/botManager.js),
// so this is only the backstop for a human who hasn't decided yet.
export const SOLO_TIMER_SECONDS = 10

export const IPL_TEAMS = {
    MI:   'Mumbai Indians',
    CSK:  'Chennai Super Kings',
    RCB:  'Royal Challengers Bengaluru',
    KKR:  'Kolkata Knight Riders',
    DC:   'Delhi Capitals',
    SRH:  'Sunrisers Hyderabad',
    PBKS: 'Punjab Kings',
    RR:   'Rajasthan Royals',
    GT:   'Gujarat Titans',
    LSG:  'Lucknow Super Giants'
}
