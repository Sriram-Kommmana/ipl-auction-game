// apps/web/src/constants/teams.js

// Static reference data for the 10 IPL teams.
// teamId matches the shortcuts used across the backend (team selection,
// Redis keys like room:{roomId}:team:{teamId}, socket payloads, etc.)
//
// color   → primary brand color, used for team cards, badges, borders
// accent  → secondary color, used for gradients/highlights (optional use)

export const TEAMS = [
  {
    teamId: 'MI',
    name: 'Mumbai Indians',
    color: '#004BA0',
    accent: '#D1AB3E'
  },
  {
    teamId: 'CSK',
    name: 'Chennai Super Kings',
    color: '#FFFF3C',
    accent: '#0C2340'
  },
  {
    teamId: 'RCB',
    name: 'Royal Challengers Bengaluru',
    color: '#EC1C24',
    accent: '#000000'
  },
  {
    teamId: 'KKR',
    name: 'Kolkata Knight Riders',
    color: '#3A225D',
    accent: '#B3A123'
  },
  {
    teamId: 'DC',
    name: 'Delhi Capitals',
    color: '#17479E',
    accent: '#EF1C25'
  },
  {
    teamId: 'SRH',
    name: 'Sunrisers Hyderabad',
    color: '#FF822A',
    accent: '#000000'
  },
  {
    teamId: 'PBKS',
    name: 'Punjab Kings',
    color: '#ED1B24',
    accent: '#A7A9AC'
  },
  {
    teamId: 'RR',
    name: 'Rajasthan Royals',
    color: '#EA1A85',
    accent: '#004C93'
  },
  {
    teamId: 'GT',
    name: 'Gujarat Titans',
    color: '#1B2133',
    accent: '#B1974C'
  },
  {
    teamId: 'LSG',
    name: 'Lucknow Super Giants',
    color: '#0057A5',
    accent: '#F2B72C'
  }
]

// Quick lookup map — useful when you have a teamId (from socket events,
// Redis data) and need the full team object without a .find() every time.
// e.g. TEAMS_BY_ID['MI'].name
export const TEAMS_BY_ID = Object.fromEntries(
  TEAMS.map((team) => [team.teamId, team])
)

// Picks black or white text for a given hex background, so team chips stay
// readable on light brand colors (CSK yellow, SRH orange) as well as dark
// ones (GT navy, KKR purple).
const readableTextOn = (hex) => {
  if (!hex) return '#FFFFFF'
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return luminance > 0.6 ? '#0A0A0A' : '#FFFFFF'
}

// Inline style for a team chip/badge: brand color fill, readable text, and
// a thin accent-color underline so very dark brand colors (GT) still read
// against the black UI.
export const teamChipStyle = (team) =>
  team
    ? {
        backgroundColor: team.color,
        color: readableTextOn(team.color),
        textShadow: 'none',
        boxShadow: `inset 0 -2px 0 ${team.accent}`
      }
    : undefined
