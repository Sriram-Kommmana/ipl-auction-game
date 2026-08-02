import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import socket from '../lib/socket'
import { clearSession as clearStoredSession } from '../lib/session'
import { useSessionStore } from '../store/sessionStore'
import { useRoomStore } from '../store/roomStore'
import { useAuctionStore } from '../store/auctionStore'
import { useChatStore } from '../store/chatStore'

/**
 * The single source of truth wiring every server -> client socket event
 * to the correct store action. Should be called ONCE, at a high level
 * (e.g. inside App.jsx), not per-page — otherwise listeners get
 * registered multiple times.
 *
 * NOTE on selector style: every store value/action below is selected
 * individually (e.g. useRoomStore((s) => s.setPlayerStatus)) rather than
 * destructuring the whole store (useRoomStore()). This is deliberate, not
 * just style — action functions are stable references that never change,
 * so selecting only actions means THIS component never re-renders from
 * store changes. Since useSocket() is called once at the App level
 * (wrapping the whole route tree), subscribing to whole stores here would
 * re-render the entire app on every bid, every player online/offline, etc.
 *
 * Two categories of event are NOT written into any of the 4 stores,
 * since they're ephemeral UI notices rather than "room truth":
 *   - managerNotice   (managerDisconnected / managerReconnected / auctionAutoPaused)
 *   - socketError     (every *Error event — bidError, selectTeamError, etc.)
 * Both are returned from this hook as local state, for a banner/Toast
 * component to render. Toast.jsx doesn't exist yet (Phase 5) — until then,
 * socketError just holds the latest message.
 */
export const useSocket = () => {
  const navigate = useNavigate()

  const [managerNotice, setManagerNotice] = useState(null)
  const [socketError, setSocketError] = useState(null)

  const setSession = useSessionStore((s) => s.setSession)
  const setTeamId = useSessionStore((s) => s.setTeamId)
  const clearSessionStore = useSessionStore((s) => s.clearSession)
  const playerId = useSessionStore((s) => s.playerId)

  const setRoomState = useRoomStore((s) => s.setRoomState)
  const setRoomStatus = useRoomStore((s) => s.setRoomStatus)
  const roomId = useRoomStore((s) => s.roomId)
  const applyTeamSelection = useRoomStore((s) => s.applyTeamSelection)
  const setPlayerStatus = useRoomStore((s) => s.setPlayerStatus)
  const upsertPlayerOnline = useRoomStore((s) => s.upsertPlayerOnline)
  const updateTeamAfterPurchase = useRoomStore((s) => s.updateTeamAfterPurchase)

  const setAuctionState = useAuctionStore((s) => s.setAuctionState)
  const startAuction = useAuctionStore((s) => s.startAuction)
  const goToNextPlayer = useAuctionStore((s) => s.goToNextPlayer)
  const applyBid = useAuctionStore((s) => s.applyBid)
  const setTimerRunning = useAuctionStore((s) => s.setTimerRunning)
  const setTimerPaused = useAuctionStore((s) => s.setTimerPaused)
  const setLastResult = useAuctionStore((s) => s.setLastResult)

  const setMessages = useChatStore((s) => s.setMessages)
  const addMessage = useChatStore((s) => s.addMessage)

  useEffect(() => {
    // ---------------------------------------------------------------
    // stateSync — split into focused sync functions per store, per review
    // ---------------------------------------------------------------

    const syncRoom = (data) => {
      setRoomState({
        roomId: data.room.roomId,
        status: data.room.status,
        auctionPhase: data.room.auctionPhase,
        players: data.players,
        teams: data.teams,
        pursePerTeam: data.room.pursePerTeam,
        managerPlayerId: data.room.managerPlayerId,
        maxPlayers: data.room.maxPlayers,
        maxOverseas: data.room.maxOverseas
      })
    }

    const syncAuction = (data) => {
      setAuctionState({
        currentPlayerIndex: data.room.currentPlayerIndex,
        currentPlayer: data.current.iplPlayerId ? {
          slNo: data.current.iplPlayerId,
          playerName: data.current.playerName,
          country: data.current.country,
          nationality: data.current.nationality,
          role: data.current.role,
          basePrice: data.current.basePrice,
          rating: data.current.rating,
          stats: data.current.stats
        } : null,
        currentBid: data.current.currentBid,
        currentBidderId: data.current.currentBidderId,
        timerState: data.current.timerState,
        timerEndsAt: data.current.timerEndsAt,
        pausedTimeRemaining: data.current.pausedTimeRemaining,
        // isReauction has no dedicated field in the snapshot — derived from
        // the authoritative auctionPhase field instead (sticky from here on,
        // see auctionStore.js's goToNextPlayer comment)
        isReauction: data.room.auctionPhase === 'reauction'
      })
    }

    const syncChat = (data) => {
      setMessages(data.chat || [])
    }

    // teamId isn't a separate top-level field on stateSync — find our own
    // entry inside the players[] array that's already sent
    const syncSession = (data) => {
      const me = data.players.find((p) => p.playerId === playerId)
      if (me) setTeamId(me.teamId)
    }

    const onStateSync = (data) => {
      syncRoom(data)
      syncAuction(data)
      syncChat(data)
      syncSession(data)
    }

    const onReconnectError = () => {
      clearStoredSession()
      clearSessionStore()
      navigate('/')
    }

    // ---------------------------------------------------------------
    // Lobby — team selection
    // ---------------------------------------------------------------

    const onTeamSelected = ({ playerId: selectedPlayerId, teamId, teamName, previousTeamId }) => {
      applyTeamSelection(selectedPlayerId, teamId, previousTeamId, teamName)
      if (selectedPlayerId === playerId) {
        setTeamId(teamId)
      }
    }

    // ---------------------------------------------------------------
    // Auction lifecycle
    // ---------------------------------------------------------------

    // auctionStarted's payload doesn't include the room's new status, so we
    // set it explicitly — without this, roomStore.roomStatus stays 'lobby'
    // forever and nothing ever navigates anyone to the Auction page.
    const onAuctionStarted = (data) => {
      startAuction(data)
      setRoomStatus('active')
      navigate(`/auction/${roomId}`)
    }

    const onTimerStarted = ({ timerEndsAt }) => setTimerRunning(timerEndsAt)
    const onTimerPaused = ({ pausedTimeRemaining }) => setTimerPaused(pausedTimeRemaining)
    const onTimerResumed = ({ timerEndsAt }) => setTimerRunning(timerEndsAt)

    const onBidPlaced = ({ teamId, newBid }) => applyBid(teamId, newBid)

    const onPlayerSold = ({ playerName, soldTo, teamName, soldFor, isOverseas }) => {
      updateTeamAfterPurchase(soldTo, soldFor, isOverseas)
      setLastResult({ status: 'sold', playerName, soldTo, teamName, soldFor })
    }

    const onPlayerUnsold = ({ playerName }) => {
      setLastResult({ status: 'unsold', playerName })
    }

    // No store update, no banner — skip is a manager action, not a sale
    // outcome. Left as a no-op hook for now.
    const onPlayerSkipped = () => {}

    const onNextPlayer = (data) => goToNextPlayer(data)

    const onAuctionCompleted = ({ roomId }) => navigate(`/results/${roomId}`)

    // ---------------------------------------------------------------
    // Chat
    // ---------------------------------------------------------------

    const onNewChatMessage = (message) => addMessage(message)

    // ---------------------------------------------------------------
    // Presence
    // ---------------------------------------------------------------

    const onPlayerOnline = ({ playerId: pid, nickname }) => upsertPlayerOnline(pid, nickname)
    const onPlayerOffline = ({ playerId: pid }) => setPlayerStatus(pid, 'offline')

    // ---------------------------------------------------------------
    // Manager disconnect / grace period — ephemeral, not stored
    // ---------------------------------------------------------------

    const onManagerDisconnected = ({ gracePeriodSeconds, message }) =>
      setManagerNotice({ type: 'disconnected', message, gracePeriodSeconds })

    const onManagerReconnected = () => setManagerNotice(null)

    const onAuctionAutoPaused = ({ message }) =>
      setManagerNotice({ type: 'autoPaused', message })

    // ---------------------------------------------------------------
    // Errors — every *Error event shares this one handler (per review)
    // ---------------------------------------------------------------

    const onSocketError = ({ message }) => setSocketError({ message })

    // ---------------------------------------------------------------
    // Register + clean up via a single list — one source of truth,
    // impossible to register an event and forget its .off() (per review)
    // ---------------------------------------------------------------

    const listeners = [
      ['stateSync', onStateSync],
      ['reconnectError', onReconnectError],
      ['teamSelected', onTeamSelected],
      ['selectTeamError', onSocketError],
      ['auctionStarted', onAuctionStarted],
      ['timerStarted', onTimerStarted],
      ['timerPaused', onTimerPaused],
      ['timerResumed', onTimerResumed],
      ['bidPlaced', onBidPlaced],
      ['bidError', onSocketError],
      ['playerSold', onPlayerSold],
      ['playerUnsold', onPlayerUnsold],
      ['playerSkipped', onPlayerSkipped],
      ['nextPlayer', onNextPlayer],
      ['auctionCompleted', onAuctionCompleted],
      ['auctionError', onSocketError],
      ['newChatMessage', onNewChatMessage],
      ['playerOnline', onPlayerOnline],
      ['playerOffline', onPlayerOffline],
      ['managerDisconnected', onManagerDisconnected],
      ['managerReconnected', onManagerReconnected],
      ['auctionAutoPaused', onAuctionAutoPaused],
      ['startAuctionError', onSocketError],
      ['pauseError', onSocketError],
      ['resumeError', onSocketError],
      ['skipError', onSocketError]
    ]

    listeners.forEach(([event, handler]) => socket.on(event, handler))

    return () => {
      listeners.forEach(([event, handler]) => socket.off(event, handler))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigate, playerId, roomId])

  return {
    managerNotice,
    socketError,
    clearSocketError: () => setSocketError(null)
  }
}