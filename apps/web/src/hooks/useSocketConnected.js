// apps/web/src/hooks/useSocketConnected.js
import { useEffect, useState } from 'react'
import socket from '../lib/socket'

/**
 * socket.connected is a plain property, not reactive — React won't
 * re-render when it flips. This turns it into real state so UI can
 * disable/relabel buttons instead of silently no-op'ing on click.
 */
export const useSocketConnected = () => {
  const [isConnected, setIsConnected] = useState(socket.connected)

  useEffect(() => {
    const onConnect = () => setIsConnected(true)
    const onDisconnect = () => setIsConnected(false)

    socket.on('connect', onConnect)
    socket.on('disconnect', onDisconnect)

    return () => {
      socket.off('connect', onConnect)
      socket.off('disconnect', onDisconnect)
    }
  }, [])

  return isConnected
}