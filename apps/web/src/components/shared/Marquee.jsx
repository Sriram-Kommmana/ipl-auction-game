import { useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react'

// Single-line text that scrolls like a ticker — but only when it doesn't fit.
// Text that fits sits still. Each loop rests at the start (so the beginning is
// readable), scrolls at a constant speed, then wraps round seamlessly using a
// second copy. With "reduce motion" on it never moves: it truncates with "…"
// and the full text is in the tooltip.

const SPEED_PX_PER_SEC = 45
const REST_MS = 1400
const GAP_PX = 48

const reducedMotionQuery = '(prefers-reduced-motion: reduce)'
const subscribeReducedMotion = (onChange) => {
  const mq = window.matchMedia(reducedMotionQuery)
  mq.addEventListener('change', onChange)
  return () => mq.removeEventListener('change', onChange)
}
const getReducedMotion = () => window.matchMedia(reducedMotionQuery).matches

const Marquee = ({ text, className = '' }) => {
  const boxRef = useRef(null)
  const textRef = useRef(null)
  const trackRef = useRef(null)
  const [distance, setDistance] = useState(0) // 0 = fits, no scrolling
  const reducedMotion = useSyncExternalStore(subscribeReducedMotion, getReducedMotion, () => false)

  // Re-measure whenever the box or the text changes size (resizes, web font arriving).
  useLayoutEffect(() => {
    const box = boxRef.current
    const label = textRef.current
    if (!box || !label) return
    const measure = () => {
      const overflow = label.offsetWidth > box.clientWidth + 1
      setDistance(overflow ? label.offsetWidth + GAP_PX : 0)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(box)
    observer.observe(label)
    return () => observer.disconnect()
  }, [text])

  const scrolling = distance > 0 && !reducedMotion

  useLayoutEffect(() => {
    const track = trackRef.current
    if (!scrolling || !track) return
    const travelMs = (distance / SPEED_PX_PER_SEC) * 1000
    const total = REST_MS + travelMs
    const animation = track.animate(
      [
        { transform: 'translateX(0)', offset: 0 },
        { transform: 'translateX(0)', offset: REST_MS / total },
        { transform: `translateX(-${distance}px)`, offset: 1 }
      ],
      { duration: total, iterations: Infinity, easing: 'linear' }
    )
    return () => animation.cancel()
  }, [scrolling, distance])

  // Not scrolling: the text is plain inline content, so the box's ellipsis
  // applies (only visible with reduce-motion on). Scrolling: the track is one
  // flex row sliding inside the clipped box.
  return (
    <span
      ref={boxRef}
      title={distance > 0 ? text : undefined}
      className={`block overflow-hidden whitespace-nowrap ${scrolling ? '' : 'text-ellipsis'} ${className}`}
      style={scrolling ? { maskImage: 'linear-gradient(to right, #000 88%, transparent)' } : undefined}
    >
      <span ref={trackRef} className={scrolling ? 'inline-flex will-change-transform' : ''} style={scrolling ? { gap: GAP_PX } : undefined}>
        <span ref={textRef}>{text}</span>
        {scrolling && <span aria-hidden="true">{text}</span>}
      </span>
    </span>
  )
}

export default Marquee
