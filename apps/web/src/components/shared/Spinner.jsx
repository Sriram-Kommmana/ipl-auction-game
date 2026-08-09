const Spinner = ({ size = 20, className = '' }) => (
  <div
    className={`inline-block rounded-full border-2 border-ink/20 border-t-brand-red animate-spin ${className}`}
    style={{ width: size, height: size }}
  />
)

export default Spinner