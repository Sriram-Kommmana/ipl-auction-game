const VARIANTS = {
  default: 'border-ink/20 border-t-brand-red',
  light: 'border-paper/30 border-t-paper'
}

const Spinner = ({ size = 20, variant = 'default', className = '' }) => (
  <div
    className={`inline-block rounded-full animate-spin border-2 ${VARIANTS[variant]} ${className}`}
    style={{ width: size, height: size }}
  />
)

export default Spinner