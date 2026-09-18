const VARIANTS = {
  default: 'border-bone/15 border-t-red',
  light: 'border-bone/30 border-t-bone'
}

const Spinner = ({ size = 20, variant = 'default', className = '' }) => (
  <div
    className={`inline-block rounded-full animate-spin border-2 ${VARIANTS[variant]} ${className}`}
    style={{ width: size, height: size }}
  />
)

export default Spinner
