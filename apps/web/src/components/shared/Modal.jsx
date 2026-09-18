const Modal = ({
  isOpen,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  danger = false
}) => {
  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 bg-void/80 backdrop-blur-sm flex items-center justify-center z-50 px-4"
      onClick={onCancel}
    >
      <div
        className={`panel max-w-sm w-full ${danger ? 'shadow-brutal' : 'shadow-brutal-bone'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={`flex items-center justify-between px-5 py-2 border-b border-line
          ${danger ? 'bg-red text-bone' : 'bg-raised text-bone/60'}`}
        >
          <span className="font-mono text-[10px] tracking-[0.25em] uppercase">
            {danger ? '⚠ Warning' : 'System'}
          </span>
          <span className="font-jp text-[10px] tracking-[0.3em] opacity-70">
            {danger ? '警告' : '確認'}
          </span>
        </div>
        <div className="p-5">
          {title && <h2 className="font-display text-2xl uppercase tracking-wide text-bone mb-2">{title}</h2>}
          {message && <p className="text-sm text-bone/60 leading-relaxed mb-6">{message}</p>}
          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onCancel}
              className="font-mono text-xs uppercase tracking-[0.18em] text-bone/50 hover:text-bone px-4 py-2 transition-colors"
            >
              {cancelLabel}
            </button>
            <button
              type="button"
              onClick={onConfirm}
              className={`text-base px-5 py-2 ${danger ? 'btn-primary' : 'btn-outline'}`}
            >
              {confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Modal
