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
      className="fixed inset-0 bg-ink/60 flex items-center justify-center z-50 px-4"
      onClick={onCancel}
    >
      <div
        className="bg-paper rounded-2xl p-6 max-w-sm w-full"
        onClick={(e) => e.stopPropagation()}
      >
        {title && <h2 className="font-display text-xl text-ink mb-2">{title}</h2>}
        {message && <p className="text-sm text-ink/60 mb-5">{message}</p>}
        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="text-sm text-ink/50 hover:text-ink px-4 py-2 transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`font-display text-sm px-4 py-2 rounded-lg transition-colors text-paper
              ${danger ? 'bg-brand-red hover:bg-brand-red-dark' : 'bg-ink hover:bg-charcoal'}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export default Modal