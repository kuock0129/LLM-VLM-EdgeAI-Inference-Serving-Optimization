import { useState, useRef } from 'react'
import { Send, Image as ImageIcon, X, AlertCircle } from 'lucide-react'
import { fileToBase64 } from '../utils/helpers'
import config, { getCurrentModel } from '../config/config'

function ChatInput({ onSend }) {
  const [input, setInput] = useState('')
  const [images, setImages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)

  const currentModel = getCurrentModel()

  const handleImageSelect = async (e) => {
    setError('')
    const files = Array.from(e.target.files)

    // Check if adding these images would exceed the maximum
    if (images.length + files.length > config.ui.maxImagesPerMessage) {
      setError(`Maximum ${config.ui.maxImagesPerMessage} images allowed per message`)
      return
    }

    // Validate file sizes and convert to base64
    const imagePromises = files.map(async (file) => {
      if (!file.type.startsWith('image/')) {
        return null
      }

      // Check file size
      if (file.size > config.ui.maxImageSize) {
        const maxSizeMB = (config.ui.maxImageSize / (1024 * 1024)).toFixed(1)
        setError(`Image ${file.name} is too large. Maximum size: ${maxSizeMB}MB`)
        return null
      }

      const base64 = await fileToBase64(file)
      return base64
    })

    const base64Images = (await Promise.all(imagePromises)).filter(Boolean)
    setImages(prev => [...prev, ...base64Images].slice(0, config.ui.maxImagesPerMessage))
  }

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if ((!input.trim() && images.length === 0) || isLoading) {
      return
    }

    setIsLoading(true)

    try {
      await onSend(input.trim(), images)
      setInput('')
      setImages([])
      textareaRef.current?.focus()
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      {/* Error message */}
      {error && (
        <div className="mb-3 flex items-center gap-2 px-4 py-2 bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Model vision warning */}
      {images.length > 0 && !currentModel.supportsVision && (
        <div className="mb-3 flex items-center gap-2 px-4 py-2 bg-yellow-900/20 border border-yellow-800 rounded-lg text-yellow-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>Your current model doesn't support vision. Images will not be processed.</span>
        </div>
      )}

      {/* Image previews */}
      {images.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {images.map((img, index) => (
            <div key={index} className="relative group">
              <img
                src={img}
                alt={`Preview ${index + 1}`}
                className="w-20 h-20 object-cover rounded-lg border border-gray-700"
              />
              <button
                type="button"
                onClick={() => removeImage(index)}
                className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input container */}
      <div className="flex items-end gap-2 bg-gray-900 rounded-2xl p-2 border border-gray-800 focus-within:border-edge-primary transition-colors">
        {/* Image upload button (only show if enabled in config) */}
        {config.ui.enableImageUpload && (
          <>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-2 hover:bg-gray-800 rounded-lg transition-colors flex-shrink-0"
              disabled={isLoading || images.length >= config.ui.maxImagesPerMessage}
              title={currentModel.supportsVision ? 'Upload images' : 'Vision not supported by current model'}
            >
              <ImageIcon className={`w-5 h-5 ${images.length >= config.ui.maxImagesPerMessage ? 'text-gray-600' : 'text-gray-400'}`} />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageSelect}
              className="hidden"
            />
          </>
        )}

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          rows="1"
          disabled={isLoading}
          className="flex-1 bg-transparent outline-none resize-none max-h-32 text-white placeholder-gray-500"
          style={{
            minHeight: '24px',
            height: 'auto'
          }}
          onInput={(e) => {
            e.target.style.height = 'auto'
            e.target.style.height = e.target.scrollHeight + 'px'
          }}
        />

        {/* Send button */}
        <button
          type="submit"
          disabled={(!input.trim() && images.length === 0) || isLoading}
          className="p-2 bg-edge-primary hover:bg-edge-primary/80 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors flex-shrink-0"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>

      {/* Helper text */}
      <div className="mt-2 text-xs text-gray-500 text-center">
        {config.ui.footer} • Using {currentModel.displayName}
      </div>
    </form>
  )
}

export default ChatInput
