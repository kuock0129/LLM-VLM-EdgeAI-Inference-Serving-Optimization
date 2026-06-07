// Generate unique ID
export const generateId = () => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Convert image file to base64
export const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result)
    reader.onerror = error => reject(error)
  })
}

// Format timestamp
export const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  // Less than 1 minute
  if (diff < 60000) {
    return 'Just now'
  }
  // Less than 1 hour
  if (diff < 3600000) {
    const minutes = Math.floor(diff / 60000)
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`
  }
  // Less than 1 day
  if (diff < 86400000) {
    const hours = Math.floor(diff / 3600000)
    return `${hours} hour${hours > 1 ? 's' : ''} ago`
  }
  // Less than 7 days
  if (diff < 604800000) {
    const days = Math.floor(diff / 86400000)
    return `${days} day${days > 1 ? 's' : ''} ago`
  }
  // Default format
  return date.toLocaleDateString()
}

// Generate conversation title from first message
export const generateTitle = (firstMessage) => {
  if (!firstMessage) return 'New Chat'
  const text = firstMessage.content
  return text.length > 40 ? text.substring(0, 40) + '...' : text
}
