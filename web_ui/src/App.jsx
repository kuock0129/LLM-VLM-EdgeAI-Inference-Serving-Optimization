import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import { generateId } from './utils/helpers'

function App() {
  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem('edge-ai-conversations')
    return saved ? JSON.parse(saved) : []
  })

  const [currentConversationId, setCurrentConversationId] = useState(() => {
    const saved = localStorage.getItem('edge-ai-current-conversation')
    return saved || null
  })

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('edge-ai-conversations', JSON.stringify(conversations))
  }, [conversations])

  useEffect(() => {
    if (currentConversationId) {
      localStorage.setItem('edge-ai-current-conversation', currentConversationId)
    }
  }, [currentConversationId])

  const createNewConversation = () => {
    const newConversation = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
    setConversations(prev => [newConversation, ...prev])
    setCurrentConversationId(newConversation.id)
    return newConversation
  }

  const updateConversation = (id, updates) => {
    setConversations(prev =>
      prev.map(conv =>
        conv.id === id
          ? { ...conv, ...updates, updatedAt: Date.now() }
          : conv
      )
    )
  }

  const deleteConversation = (id) => {
    setConversations(prev => prev.filter(conv => conv.id !== id))
    if (currentConversationId === id) {
      setCurrentConversationId(null)
    }
  }

  const currentConversation = conversations.find(c => c.id === currentConversationId)

  return (
    <div className="flex h-screen bg-[#0f0f0f] text-white">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewConversation={createNewConversation}
        onDeleteConversation={deleteConversation}
      />
      <ChatArea
        conversation={currentConversation}
        onUpdateConversation={updateConversation}
        onNewConversation={createNewConversation}
      />
    </div>
  )
}

export default App
