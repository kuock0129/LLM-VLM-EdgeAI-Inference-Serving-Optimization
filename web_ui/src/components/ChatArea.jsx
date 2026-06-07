import { useRef, useEffect } from 'react'
import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'
import { Sparkles } from 'lucide-react'
import config, { getCurrentModel } from '../config/config'

function ChatArea({ conversation, onUpdateConversation, onNewConversation }) {
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [conversation?.messages])

  const handleSendMessage = async (content, images) => {
    if (!conversation) {
      const newConv = onNewConversation()
      await handleSendMessageToConversation(newConv, content, images)
    } else {
      await handleSendMessageToConversation(conversation, content, images)
    }
  }

  const handleSendMessageToConversation = async (conv, content, images) => {
    // Create user message
    const userMessage = {
      role: 'user',
      content,
      images: images || [],
      timestamp: Date.now()
    }

    // Add user message to conversation
    const updatedMessages = [...conv.messages, userMessage]
    onUpdateConversation(conv.id, {
      messages: updatedMessages,
      title: conv.messages.length === 0 ? (content.length > 40 ? content.substring(0, 40) + '...' : content) : conv.title
    })

    // Create assistant message placeholder
    const assistantMessage = {
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true
    }

    onUpdateConversation(conv.id, {
      messages: [...updatedMessages, assistantMessage]
    })

    // Call API
    try {
      await streamChatCompletion(conv.id, updatedMessages, assistantMessage)
    } catch (error) {
      console.error('Error calling API:', error)
      onUpdateConversation(conv.id, {
        messages: [...updatedMessages, {
          ...assistantMessage,
          content: `Error: ${error.message}`,
          isStreaming: false,
          error: true
        }]
      })
    }
  }

  const streamChatCompletion = async (conversationId, messages, assistantMessage) => {
    // Get current model configuration
    const currentModel = getCurrentModel()

    // Format messages for API
    const apiMessages = messages.map(msg => {
      const content = []

      // Add text
      if (msg.content) {
        content.push({
          type: 'text',
          text: msg.content
        })
      }

      // Add images (only if model supports vision)
      if (msg.images && msg.images.length > 0 && currentModel.supportsVision) {
        msg.images.forEach(img => {
          content.push({
            type: 'image_url',
            image_url: {
              url: img
            }
          })
        })
      }

      return {
        role: msg.role,
        content: content.length === 1 && content[0].type === 'text' ? content[0].text : content
      }
    })

    // Build request body with configuration
    const requestBody = {
      model: currentModel.name,
      messages: apiMessages,
      stream: config.generation.stream,
      temperature: config.generation.temperature,
      max_tokens: config.generation.maxTokens,
    }

    // Add optional parameters if they're set
    if (config.generation.topP !== 1.0) {
      requestBody.top_p = config.generation.topP
    }
    if (config.generation.topK !== 40) {
      requestBody.top_k = config.generation.topK
    }
    if (config.generation.frequencyPenalty !== 0.0) {
      requestBody.frequency_penalty = config.generation.frequencyPenalty
    }
    if (config.generation.presencePenalty !== 0.0) {
      requestBody.presence_penalty = config.generation.presencePenalty
    }
    if (config.generation.stop.length > 0) {
      requestBody.stop = config.generation.stop
    }

    const response = await fetch(`${config.api.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullContent = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split('\n').filter(line => line.trim() !== '')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            const content = parsed.choices[0]?.delta?.content
            if (content) {
              fullContent += content
              onUpdateConversation(conversationId, {
                messages: [...messages, {
                  ...assistantMessage,
                  content: fullContent,
                  isStreaming: true
                }]
              })
            }
          } catch (e) {
            console.error('Error parsing chunk:', e)
          }
        }
      }
    }

    // Mark streaming as complete
    onUpdateConversation(conversationId, {
      messages: [...messages, {
        ...assistantMessage,
        content: fullContent,
        isStreaming: false
      }]
    })
  }

  if (!conversation) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 bg-[#0f0f0f]">
        <div className="max-w-2xl w-full text-center">
          <div className="mb-6 inline-flex p-4 bg-gradient-to-br from-edge-primary to-edge-secondary rounded-2xl">
            <Sparkles className="w-12 h-12" />
          </div>
          <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-edge-primary to-edge-secondary bg-clip-text text-transparent">
            Welcome to Edge AI
          </h2>
          <p className="text-gray-400 mb-8">
            Start a conversation with your local AI model running on Jetson Orin Nano
          </p>
          <div className="space-y-3 text-left bg-gray-900/50 rounded-xl p-6">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-edge-primary mt-2"></div>
              <p className="text-gray-300">Send text messages and get intelligent responses</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-edge-primary mt-2"></div>
              <p className="text-gray-300">Upload images for vision-based analysis</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-edge-primary mt-2"></div>
              <p className="text-gray-300">All processing happens locally on your device</p>
            </div>
          </div>
        </div>
        <div className="absolute bottom-0 left-64 right-0 p-4">
          <ChatInput onSend={handleSendMessage} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col bg-[#0f0f0f]">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto p-4 space-y-4">
          {conversation.messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="border-t border-gray-800 p-4">
        <div className="max-w-4xl mx-auto">
          <ChatInput onSend={handleSendMessage} />
        </div>
      </div>
    </div>
  )
}

export default ChatArea
