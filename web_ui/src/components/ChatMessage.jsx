import { User, Bot } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex gap-4 max-w-3xl ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
          isUser
            ? 'bg-gradient-to-br from-edge-primary to-edge-secondary'
            : 'bg-gray-800'
        }`}>
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>

        {/* Message content */}
        <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
          <div className={`inline-block px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-gradient-to-br from-edge-primary to-edge-secondary'
              : 'bg-gray-800'
          } ${message.error ? 'bg-red-900/50' : ''}`}>
            {/* Images */}
            {message.images && message.images.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {message.images.map((img, idx) => (
                  <img
                    key={idx}
                    src={img}
                    alt={`Upload ${idx + 1}`}
                    className="max-w-xs rounded-lg border border-gray-700"
                  />
                ))}
              </div>
            )}

            {/* Text content */}
            {message.content && (
              <div className="prose prose-invert prose-sm max-w-none">
                {isUser ? (
                  <p className="whitespace-pre-wrap m-0">{message.content}</p>
                ) : (
                  <ReactMarkdown
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        return inline ? (
                          <code className="bg-gray-900 px-1 py-0.5 rounded text-sm" {...props}>
                            {children}
                          </code>
                        ) : (
                          <pre className="bg-gray-900 p-3 rounded-lg overflow-x-auto">
                            <code className={className} {...props}>
                              {children}
                            </code>
                          </pre>
                        )
                      }
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                )}
              </div>
            )}

            {/* Streaming indicator */}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 bg-white/50 ml-1 animate-pulse"></span>
            )}
          </div>

          {/* Timestamp */}
          <div className={`text-xs text-gray-500 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatMessage
