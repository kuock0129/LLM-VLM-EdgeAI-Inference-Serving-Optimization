import { PlusCircle, MessageSquare, Trash2, Cpu } from 'lucide-react'
import { formatTime } from '../utils/helpers'
import config from '../config/config'

function Sidebar({ conversations, currentConversationId, onSelectConversation, onNewConversation, onDeleteConversation }) {
  return (
    <div className="w-64 bg-[#171717] border-r border-gray-800 flex flex-col">
      {/* Header with branding */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-2 bg-gradient-to-br from-edge-primary to-edge-secondary rounded-lg">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-edge-primary to-edge-secondary bg-clip-text text-transparent">
              {config.ui.title}
            </h1>
            <p className="text-xs text-gray-400">{config.ui.credits}</p>
          </div>
        </div>

        <button
          onClick={onNewConversation}
          className="w-full flex items-center gap-2 px-3 py-2 bg-edge-primary hover:bg-edge-primary/80 rounded-lg transition-colors"
        >
          <PlusCircle className="w-5 h-5" />
          <span className="font-medium">New Chat</span>
        </button>
      </div>

      {/* Conversations list */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500 text-sm">
            No conversations yet
          </div>
        ) : (
          <div className="p-2">
            {conversations.map(conv => (
              <div
                key={conv.id}
                className={`group relative flex items-center gap-2 px-3 py-3 mb-1 rounded-lg cursor-pointer transition-colors ${
                  currentConversationId === conv.id
                    ? 'bg-gray-800'
                    : 'hover:bg-gray-800/50'
                }`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0 text-gray-400" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {conv.title}
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatTime(conv.updatedAt)}
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteConversation(conv.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-opacity"
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800 text-xs text-gray-500">
        {config.ui.footer}
      </div>
    </div>
  )
}

export default Sidebar
