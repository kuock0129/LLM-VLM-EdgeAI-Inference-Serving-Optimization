# Edge AI - Chat Interface

A modern, ChatGPT-like web interface for interacting with AI models running on Jetson Orin Nano. Built by Sasha & Kevin.

## Features

- **Modern Chat Interface**: Clean, responsive UI similar to ChatGPT
- **Flexible Model Support**: Easy configuration for any quantized llama.cpp model
- **Conversation History**: Manage multiple conversations with automatic saving
- **Text & Image Input**: Send text messages and upload images for vision-based analysis
- **Streaming Responses**: Real-time token-by-token response streaming
- **Local Processing**: All AI processing happens locally on your Jetson Orin Nano
- **Markdown Support**: Rendered markdown in AI responses with code highlighting
- **Dark Theme**: Beautiful dark mode interface optimized for extended use

## Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Markdown**: React Markdown
- **Backend**: llama.cpp server (OpenAI-compatible API)
- **Hardware**: Jetson Orin Nano

## Prerequisites

1. **Jetson Orin Nano** with llama.cpp server running
2. **Node.js** (v18 or higher)
3. **npm** or **yarn**

## Using Your Own Model

**This UI works with any llama.cpp compatible model!**

To configure your custom quantized model:

1. Edit `src/config/config.js`
2. Add your model to the `modelPresets` object
3. Set `activeModelPreset` to your model's key
4. Adjust generation parameters as needed

**Quick Example:**
```javascript
// In src/config/config.js
export const modelPresets = {
  'my-model': {
    name: 'my-quantized-llama-q4',
    displayName: 'My Custom Model (Q4)',
    supportsVision: false,
    temperature: 0.7,
    maxTokens: 2048,
  }
}

export const activeModelPreset = 'my-model'  // Activate your model
```

**📖 For detailed instructions, see [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md)**

## Backend Setup

First, ensure your llama.cpp server is running on the Jetson Orin Nano:

```bash
DISPLAY= jetson-containers run dustynv/llama_cpp:r36.4.0 \
  llama-server \
  --hf-repo unsloth/Llama-3.2-3B-Instruct-GGUF \
  --hf-file Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --gpu-layers 34 \
  --host 0.0.0.0 \
  --port 8080 \
  --chat-template llama3
```

The server should be accessible at `http://<jetson-ip>:8080`

### For Vision/Multimodal Support

If you want to use image upload features, you'll need a vision-capable model with the multimodal projector:

```bash
DISPLAY= jetson-containers run dustynv/llama_cpp:r36.4.0 \
  llama-server \
  --hf-repo <vision-model-repo> \
  --hf-file <vision-model-file> \
  --mmproj <path-to-projector> \
  --gpu-layers 34 \
  --host 0.0.0.0 \
  --port 8080 \
  --chat-template llama3
```

## Frontend Setup

1. **Navigate to the web_ui directory**:
   ```bash
   cd web_ui
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure the backend URL** (if not using localhost):

   Edit `vite.config.js` and update the proxy target:
   ```javascript
   proxy: {
     '/v1': {
       target: 'http://<jetson-ip>:8080',  // Replace with your Jetson IP
       changeOrigin: true,
     }
   }
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:3000
   ```

## Production Build

To create a production build:

```bash
npm run build
```

The built files will be in the `dist/` directory. You can serve them with any static file server.

To preview the production build:
```bash
npm run preview
```

## API Integration

The UI communicates with the llama.cpp server using the OpenAI-compatible API:

### Endpoint
```
POST /v1/chat/completions
```

### Request Format
```javascript
{
  "model": "llama-3.2-3b-instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Hello!" },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### Response Format (Streaming)
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]
```

## Usage

### Starting a New Conversation
- Click the "New Chat" button in the sidebar
- Or start typing in the input box when no conversation is active

### Sending Messages
- Type your message in the input box
- Press Enter to send (Shift+Enter for new line)
- The AI response will stream in real-time

### Uploading Images
- Click the image icon in the input box
- Select one or more images
- Images will be previewed before sending
- Remove images by clicking the X on the preview
- Send the message with or without text

### Managing Conversations
- Click on any conversation in the sidebar to switch to it
- Hover over a conversation and click the trash icon to delete it
- Conversations are automatically saved in your browser's local storage

## File Structure

```
web_ui/
├── public/
├── src/
│   ├── components/
│   │   ├── ChatArea.jsx       # Main chat interface
│   │   ├── ChatInput.jsx      # Input box with image upload
│   │   ├── ChatMessage.jsx    # Individual message display
│   │   └── Sidebar.jsx        # Conversation history sidebar
│   ├── utils/
│   │   └── helpers.js         # Utility functions
│   ├── App.jsx                # Root component
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── index.html
├── package.json
├── vite.config.js             # Vite configuration
├── tailwind.config.js         # Tailwind CSS configuration
└── README.md
```

## Customization

### Changing Colors
Edit `tailwind.config.js` to customize the color scheme:
```javascript
theme: {
  extend: {
    colors: {
      'edge-primary': '#3b82f6',    // Primary color
      'edge-secondary': '#8b5cf6',  // Secondary color
    }
  },
}
```

### Adjusting API Parameters
Edit the `streamChatCompletion` function in `ChatArea.jsx`:
```javascript
{
  model: 'llama-3.2-3b-instruct',
  stream: true,
  temperature: 0.7,      // Adjust creativity (0.0 - 2.0)
  max_tokens: 2048,      // Maximum response length
}
```

### Changing Branding
Update the branding in `Sidebar.jsx`:
```javascript
<h1 className="...">Edge AI</h1>
<p className="...">by Your Name</p>
```

## Troubleshooting

### CORS Errors
If you encounter CORS errors, ensure the llama.cpp server is configured to allow requests from your frontend origin, or use the Vite proxy configuration.

### Images Not Working
1. Verify your model supports vision (has `--mmproj` flag)
2. Check that images are being converted to base64 correctly
3. Check browser console for errors

### API Connection Issues
1. Verify the backend URL in `vite.config.js`
2. Ensure the llama.cpp server is running and accessible
3. Check network connectivity to the Jetson device
4. Test the API endpoint directly: `curl http://<jetson-ip>:8080/v1/models`

### Streaming Not Working
1. Check browser console for errors
2. Verify the API response includes `Content-Type: text/event-stream`
3. Test streaming with curl:
   ```bash
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"Hello"}],"stream":true}'
   ```

## Performance Tips

1. **Optimize Model**: Use quantized models (Q4_K_M) for better performance
2. **GPU Layers**: Adjust `--gpu-layers` based on available GPU memory
3. **Context Length**: Smaller context windows (`n_ctx`) reduce memory usage
4. **Batch Size**: Adjust `n_batch` for optimal throughput

## License

MIT

## Credits

**Developed by**: Sasha & Kevin
**Hardware**: NVIDIA Jetson Orin Nano
**Backend**: [llama.cpp](https://github.com/ggml-org/llama.cpp)
**Model**: Llama 3.2 3B Instruct (Unsloth GGUF)

## Contributing

Feel free to submit issues and pull requests!
