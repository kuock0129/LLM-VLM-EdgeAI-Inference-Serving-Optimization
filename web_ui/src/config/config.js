// =============================================================================
// Edge AI Chat UI - Configuration File
// =============================================================================
// This file controls all aspects of the chat UI behavior and appearance.
// Modify these settings to match your custom model and backend setup.
//
// IMPORTANT: Also update vite.config.js with your backend URL!
//
// Quick Start:
//   1. Update vite.config.js: Set BACKEND_URL to your server address
//   2. Update model.name below to match your model identifier
//   3. Adjust generation parameters (temperature, maxTokens, etc.)
//   4. Customize UI branding (title, credits, footer)
// =============================================================================

export const config = {
  // Backend API Configuration
  api: {
    // Base URL for the llama.cpp server
    // For development, this is proxied through Vite (see vite.config.js)
    // For production, set this to your Jetson IP address
    baseUrl: '/v1',

    // Timeout for API requests (milliseconds)
    timeout: 120000, // 2 minutes
  },

  // Model Configuration
  // ==========================================================================
  // Configure this section to match your quantized model
  // ==========================================================================
  model: {
    // Model identifier sent to backend API
    // For llama.cpp, this can be any string
    name: 'Llama-3.2-3B-Instruct',

    // Display name shown in the UI footer
    displayName: 'Llama 3.2 3B Instruct Q4_K_M',

    // Set to true if your model supports vision/image input
    // Llama 3.2 3B Instruct is text-only
    supportsVision: false,

    // Set to true if your model supports function calling
    supportsFunctions: false,
  },

  // Generation Parameters
  // These can be adjusted based on your model's capabilities
  generation: {
    // Temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
    temperature: 0.7,

    // Maximum tokens to generate in response
    maxTokens: 2048,

    // Top-p (nucleus sampling): Only consider tokens with cumulative probability p
    topP: 1.0,

    // Top-k: Only consider top k most likely tokens
    topK: 40,

    // Frequency penalty: Reduce likelihood of repeating tokens
    frequencyPenalty: 0.0,

    // Presence penalty: Reduce likelihood of repeating topics
    presencePenalty: 0.0,

    // Enable streaming responses (recommended)
    stream: true,

    // Stop sequences (tokens that end generation)
    stop: [],
  },

  // UI Configuration
  ui: {
    // Application title
    title: 'Edge AI',

    // Credits/authors
    credits: 'by Sasha & Kevin',

    // Footer text
    footer: 'Powered by Jetson Orin Nano',

    // Show model selector in UI (allows switching models at runtime)
    showModelSelector: false,

    // Enable image upload button
    enableImageUpload: true,

    // Maximum number of images per message
    maxImagesPerMessage: 5,

    // Maximum image file size (bytes) - 5MB default
    maxImageSize: 5 * 1024 * 1024,
  },

  // Advanced Settings
  advanced: {
    // Retry failed requests
    retryOnError: true,
    maxRetries: 3,

    // Save conversations to localStorage
    saveConversations: true,

    // Maximum number of saved conversations
    maxSavedConversations: 100,

    // Auto-generate conversation titles
    autoGenerateTitles: true,
  }
}

// =============================================================================
// Multiple Model Configurations (Optional)
// =============================================================================
// Define multiple model presets and easily switch between them by changing
// the activeModelPreset variable below.
//
// This is useful if you want to test different models without changing code.
// =============================================================================

export const modelPresets = {
  // Llama.cpp - Llama 3.2 3B Instruct Q4_K_M (Currently Active)
  'llama-3.2-3b': {
    name: 'Llama-3.2-3B-Instruct',
    displayName: 'Llama 3.2 3B Instruct Q4_K_M',
    supportsVision: false,
    temperature: 0.7,
    maxTokens: 2048,
  },

  // Example: Your custom quantized model
  // 👇 MODIFY THIS FOR YOUR CUSTOM MODEL
  'custom-quantized': {
    name: 'my-custom-model',        // Change to your model identifier
    displayName: 'My Custom Model',  // Change to your display name
    supportsVision: false,           // Set to true if your model supports vision
    temperature: 0.7,                // Adjust as needed
    maxTokens: 2048,                 // Adjust based on your model's capabilities
  },

  // Example: Vision-enabled model
  'llama-vision': {
    name: 'llama-3.2-11b-vision',
    displayName: 'Llama 3.2 11B Vision',
    supportsVision: true,
    temperature: 0.7,
    maxTokens: 2048,
  },

  // Add your own model configurations here
  'my-model': {
    name: 'my-custom-model',
    displayName: 'My Custom Model',
    supportsVision: false,
    temperature: 0.8,
    maxTokens: 4096,
  }
}

// =============================================================================
// Active Model Selection
// =============================================================================
// Change this to switch between model presets defined above.
// Options: 'llama-3.2-3b', 'custom-quantized', 'llama-vision', 'my-model'
// 👇 CHANGE THIS TO SELECT YOUR MODEL
// =============================================================================
export const activeModelPreset = 'llama-3.2-3b'  // Using llama.cpp for testing

// Helper function to get current model configuration
export const getCurrentModel = () => {
  const preset = modelPresets[activeModelPreset]
  if (!preset) {
    console.warn(`Model preset '${activeModelPreset}' not found, using default config`)
    return config.model
  }
  return { ...config.model, ...preset }
}

export default config
