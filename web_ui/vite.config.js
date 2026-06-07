import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync } from 'child_process'

// =============================================================================
// BACKEND CONFIGURATION (Automatic IP Detection)
// =============================================================================
// The backend IP is automatically detected from your system!
//
// Priority:
//   1. Environment variable: BACKEND_HOST=<IP> npm run dev
//   2. Auto-detect local IP using system commands
//   3. localhost (fallback)
//
// The system will try to find a device on your network at port 8080.
// If you want to override, set BACKEND_HOST environment variable.
//
// Example: BACKEND_HOST=jetson.local npm run dev
// =============================================================================

/**
 * Auto-detect backend IP by scanning network for llama-server
 */
function detectBackendIP() {
  try {
    // Get all network IPs from this machine
    const localIPs = getLocalIPs()

    // Common patterns for Jetson IPs on same network
    // Try to find a device with llama-server running on port 8080
    const subnet = getSubnet(localIPs[0])

    if (subnet) {
      console.log(`🔍 Scanning subnet ${subnet}.0/24 for llama-server...`)

      // Try common Jetson IP patterns first (faster)
      const commonIPs = [
        `${subnet}.3`,   // Common for USB tethering
        `${subnet}.10`,
        `${subnet}.100`,
        `${subnet}.2`,
      ]

      for (const ip of commonIPs) {
        if (checkLlamaServer(ip)) {
          return ip
        }
      }
    }

    // Fallback: try mDNS hostname
    if (checkLlamaServer('jetson.local')) {
      return 'jetson.local'
    }

    return null
  } catch (error) {
    console.warn('⚠️  Could not auto-detect backend IP:', error.message)
    return null
  }
}

/**
 * Get local machine IP addresses
 */
function getLocalIPs() {
  try {
    // macOS/Linux
    const output = execSync("hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo ''",
      { encoding: 'utf8', timeout: 2000 }).trim()

    if (output) {
      return output.split(' ').filter(ip => ip && ip !== '127.0.0.1')
    }

    return []
  } catch (error) {
    return []
  }
}

/**
 * Extract subnet from IP (e.g., 192.168.1.5 -> 192.168.1)
 */
function getSubnet(ip) {
  if (!ip) return null
  const parts = ip.split('.')
  if (parts.length === 4) {
    return `${parts[0]}.${parts[1]}.${parts[2]}`
  }
  return null
}

/**
 * Check if llama-server is running at given IP
 */
function checkLlamaServer(ip) {
  try {
    // Quick check using curl with 1 second timeout
    execSync(`curl -s -m 1 http://${ip}:8080/health > /dev/null 2>&1`,
      { timeout: 1500 })
    console.log(`✅ Found llama-server at ${ip}`)
    return true
  } catch (error) {
    return false
  }
}

export default defineConfig(() => {
  // Determine backend host
  let BACKEND_HOST
  let configSource

  // Priority 1: Environment variable
  if (process.env.BACKEND_HOST) {
    BACKEND_HOST = process.env.BACKEND_HOST
    configSource = 'Environment Variable'
  }
  // Priority 2: Auto-detect
  else {
    console.log('🔍 Auto-detecting backend IP address...')
    const detectedIP = detectBackendIP()

    if (detectedIP) {
      BACKEND_HOST = detectedIP
      configSource = 'Auto-Detected'
    } else {
      BACKEND_HOST = 'localhost'
      configSource = 'Default (localhost)'
      console.log('⚠️  Could not find llama-server. Using localhost.')
      console.log('💡 Start llama-server or set BACKEND_HOST=<IP> npm run dev')
    }
  }

  const BACKEND_PORT = process.env.BACKEND_PORT || '8080'
  const BACKEND_PROTOCOL = process.env.BACKEND_PROTOCOL || 'http'

  const BACKEND_URL = `${BACKEND_PROTOCOL}://${BACKEND_HOST}:${BACKEND_PORT}`

  console.log('\n' + '='.repeat(70))
  console.log('🚀 Edge AI Web UI Configuration')
  console.log('='.repeat(70))
  console.log(`📡 Backend URL: ${BACKEND_URL}`)
  console.log(`📋 Source: ${configSource}`)
  if (configSource !== 'Environment Variable') {
    console.log(`💡 Override: BACKEND_HOST=<IP> npm run dev`)
  }
  console.log('='.repeat(70) + '\n')

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/v1': {
          target: BACKEND_URL,
          changeOrigin: true,
          secure: false,
          ws: true,
        }
      }
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    }
  }
})
