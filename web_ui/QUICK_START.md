# Quick Start - Dynamic IP Configuration

## 🚀 The IP Address Updates Automatically!

No need to edit files when your Jetson's IP changes. Just set it once and run!

---

## Method 1: One-Time Setup (Recommended)

### Step 1: Add to your shell profile

Add this line to `~/.zshrc` (or `~/.bashrc` if using bash):

```bash
export JETSON_IP=172.20.10.3
```

### Step 2: Reload shell
```bash
source ~/.zshrc
```

### Step 3: Run the web UI
```bash
npm run dev:jetson
```

**When IP changes:** Just update the IP in `~/.zshrc` and restart the server. No code changes needed!

---

## Method 2: Set IP Each Time

Run with your current Jetson IP:

```bash
JETSON_IP=172.20.10.3 npm run dev:jetson
```

Or use BACKEND_HOST directly:

```bash
BACKEND_HOST=172.20.10.3 npm run dev
```

---

## Method 3: Use Helper Script

```bash
# Run the helper script (saves IP for future use)
source ./set-jetson-ip.sh

# Then start the web UI
npm run dev:jetson
```

---

## Method 4: Use Hostname (mDNS)

If your Jetson has mDNS/Avahi enabled:

```bash
BACKEND_HOST=jetson.local npm run dev
```

This way the IP never needs to change!

To enable mDNS on Jetson:
```bash
# On Jetson
sudo apt install avahi-daemon
sudo systemctl enable avahi-daemon
sudo systemctl start avahi-daemon
```

Then access it as `jetson.local` from your Mac!

---

## Quick Commands

| Command | Use Case |
|---------|----------|
| `npm run dev` | Local backend (localhost:8080) |
| `npm run dev:local` | Same as above |
| `npm run dev:jetson` | Jetson backend (uses $JETSON_IP env var) |
| `BACKEND_HOST=<IP> npm run dev` | Custom IP on the fly |

---

## Example Workflow

### First Time Setup:

```bash
# On Jetson - Get IP
hostname -I | awk '{print $1}'
# Output: 172.20.10.3

# On your Mac - Set it permanently
echo 'export JETSON_IP=172.20.10.3' >> ~/.zshrc
source ~/.zshrc

# Start web UI
npm run dev:jetson
```

### Every Time After:

```bash
npm run dev:jetson
```

### When IP Changes:

```bash
# Get new IP from Jetson
hostname -I | awk '{print $1}'
# Output: 192.168.1.100

# Update your shell profile
export JETSON_IP=192.168.1.100
echo 'export JETSON_IP=192.168.1.100' >> ~/.zshrc

# Restart web UI (Ctrl+C then):
npm run dev:jetson
```

---

## How It Works

The `vite.config.js` reads from environment variables:

```javascript
const BACKEND_HOST = process.env.BACKEND_HOST || process.env.JETSON_IP || 'localhost'
const BACKEND_URL = `http://${BACKEND_HOST}:8080`
```

Priority:
1. `BACKEND_HOST` environment variable
2. `JETSON_IP` environment variable
3. `localhost` (default)

---

## Verification

After starting, you'll see:

```
============================================================
🚀 Edge AI Web UI Configuration
============================================================
📡 Backend URL: http://172.20.10.3:8080
🔧 To change: BACKEND_HOST=<IP> npm run dev
============================================================

  VITE v6.4.3  ready in 140 ms

  ➜  Local:   http://localhost:3000/
```

Open http://localhost:3000 and start chatting!

---

## Troubleshooting

**Can't connect to Jetson:**
```bash
# Verify Jetson IP
ping $JETSON_IP

# Test llama-server
curl http://$JETSON_IP:8080/health
```

**Wrong IP being used:**
```bash
# Check what's set
echo $JETSON_IP
echo $BACKEND_HOST

# Override
BACKEND_HOST=<correct-ip> npm run dev
```

---

**No more manual file editing! Just set the environment variable and go!** 🎉
