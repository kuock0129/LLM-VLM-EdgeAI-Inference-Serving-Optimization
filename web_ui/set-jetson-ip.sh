#!/bin/bash

# =============================================================================
# Auto-detect or set Jetson IP Address
# =============================================================================
# This script helps you set the JETSON_IP environment variable.
#
# Usage:
#   source ./set-jetson-ip.sh
#   npm run dev:jetson
#
# Or add to your ~/.zshrc or ~/.bashrc:
#   export JETSON_IP=172.20.10.3
# =============================================================================

# Method 1: Try to find Jetson on network (requires nmap or arp-scan)
find_jetson_ip() {
    echo "🔍 Searching for Jetson on network..."

    # Try using arp to find devices
    if command -v arp &> /dev/null; then
        # This won't work reliably, just showing concept
        echo "   Using arp command..."
    fi

    # Try ping sweep (won't work without knowing subnet)
    # For now, return empty
    return 1
}

# Method 2: Get from last known IP file
get_saved_ip() {
    if [ -f ~/.jetson_ip ]; then
        cat ~/.jetson_ip
        return 0
    fi
    return 1
}

# Method 3: Prompt user
prompt_for_ip() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          Jetson Orin Nano IP Configuration                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "On your Jetson, run this command to get the IP:"
    echo "  hostname -I | awk '{print \$1}'"
    echo ""
    read -p "Enter Jetson IP address (or press Enter for default 172.20.10.3): " ip

    if [ -z "$ip" ]; then
        ip="172.20.10.3"
    fi

    echo "$ip" > ~/.jetson_ip
    echo "$ip"
}

# Main logic
if [ -n "$JETSON_IP" ]; then
    echo "✅ JETSON_IP already set: $JETSON_IP"
elif SAVED_IP=$(get_saved_ip); then
    export JETSON_IP="$SAVED_IP"
    echo "✅ Using saved Jetson IP: $JETSON_IP"
else
    NEW_IP=$(prompt_for_ip)
    export JETSON_IP="$NEW_IP"
    echo "✅ Jetson IP set to: $JETSON_IP"
fi

echo ""
echo "To make this permanent, add to your ~/.zshrc or ~/.bashrc:"
echo "  export JETSON_IP=$JETSON_IP"
echo ""
echo "Now run: npm run dev:jetson"
echo ""
