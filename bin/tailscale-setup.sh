#!/usr/bin/env bash
# Register linter-lm instances with tailscale serve (tailnet-only HTTPS).
# Run once. Idempotent — safe to re-run after changes.
#
# Result:
#   https://comrade.taile6163a.ts.net:8451  →  local  instance (Ollama,   port 8099)
#   https://comrade.taile6163a.ts.net:8452  →  remote instance (DeepSeek, port 8098)
set -e

tailscale serve --https=8451 --bg http://127.0.0.1:8099
tailscale serve --https=8452 --bg http://127.0.0.1:8098

echo ""
echo "Tailscale serve registered:"
tailscale serve status
