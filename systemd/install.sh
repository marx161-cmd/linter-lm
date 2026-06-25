#!/usr/bin/env bash
# Install both linter-lm systemd user services.
# Run once after cloning / updating; safe to re-run.
set -e
UNIT_DIR="${HOME}/.config/systemd/user"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/linter-lm-local.service"    "$UNIT_DIR/"
cp "$SCRIPT_DIR/linter-lm-deepseek.service" "$UNIT_DIR/"

systemctl --user daemon-reload
echo "Services installed. Next steps:"
echo "  cp ${SCRIPT_DIR}/../.env.local.example    ${SCRIPT_DIR}/../.env.local"
echo "  cp ${SCRIPT_DIR}/../.env.deepseek.example ${SCRIPT_DIR}/../.env.deepseek"
echo "  # edit .env.deepseek and set LINTR_BACKEND_API_KEY"
echo "  systemctl --user enable --now linter-lm-local"
echo "  systemctl --user enable --now linter-lm-deepseek"
