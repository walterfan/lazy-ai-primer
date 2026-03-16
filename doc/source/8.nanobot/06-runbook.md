# Operations Runbook

## Installation

### From Source (recommended for development)

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

### From PyPI

```bash
pip install nanobot-ai
```

### With uv

```bash
uv tool install nanobot-ai
```

### With Matrix Support

```bash
pip install nanobot-ai[matrix]
```

## First-Time Setup

```bash
# 1. Initialize config and workspace
nanobot onboard

# 2. Edit config to add API keys
# macOS/Linux:
vim ~/.nanobot/config.json
# Or use any editor

# 3. Verify setup
nanobot status

# 4. Test with a message
nanobot agent -m "Hello!"
```

### Minimum Viable Config

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-your-key-here"
    }
  }
}
```

## Running

### CLI Mode (Interactive)

```bash
nanobot agent              # Interactive REPL
nanobot agent -m "Hello"   # Single message
nanobot agent --no-markdown # Plain text output
nanobot agent --logs       # Show debug logs
```

### Gateway Mode (Daemon)

```bash
nanobot gateway            # Foreground
```

### Docker

```bash
# Build
docker build -t nanobot .

# First-time setup
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard

# Run gateway
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway

# Single command
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot agent -m "Hello!"
```

### Docker Compose

```bash
docker compose run --rm nanobot-cli onboard   # First-time setup
docker compose up -d nanobot-gateway           # Start gateway
docker compose logs -f nanobot-gateway         # View logs
docker compose down                            # Stop
```

### systemd User Service (Linux)

Create `~/.config/systemd/user/nanobot-gateway.service`:

```ini
[Unit]
Description=Nanobot Gateway
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/nanobot gateway
Restart=always
RestartSec=10
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=%h

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now nanobot-gateway
loginctl enable-linger $USER   # Keep running after logout
```

## Common Operations

### Check Status

```bash
nanobot status
```

Shows: configured providers, enabled channels, workspace path, model selection.

### Reset a Session

Delete the session file:

```bash
rm ~/.nanobot/workspace/sessions/telegram_12345.jsonl
```

Or clear all sessions:

```bash
rm ~/.nanobot/workspace/sessions/*.jsonl
```

### View Memory

```bash
cat ~/.nanobot/workspace/memory/MEMORY.md    # Long-term facts
cat ~/.nanobot/workspace/memory/HISTORY.md   # Chronological log
```

### Edit Heartbeat Tasks

```bash
vim ~/.nanobot/workspace/HEARTBEAT.md
```

Format:
```markdown
## Periodic Tasks
- [ ] Check weather forecast and send a summary
- [ ] Scan inbox for urgent emails
```

### Add/Update Provider Credentials

Edit `~/.nanobot/config.json` and restart:

```bash
vim ~/.nanobot/config.json
# If running as systemd service:
systemctl --user restart nanobot-gateway
```

### OAuth Provider Login

```bash
nanobot provider login openai-codex
nanobot provider login github-copilot
```

### WhatsApp QR Code Setup

```bash
# Terminal 1: start WhatsApp bridge
nanobot channels login

# Terminal 2: start gateway
nanobot gateway
```

## Troubleshooting

### Problem: "No provider configured"

**Symptoms**: `nanobot agent` returns an error about no API key.

**Solution**:
1. Run `nanobot status` to check provider configuration
2. Ensure at least one provider has a valid `apiKey` in `~/.nanobot/config.json`
3. Check the model name matches a configured provider

### Problem: "Access denied for sender"

**Symptoms**: Bot ignores messages from a chat platform.

**Solution**:
1. Check the `allowFrom` list in the channel config
2. Add the sender's ID to `allowFrom`
3. Use `["*"]` to allow all senders (for testing only)
4. Check the logs for the actual sender ID being rejected

### Problem: Channel fails to connect

**Symptoms**: "Channel not available" in logs.

**Solution**:
1. Check that the channel SDK is installed (some are optional)
2. Verify credentials (bot token, app secret, etc.)
3. Check network connectivity and proxy settings
4. For WebSocket channels (Feishu, DingTalk, Slack): ensure firewall allows outbound WebSocket

### Problem: Tool execution failures

**Symptoms**: Agent reports tool errors.

**Solution**:
1. If workspace-restricted (`restrictToWorkspace: true`), ensure the file/command is within `~/.nanobot/workspace/`
2. For shell commands: check `tools.exec.timeout` (default: 60s)
3. For MCP tools: check `toolTimeout` on the MCP server config (default: 30s)
4. Check file permissions

### Problem: Memory/session corruption

**Symptoms**: Garbled responses, repeated context, or "tool_call_id not found" errors.

**Solution**:
1. Delete the affected session file: `rm ~/.nanobot/workspace/sessions/{key}.jsonl`
2. Memory files can be manually edited: `~/.nanobot/workspace/memory/MEMORY.md`
3. Restart the gateway

### Problem: High memory usage

**Symptoms**: Python process consuming excessive RAM.

**Solution**:
1. Lower `agents.defaults.memory_window` (default: 100) to consolidate more aggressively
2. Delete old session files
3. Reduce `agents.defaults.max_tool_iterations` (default: 40)

## Configuration Reference

### Environment Variables

The config supports environment variable overrides with prefix `NANOBOT_` and nested delimiter `__`:

```bash
export NANOBOT_AGENTS__DEFAULTS__MODEL="openai/gpt-4o"
export NANOBOT_PROVIDERS__OPENAI__API_KEY="sk-xxx"
```

### Key Configuration Paths

| Setting | Path in JSON | Default | Description |
|---------|-------------|---------|-------------|
| Model | `agents.defaults.model` | `anthropic/claude-opus-4-5` | LLM model identifier |
| Provider | `agents.defaults.provider` | `auto` | Provider name or `auto` |
| Max tokens | `agents.defaults.maxTokens` | `8192` | Max response tokens |
| Temperature | `agents.defaults.temperature` | `0.1` | Sampling temperature |
| Workspace | `agents.defaults.workspace` | `~/.nanobot/workspace` | Working directory |
| Memory window | `agents.defaults.memoryWindow` | `100` | Messages before consolidation |
| Tool iterations | `agents.defaults.maxToolIterations` | `40` | Max tool call rounds |
| Workspace sandbox | `tools.restrictToWorkspace` | `false` | Sandbox all tools |
| Shell timeout | `tools.exec.timeout` | `60` | Shell command timeout (seconds) |
| Heartbeat interval | `gateway.heartbeat.intervalS` | `1800` | Heartbeat wake-up interval (seconds) |
| Progress streaming | `channels.sendProgress` | `true` | Stream partial responses |

## Monitoring

### Log Output

- **CLI mode**: logs to stderr (visible with `--logs` flag)
- **Gateway mode**: logs to stderr (pipe to file or use journalctl for systemd)
- **Docker**: `docker compose logs -f nanobot-gateway`
- **systemd**: `journalctl --user -u nanobot-gateway -f`

### Health Indicators

- Gateway process is running
- Channels show "enabled" in `nanobot status`
- Heartbeat fires every 30 minutes (visible in logs)
- Session files are being updated (check timestamps)

## Backup & Recovery

### What to Back Up

| Path | Priority | Content |
|------|----------|---------|
| `~/.nanobot/config.json` | Critical | All credentials and settings |
| `~/.nanobot/workspace/memory/` | High | Long-term memory and history |
| `~/.nanobot/workspace/sessions/` | Medium | Conversation history |
| `~/.nanobot/workspace/skills/` | Medium | Custom user skills |
| `~/.nanobot/workspace/HEARTBEAT.md` | Low | Periodic task definitions |

### Recovery

1. Reinstall nanobot
2. Restore `~/.nanobot/config.json`
3. Restore `~/.nanobot/workspace/` directory
4. Run `nanobot status` to verify

## Related Documentation

- [Architecture](02-architecture.md) — System design
- [Data Model](04-data-and-api.md) — Storage formats
- [Conventions](05-conventions.md) — Development conventions

---

**Last Updated**: 2026-03-15
**Version**: 1.0
