# MLOps Stack

**Authored by Ayush Singh** | Production-grade MLOps skills for Claude Code

A collection of opinionated, human-in-the-loop Claude Code skills that guide you through building production ML systems from scratch. Not templates. Not tutorials. Co-pilots that adapt to YOUR problem, YOUR data, and YOUR constraints.

---

## Skills

### `/mlops-tabular` — Tabular Supervised Learning

The flagship skill. Takes you end-to-end from business problem to deployed, monitored ML pipeline on tabular data.

**What it does:**
- **Phase 1: Problem Framing** — Translates your business problem into an ML formulation. Defines metrics, error costs, success criteria.
- **Phase 2: MLOps Architecture Design** — Designs the complete MLOps stack: data plan, feature engineering plan, training plan, deployment plan, monitoring plan, versioning plan, and selects specific framework components.
- **Phase 3: Implementation** — Builds every pipeline step incrementally. Explains every decision in plain language with deep technical substance. Asks permission before writing every file.
- **Phase 4: Ship** — Final verification, documentation, GitHub push.

**Key properties:**
- Human-in-the-loop at every business decision
- Teaches as it builds — simple words, PhD-level depth
- Fetches current library docs before generating code (no stale API errors)
- Framework-extensible — ZenML by default, contributors can add Prefect, Airflow, etc.
- Covers classification AND regression on tabular data

**Supported frameworks:**
- ZenML (built-in, full support with 15+ component types)
- Others can be added via `references/tooling/` — see [contributor guide](skills/mlops-tabular/references/tooling/README.md)

### `/mlops-code-review` — Code Review (SE + ML)

Full software engineering and ML-specific code review. Three-pass protocol: general code quality, ML-specific issues (leakage, skew, reproducibility), then severity triage.

- Reviews against Google Python Style Guide, SOLID, OWASP, and ML best practices
- Detects data leakage, training-serving skew, and pipeline smells in code
- Produces structured findings (Critical / Major / Minor) with fix proposals
- Cross-cutting: invoke at any phase to audit code quality

### `/mlops-system-design` — System Design (General + ML)

System design covering distributed systems and ML infrastructure. Six-step protocol: requirements, estimation, high-level design, deep-dive, tradeoffs, operational concerns.

- General: API design, databases, distributed systems, scalability, reliability/SRE, microservices, messaging
- ML: model serving architecture, feature store design, ML platform design, ML infrastructure patterns
- Sources: DDIA, Google SRE Book, Chip Huyen, Stanford CS329S
- Produces `system_design.md` with ASCII diagrams, tradeoff tables, and cost estimates

### `/mlops-agent-workflow` — Anti-Slop Agentic Engineering

Sets up disciplined workflows for AI coding agents. Five pillars: RPI workflow, context management, quality gates, agent isolation, anti-slop patterns.

- Research-Plan-Implement (RPI) workflow with context compaction
- Smart zone / dumb zone context management
- Quality gates: strict linting, type checking, 100% test pass rate, pre-commit hooks
- Per-agent isolation with git worktrees
- Framework-agnostic: works for any software project, not just ML

### Specialized Deep-Dive Skills

| Skill | Focus |
|-------|-------|
| `/mlops-problem-framing` | Business problem → ML formulation |
| `/mlops-architecture` | Full MLOps pipeline design (10 stages, 5 pipelines) |
| `/mlops-data-and-features` | Data loading, EDA, preprocessing, feature engineering |
| `/mlops-training-eval` | Experiment tracking, training, evaluation |
| `/mlops-deploy-monitor` | Deployment, monitoring, drift detection, production hardening |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/ayush488-glitch/mlops-stack.git

# Symlink skills into Claude Code
ln -sfn $(pwd)/mlops-stack/skills/mlops-tabular ~/.claude/skills/mlops-tabular
ln -sfn $(pwd)/mlops-stack/skills/mlops-code-review ~/.claude/skills/mlops-code-review
ln -sfn $(pwd)/mlops-stack/skills/mlops-system-design ~/.claude/skills/mlops-system-design
ln -sfn $(pwd)/mlops-stack/skills/mlops-agent-workflow ~/.claude/skills/mlops-agent-workflow
ln -sfn $(pwd)/mlops-stack/skills/mlops-problem-framing ~/.claude/skills/mlops-problem-framing
ln -sfn $(pwd)/mlops-stack/skills/mlops-architecture ~/.claude/skills/mlops-architecture
ln -sfn $(pwd)/mlops-stack/skills/mlops-data-and-features ~/.claude/skills/mlops-data-and-features
ln -sfn $(pwd)/mlops-stack/skills/mlops-training-eval ~/.claude/skills/mlops-training-eval
ln -sfn $(pwd)/mlops-stack/skills/mlops-deploy-monitor ~/.claude/skills/mlops-deploy-monitor
```

Then invoke with `/mlops-tabular` (recommended entry point) or any skill directly.

---

## Context7 MCP Setup (Required for Live Docs)

The skills use [Context7](https://github.com/upstash/context7-mcp) to fetch **live, up-to-date documentation** for libraries like ZenML, scikit-learn, MLflow, pandas, etc. Without it, the agent relies on its training data which may have stale API information.

> **Warning**: If Context7 is not configured, the skills will still work but may generate code against outdated APIs. You'll see a warning at the start of each session.

### Step 1: Get a Context7 API Key

1. Go to [context7.com](https://context7.com) and sign up
2. Copy your API key (starts with `ctx7sk-...`)

### Step 2: Set the API Key as an Environment Variable

**macOS / Linux (zsh):**
```bash
echo 'export CONTEXT7_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

**macOS / Linux (bash):**
```bash
echo 'export CONTEXT7_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell — permanent):**
```powershell
[System.Environment]::SetEnvironmentVariable("CONTEXT7_API_KEY", "your-api-key-here", "User")
```
Then restart your terminal.

**Windows (Command Prompt — permanent):**
```cmd
setx CONTEXT7_API_KEY "your-api-key-here"
```
Then restart your terminal.

### Step 3: Add the MCP Server to Claude Code

Create or edit `~/.claude/.mcp.json` (global — works across all projects):

**macOS / Linux:**
```bash
cat > ~/.claude/.mcp.json << 'EOF'
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "env": {
        "DEFAULT_MINIMUM_TOKENS": "10000",
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
      }
    }
  }
}
EOF
```

**Windows (PowerShell):**
```powershell
$mcpConfig = @'
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "env": {
        "DEFAULT_MINIMUM_TOKENS": "10000",
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
      }
    }
  }
}
'@
$mcpConfig | Out-File -FilePath "$env:USERPROFILE\.claude\.mcp.json" -Encoding utf8
```

### Step 4: Restart Claude Code

Restart Claude Code to load the MCP server. You should see `context7` in the available tools.

### Verifying It Works

In a Claude Code session, you can test it:
```
Use Context7 to fetch the latest ZenML pipeline documentation
```

If Context7 is working, it will fetch live docs. If not, you'll get a tool-not-found error — check your `.mcp.json` path and API key.

---

## Project Structure

```
mlops-stack/
└── skills/
    ├── mlops-tabular/                 # Flagship orchestrator
    │   ├── SKILL.md                   # Skill definition
    │   ├── SKILL.md.tmpl              # Template source (edit this)
    │   ├── gen-skill.sh               # Template generator
    │   └── references/
    │       ├── capabilities/          # 19 MLOps concept guides
    │       ├── tooling/zenml/         # 6 ZenML implementation guides
    │       └── examples/              # Reference timelines & case studies
    ├── mlops-code-review/             # Code review (SE + ML)
    │   ├── SKILL.md
    │   └── references/capabilities/   # 10 review reference guides
    ├── mlops-system-design/           # System design (general + ML)
    │   ├── SKILL.md
    │   └── references/capabilities/   # 10 system design reference guides
    ├── mlops-agent-workflow/          # Anti-slop agentic engineering
    │   ├── SKILL.md
    │   └── references/capabilities/   # 5 workflow reference guides
    ├── mlops-problem-framing/         # Problem framing deep-dive
    ├── mlops-architecture/            # Architecture design deep-dive
    ├── mlops-data-and-features/       # Data & features deep-dive
    ├── mlops-training-eval/           # Training & evaluation deep-dive
    └── mlops-deploy-monitor/          # Deployment & monitoring deep-dive
```

---

## Adding New Skills

This repo is designed for multiple skills. To add a new one:

1. Create `skills/<skill-name>/`
2. Add `SKILL.md` with frontmatter (`name`, `description`, `allowed-tools`)
3. Add `references/` for knowledge base
4. Symlink to `~/.claude/skills/<skill-name>`
5. Update this README

---

## Contributing

Contributions welcome — especially:
- New orchestration framework support (Prefect, Airflow, Kubeflow, etc.) in `references/tooling/`
- New skill types (time series, NLP, computer vision pipelines)
- Bug fixes and capability reference improvements

---

## Author

**Ayush Singh** — MLOps Engineer

Built with the conviction that AI should handle the engineering while humans bring the judgment. Every skill in this stack is designed to produce real, deployable, production-grade ML systems — not AI slop.

---

## License

MIT
