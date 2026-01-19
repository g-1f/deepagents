# DeepAgents Skills Demo

This demo showcases the middleware-free skills architecture for DeepAgents.

## Quick Start

```bash
# Install dependencies
pip install langchain-core langgraph langchain-anthropic pyyaml

# Run with mock model (no API key needed)
python run_skills_demo.py --mock

# Run with real API
export ANTHROPIC_API_KEY=your-key
python run_skills_demo.py
```

## What the Demo Shows

### Demo 1: Parallel Tool Execution
Demonstrates that tools can run in parallel for significant speedup (3x faster for fetching data on 3 symbols).

### Demo 2: Skill Registry
Shows how to register skills and list them.

### Demo 3: Skill Executor
Demonstrates running multiple skills in parallel using `SkillExecutor.aexecute_parallel()`.

### Demo 4: Full Agent with Skills
Creates a complete supervisor agent with multiple skills that can be invoked.

### Demo 5: YAML Configuration
Shows loading skills from the example YAML config file.

## Running Specific Demos

```bash
# Run only demo 1 (parallel tools)
python run_skills_demo.py --demo 1

# Run only demo 4 (full agent)
python run_skills_demo.py --demo 4 --mock
```

## Fake Tools Included

| Tool | Description | Simulated Latency |
|------|-------------|-------------------|
| `fetch_market_data` | Stock price, change, volume, P/E | 0.1s |
| `fetch_news` | News headlines for a query | 0.15s |
| `analyze_sentiment` | BULLISH/BEARISH/NEUTRAL score | 0.05s |
| `technical_analysis` | RSI, MACD, SMA trend | 0.2s |

## Skills Defined

1. **market-research**: Fetches price data and news in parallel, then analyzes sentiment
2. **technical-analysis**: Runs technical indicators (RSI, MACD, SMA)
3. **asset-comparison**: Compares multiple stocks by fetching all data in parallel

## Architecture

```
SUPERVISOR AGENT
├── Planning Tools (write_todos, read_todos)
└── invoke_skill Tool
    ├── market-research (isolated context)
    │   ├── fetch_market_data
    │   ├── fetch_news
    │   └── analyze_sentiment
    ├── technical-analysis (isolated context)
    │   ├── technical_analysis
    │   └── fetch_market_data
    └── asset-comparison (isolated context)
        ├── fetch_market_data
        └── fetch_news
```

Each skill runs in an **isolated context** with its own message history. Results are returned as summaries to the main agent.
