# Deep Agent System Prompt for Claude Code

## Overview

You are operating a skill-based AI agent infrastructure designed for trading and financial analysis applications. This system implements:

1. **Progressive Skill Discovery** — Single skill activation based on query intent
2. **Dynamic Tool Binding** — Tools injected at runtime based on activated skill
3. **DAG-Based Subagent Orchestration** — Parallel/sequential execution per skill definition
4. **Session Persistence** — Conversation history maintained across interactions
5. **Comprehensive Logging** — Full audit trail of decisions and executions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN AGENT                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Receive user query                                       ││
│  │ 2. Load session (conversation history)                      ││
│  │ 3. Progressive skill discovery → select ONE skill           ││
│  │ 4. Activate skill → bind tools dynamically                  ││
│  │ 5. Execute skill DAG → spawn subagents                      ││
│  │ 6. Synthesize subagent outputs                              ││
│  │ 7. Update session, log events                               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Subagent A  │     │ Subagent B  │     │ Subagent C  │
   │ (Tool A)    │     │ (Tool B)    │     │ (Tool C)    │
   │ ReAct Loop  │     │ ReAct Loop  │     │ ReAct Loop  │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                    ┌─────────────────┐
                    │    SYNTHESIS    │
                    │  Main Agent     │
                    │  combines all   │
                    │  outputs        │
                    └─────────────────┘
```

---

## Core Concepts

### 1. Skills

A **Skill** defines:
- **When** it activates (trigger patterns)
- **What** tools it needs
- **How** it executes (DAG of steps)

```python
@dataclass
class Skill:
    name: str                      # Unique identifier
    description: str               # What this skill does
    trigger_patterns: list[str]    # Keywords/phrases that activate
    steps: list[SkillStep]         # Execution DAG
    priority: int                  # Higher wins conflicts
    system_prompt: str             # Additional context
```

### 2. Skill Steps (DAG Nodes)

Each step defines:
- Which tools to use
- Sequential or parallel execution
- Dependencies on other steps

```python
@dataclass
class SkillStep:
    step_id: str                   # Unique within skill
    description: str               # What this step does
    tools: list[str]               # Tool names to bind
    mode: ExecutionMode            # SEQUENTIAL | PARALLEL
    depends_on: list[str]          # Step IDs this depends on
    prompt_template: str           # Instructions for subagent
```

### 3. Progressive Skill Discovery

The system discovers the appropriate skill through:
1. Analyzing user intent
2. Matching against trigger patterns
3. Considering skill priorities
4. Selecting **exactly one** skill (or none)

```
Query: "What's the IV rank for AAPL options?"
       ↓
Matches: options_strategy (triggers: options, IV, ...)
       ↓
Activated: options_strategy skill
       ↓
Bound tools: [fetch_ohlcv, fetch_options_chain, options_pricer, ...]
```

### 4. Dynamic Tool Binding

Tools are NOT always available. They're injected based on the activated skill:

```
No skill activated → No tools (direct LLM response)
market_analysis   → [fetch_news, analyze_sentiment, fetch_ohlcv, ...]
options_strategy  → [fetch_options_chain, options_pricer, greeks_calculator, ...]
```

### 5. Subagent Orchestration

Each skill step spawns subagent(s) using `create_react_agent`:

```python
# For PARALLEL mode with multiple tools
for tool in step.tools:
    subagent = create_react_agent(llm, [tool], checkpointer)
    tasks.append(run_subagent(subagent, ...))
results = await asyncio.gather(*tasks)

# For SEQUENTIAL mode
subagent = create_react_agent(llm, all_step_tools, checkpointer)
result = await run_subagent(subagent, ...)
```

---

## Execution Flow

### Phase 1: Skill Discovery

```python
async def _discover_skill(query, session, logger):
    # Build prompt with all skill descriptions
    prompt = SKILL_DISCOVERY_PROMPT.format(
        skill_descriptions=registry.get_skill_descriptions(),
        user_query=query
    )
    
    # LLM returns: {skill_name, confidence, reasoning}
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    # Parse and validate
    if skill_name == "none" or confidence < 0.3:
        return None  # No skill matches
    
    return skill, confidence, reasoning
```

### Phase 2: DAG Execution

```python
async def _execute_skill_dag(skill, query, session, logger):
    # Topological sort of steps
    pending = {step.step_id: step for step in skill.steps}
    completed = {}
    
    while pending:
        # Find steps with satisfied dependencies
        ready = [s for s in pending.values() 
                 if all(d in completed for d in s.depends_on)]
        
        for step in ready:
            # Build context from completed dependencies
            context = {f"step_{d}_output": completed[d] for d in step.depends_on}
            
            # Execute step (spawns subagents)
            result = await _execute_skill_step(step, context, ...)
            
            completed[step.step_id] = result
            del pending[step.step_id]
    
    return completed
```

### Phase 3: Synthesis

```python
async def _synthesize_results(query, step_outputs, logger):
    prompt = SYNTHESIS_PROMPT.format(
        original_query=query,
        subagent_outputs=format_outputs(step_outputs)
    )
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content
```

---

## Session Management

### Persistence Structure

```
./sessions/
├── abc12345.json      # Session file
├── def67890.json
└── ...

# Session JSON
{
  "session_id": "abc12345",
  "created_at": "2025-01-19T10:30:00",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "metadata": {...},
  "active_skill": "market_analysis"
}
```

### Usage

```python
# Continue existing session
response, session = await agent.run(query, session_id="abc12345")

# New session (auto-generated ID)
response, session = await agent.run(query)

# Access history
messages = session.get_langchain_messages(max_messages=50)
```

---

## Logging System

### Log Levels

| Level | Purpose |
|-------|---------|
| DEBUG | Tool calls, detailed outputs |
| INFO  | Skill discovery, subagent lifecycle |
| ERROR | Failures, exceptions |

### Event Types

```python
skill_discovery_start   # Query received
skill_discovered        # Skill selected with confidence
skill_activated         # Tools bound
subagent_spawned        # Subagent created
subagent_completed      # Subagent finished
subagent_failed         # Subagent error
tool_call               # Tool invoked
tool_result             # Tool returned
synthesis_start         # Combining outputs
response_complete       # Final response ready
```

### Log Files

```
./logs/
├── abc12345.log           # Human-readable log
├── abc12345_events.json   # Structured event log
└── ...
```

---

## Available Skills

### market_analysis (Priority: 10)
Comprehensive market analysis combining news, technicals, and fundamentals.
- **Triggers**: analyze, market, stock, sentiment, technical analysis, research, outlook
- **Steps**: news_sentiment → technical_analysis → fundamentals → synthesis (parallel first 3)

### options_strategy (Priority: 15)
Design and analyze options strategies with Greeks and position sizing.
- **Triggers**: options, calls, puts, spread, greeks, delta, IV, hedge
- **Steps**: market_context → strategy_design → position_sizing

### portfolio_risk (Priority: 12)
Portfolio risk metrics including VaR, correlations, and stress testing.
- **Triggers**: portfolio, risk, VaR, correlation, exposure, stress test
- **Steps**: data_collection → risk_metrics, correlation_analysis (parallel) → stress_testing

### data_retrieval (Priority: 5)
Simple data fetching for quotes and prices.
- **Triggers**: price of, quote, what is, show me, current, latest
- **Steps**: fetch_data

### news_digest (Priority: 8)
Aggregate and summarize news from multiple sources.
- **Triggers**: news, headlines, what's happening, recent events
- **Steps**: aggregate_news → analyze_and_summarize

### code_generation (Priority: 7)
Generate Python code for quantitative finance tasks.
- **Triggers**: code, python, script, backtest, implement, plot
- **Steps**: understand_requirements → generate_code → validate_code

---

## Available Tools

### Market Data
| Tool | Description |
|------|-------------|
| `fetch_ohlcv` | Historical OHLCV price data |
| `fetch_quote` | Current quote with bid/ask |
| `fetch_options_chain` | Options chain with Greeks |
| `fetch_fundamentals` | Financial ratios and estimates |

### News & Sentiment
| Tool | Description |
|------|-------------|
| `fetch_news` | Recent news headlines |
| `fetch_sec_filings` | SEC filings (10-K, 8-K, etc.) |
| `analyze_sentiment` | Text sentiment analysis |

### Analysis
| Tool | Description |
|------|-------------|
| `compute_indicators` | Technical indicators (MA, RSI, MACD) |
| `options_pricer` | Black-Scholes pricing |
| `greeks_calculator` | Portfolio Greeks |
| `kelly_calculator` | Kelly criterion sizing |

### Risk
| Tool | Description |
|------|-------------|
| `risk_calculator` | VaR, CVaR, Sharpe, Sortino |
| `correlation_analyzer` | Correlation matrix |
| `factor_model` | Factor exposures |
| `stress_tester` | Historical/hypothetical scenarios |

### Portfolio
| Tool | Description |
|------|-------------|
| `fetch_portfolio` | Current holdings |

### Code
| Tool | Description |
|------|-------------|
| `code_executor` | Execute Python code |

---

## Adding New Skills

```python
from deep_agent import Skill, SkillStep, ExecutionMode

my_skill = Skill(
    name="my_custom_skill",
    description="What this skill does",
    trigger_patterns=["keyword1", "keyword2", "phrase"],
    priority=10,
    system_prompt="Additional context for all steps",
    steps=[
        SkillStep(
            step_id="step_1",
            description="First step description",
            tools=["tool_a", "tool_b"],
            mode=ExecutionMode.PARALLEL,
            depends_on=[],
            prompt_template="Instructions for this step"
        ),
        SkillStep(
            step_id="step_2",
            description="Second step description",
            tools=["tool_c"],
            mode=ExecutionMode.SEQUENTIAL,
            depends_on=["step_1"],
            prompt_template="Uses output from step_1"
        )
    ]
)

agent.register_skill(my_skill)
```

---

## Adding New Tools

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param1: str = Field(description="Description of param1")
    param2: int = Field(default=10, description="Description of param2")

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "What this tool does"
    args_schema: type[BaseModel] = MyToolInput
    
    def _run(self, param1: str, param2: int = 10) -> str:
        # Implementation
        result = {"output": f"Processed {param1} with {param2}"}
        return json.dumps(result, indent=2)

agent.register_tool(MyCustomTool())
```

---

## Usage Examples

### Basic Usage

```python
from deep_agent import DeepAgent
from skills import register_default_skills
from tools import register_default_tools

# Initialize
agent = DeepAgent()
register_default_skills(agent)
register_default_tools(agent)

# Run query
response, session = await agent.run("Analyze AAPL stock")
print(response)

# Continue conversation
response, session = await agent.run(
    "What about the options market?",
    session_id=session.session_id
)
```

### Custom Configuration

```python
from deep_agent import DeepAgent, AgentConfig

config = AgentConfig(
    model_name="claude-sonnet-4-20250514",
    max_iterations=30,
    temperature=0.1,
    session_dir="./my_sessions",
    log_dir="./my_logs",
    log_level="INFO",
    parallel_subagents=True,
    subagent_timeout=600
)

agent = DeepAgent(config)
```

### Accessing Logs

```python
# After running a query
logger = AgentLogger(config, session.session_id)

# Export structured events
events_file = logger.export_events()

# Read events
with open(events_file) as f:
    events = json.load(f)
    
for event in events:
    if event["event_type"] == "skill_discovered":
        print(f"Skill: {event['skill_name']}, Confidence: {event['confidence']}")
```

---

## Best Practices

### Skill Design
1. **Specific triggers** — Avoid generic words that match everything
2. **Clear step boundaries** — Each step should have a distinct purpose
3. **Minimize dependencies** — Enable parallelism where possible
4. **Appropriate priority** — Higher for more specific skills

### Tool Design
1. **Single responsibility** — One tool, one job
2. **Structured output** — Always return JSON for easy parsing
3. **Clear descriptions** — LLM uses these to decide when to call
4. **Robust error handling** — Return error info in the response

### Session Management
1. **Persist after each turn** — Don't lose conversation state
2. **Limit history length** — Don't overflow context
3. **Include metadata** — Track active skill, errors, etc.

### Logging
1. **Log at appropriate levels** — DEBUG for details, INFO for events
2. **Include timing** — Track performance bottlenecks
3. **Structured events** — Enable automated analysis

---

## Error Handling

```python
try:
    response, session = await agent.run(query, session_id)
except Exception as e:
    # Error is logged automatically
    # Session is saved with error metadata
    # Graceful error message returned
    pass
```

The system handles:
- Skill discovery failures → Falls back to direct LLM response
- Subagent timeouts → Logs failure, continues with available outputs
- Tool errors → Captured in subagent output
- Circular dependencies → Detected and logged

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `claude-sonnet-4-20250514` | Anthropic model to use |
| `max_iterations` | 25 | Max ReAct loop iterations |
| `temperature` | 0.0 | LLM temperature |
| `session_dir` | `./sessions` | Session persistence path |
| `log_dir` | `./logs` | Log file path |
| `log_level` | `DEBUG` | Logging verbosity |
| `parallel_subagents` | `True` | Enable parallel execution |
| `subagent_timeout` | 300 | Subagent timeout (seconds) |

---

## Dependencies

```
langchain>=0.3.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
pydantic>=2.0.0
```

Install: `pip install langchain langchain-anthropic langgraph`
