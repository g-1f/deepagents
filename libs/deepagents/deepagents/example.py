"""
Deep Agent Usage Examples
=========================
Demonstrates various ways to use the deep agent system.
"""

import asyncio
import json
from deep_agent import DeepAgent, AgentConfig
from skills import register_default_skills, create_simple_skill, create_sequential_skill
from tools import register_default_tools


async def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize agent with default config
    agent = DeepAgent()
    register_default_skills(agent)
    register_default_tools(agent)
    
    # Run a query
    query = "What's the current price and technical outlook for AAPL?"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    
    print(f"\nSession ID: {session.session_id}")
    print(f"Active Skill: {session.active_skill}")
    print(f"\nResponse:\n{response[:500]}...")


async def example_conversation_session():
    """Multi-turn conversation with session persistence."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-turn Conversation")
    print("="*60)
    
    agent = DeepAgent()
    register_default_skills(agent)
    register_default_tools(agent)
    
    # First turn
    query1 = "Analyze the risk profile of my portfolio"
    print(f"\nTurn 1: {query1}")
    response1, session = await agent.run(query1)
    print(f"Response: {response1[:300]}...")
    
    # Second turn (continues session)
    query2 = "Now run a stress test for 2008-style crash"
    print(f"\nTurn 2: {query2}")
    response2, session = await agent.run(query2, session_id=session.session_id)
    print(f"Response: {response2[:300]}...")
    
    # Check session history
    print(f"\nConversation History ({len(session.messages)} messages):")
    for msg in session.messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")


async def example_custom_skill():
    """Creating and using a custom skill."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Skill")
    print("="*60)
    
    agent = DeepAgent()
    register_default_tools(agent)  # Need tools first
    
    # Create a simple single-step skill
    earnings_skill = create_simple_skill(
        name="earnings_analysis",
        description="Analyze earnings reports and estimates",
        triggers=["earnings", "EPS", "revenue", "quarterly results", "beat", "miss"],
        tools=["fetch_fundamentals", "fetch_news", "analyze_sentiment"],
        prompt="Focus on earnings surprises, guidance, and analyst reactions.",
        priority=12
    )
    
    agent.register_skill(earnings_skill)
    
    # Test the custom skill
    query = "Did AAPL beat earnings expectations?"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    print(f"Active Skill: {session.active_skill}")
    print(f"Response: {response[:300]}...")


async def example_sequential_skill():
    """Creating a sequential multi-step skill."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Sequential Multi-Step Skill")
    print("="*60)
    
    agent = DeepAgent()
    register_default_tools(agent)
    
    # Create a sequential skill using the helper
    trade_analysis = create_sequential_skill(
        name="trade_opportunity",
        description="Identify and analyze trading opportunities",
        triggers=["trade idea", "opportunity", "setup", "entry point"],
        steps=[
            ("screen", "Screen for potential opportunities", ["fetch_ohlcv", "compute_indicators"]),
            ("fundamental_check", "Verify fundamentals support the trade", ["fetch_fundamentals"]),
            ("risk_sizing", "Calculate position size and risk", ["kelly_calculator"]),
        ],
        priority=11
    )
    
    agent.register_skill(trade_analysis)
    
    query = "Find me a trade opportunity in tech stocks"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    print(f"Active Skill: {session.active_skill}")
    print(f"Response: {response[:400]}...")


async def example_custom_config():
    """Using custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Configuration")
    print("="*60)
    
    # Custom config
    config = AgentConfig(
        model_name="claude-sonnet-4-20250514",
        max_iterations=30,
        temperature=0.1,
        session_dir="./custom_sessions",
        log_dir="./custom_logs",
        log_level="INFO",  # Less verbose
        parallel_subagents=True,
        subagent_timeout=600
    )
    
    agent = DeepAgent(config)
    register_default_skills(agent)
    register_default_tools(agent)
    
    print(f"Config: {config}")
    
    query = "Quick quote for MSFT"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    print(f"Response: {response}")


async def example_no_skill_fallback():
    """Query that doesn't match any skill."""
    print("\n" + "="*60)
    print("EXAMPLE 6: No Skill Match (Direct Response)")
    print("="*60)
    
    agent = DeepAgent()
    register_default_skills(agent)
    register_default_tools(agent)
    
    # This shouldn't match any trading skills
    query = "What's the weather like today?"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    print(f"Active Skill: {session.active_skill}")  # Should be None
    print(f"Response: {response}")


async def example_inspect_logs():
    """Inspect logs after execution."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Inspecting Logs")
    print("="*60)
    
    from deep_agent import AgentLogger
    from pathlib import Path
    
    config = AgentConfig(log_dir="./demo_logs")
    agent = DeepAgent(config)
    register_default_skills(agent)
    register_default_tools(agent)
    
    query = "Analyze options for SPY with IV analysis"
    print(f"\nQuery: {query}")
    
    response, session = await agent.run(query)
    
    # Load and display events
    events_file = Path(config.log_dir) / f"{session.session_id}_events.json"
    
    if events_file.exists():
        with open(events_file) as f:
            events = json.load(f)
        
        print(f"\nLogged {len(events)} events:")
        for event in events[:10]:  # First 10 events
            print(f"  {event['event_type']}: ", end="")
            if event['event_type'] == 'skill_discovered':
                print(f"{event['skill_name']} (confidence: {event['confidence']})")
            elif event['event_type'] == 'subagent_spawned':
                print(f"{event['subagent_id']}")
            elif event['event_type'] == 'subagent_completed':
                print(f"{event['subagent_id']} ({event['duration_ms']}ms)")
            else:
                print("")


async def example_list_capabilities():
    """List all registered skills and tools."""
    print("\n" + "="*60)
    print("EXAMPLE 8: List Capabilities")
    print("="*60)
    
    agent = DeepAgent()
    register_default_skills(agent)
    register_default_tools(agent)
    
    print("\nRegistered Skills:")
    for skill in agent.skill_registry.all_skills():
        print(f"\n  {skill.name} (priority: {skill.priority})")
        print(f"    Description: {skill.description[:80]}...")
        print(f"    Triggers: {skill.trigger_patterns[:5]}...")
        print(f"    Steps: {[s.step_id for s in skill.steps]}")
        print(f"    Tools needed: {list(skill.get_all_tools())}")
    
    print("\nRegistered Tools:")
    for tool_name in agent.tool_registry.all_tool_names():
        tool = agent.tool_registry.get(tool_name)
        print(f"  - {tool_name}: {tool.description[:60]}...")


async def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_conversation_session,
        example_custom_skill,
        example_sequential_skill,
        example_custom_config,
        example_no_skill_fallback,
        example_inspect_logs,
        example_list_capabilities,
    ]
    
    print("="*60)
    print("  DEEP AGENT EXAMPLES")
    print("="*60)
    
    # Run list capabilities first (doesn't need API)
    await example_list_capabilities()
    
    # Note: Other examples require ANTHROPIC_API_KEY
    print("\n" + "="*60)
    print("Note: Other examples require ANTHROPIC_API_KEY environment variable")
    print("Set it with: export ANTHROPIC_API_KEY=your-key")
    print("="*60)
    
    # Uncomment to run all examples:
    # for example in examples:
    #     try:
    #         await example()
    #     except Exception as e:
    #         print(f"Error in {example.__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
