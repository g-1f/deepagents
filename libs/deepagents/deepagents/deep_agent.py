"""
Deep Agent System
=================
A skill-based agent architecture with:
- Progressive skill discovery (single skill activation)
- Dynamic tool binding based on skill prompts
- DAG-based subagent orchestration
- Session management for conversation history
- Comprehensive logging

Author: Alpha Cortex Team
"""

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    """Global configuration for the deep agent system."""
    model_name: str = "claude-sonnet-4-20250514"
    max_iterations: int = 25
    temperature: float = 0.0
    session_dir: str = "./sessions"
    log_dir: str = "./logs"
    log_level: str = "DEBUG"
    parallel_subagents: bool = True
    subagent_timeout: int = 300  # seconds


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class AgentLogger:
    """
    Comprehensive logging system for the deep agent.
    Tracks: skill discovery, tool calls, subagent lifecycle, timing, errors.
    """
    
    def __init__(self, config: AgentConfig, session_id: str):
        self.session_id = session_id
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        log_file = self.log_dir / f"{session_id}.log"
        
        # Configure logger
        self.logger = logging.getLogger(f"deep_agent.{session_id}")
        self.logger.setLevel(getattr(logging, config.log_level))
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)
        
        # Console handler with concise format
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Structured event log for analysis
        self.events: list[dict] = []
    
    def _log_event(self, event_type: str, data: dict):
        """Log a structured event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **data
        }
        self.events.append(event)
        return event
    
    def skill_discovery_start(self, user_query: str):
        """Log start of skill discovery."""
        self.logger.info(f"🔍 SKILL DISCOVERY START | Query: {user_query[:100]}...")
        return self._log_event("skill_discovery_start", {"query": user_query})
    
    def skill_discovered(self, skill_name: str, confidence: float, reasoning: str):
        """Log skill discovery result."""
        self.logger.info(f"✅ SKILL DISCOVERED | {skill_name} (confidence: {confidence:.2f})")
        self.logger.debug(f"   Reasoning: {reasoning}")
        return self._log_event("skill_discovered", {
            "skill_name": skill_name,
            "confidence": confidence,
            "reasoning": reasoning
        })
    
    def skill_activated(self, skill_name: str, bound_tools: list[str]):
        """Log skill activation with bound tools."""
        self.logger.info(f"⚡ SKILL ACTIVATED | {skill_name} | Tools: {bound_tools}")
        return self._log_event("skill_activated", {
            "skill_name": skill_name,
            "bound_tools": bound_tools
        })
    
    def subagent_spawned(self, subagent_id: str, task: str, tools: list[str]):
        """Log subagent creation."""
        self.logger.info(f"🚀 SUBAGENT SPAWNED | {subagent_id} | Task: {task[:80]}...")
        self.logger.debug(f"   Tools: {tools}")
        return self._log_event("subagent_spawned", {
            "subagent_id": subagent_id,
            "task": task,
            "tools": tools
        })
    
    def subagent_completed(self, subagent_id: str, duration_ms: int, output_preview: str):
        """Log subagent completion."""
        self.logger.info(f"✔️  SUBAGENT COMPLETE | {subagent_id} | {duration_ms}ms")
        self.logger.debug(f"   Output: {output_preview[:200]}...")
        return self._log_event("subagent_completed", {
            "subagent_id": subagent_id,
            "duration_ms": duration_ms,
            "output_preview": output_preview[:500]
        })
    
    def subagent_failed(self, subagent_id: str, error: str):
        """Log subagent failure."""
        self.logger.error(f"❌ SUBAGENT FAILED | {subagent_id} | {error}")
        return self._log_event("subagent_failed", {
            "subagent_id": subagent_id,
            "error": error
        })
    
    def tool_call(self, tool_name: str, inputs: dict, subagent_id: Optional[str] = None):
        """Log tool invocation."""
        context = f" (subagent: {subagent_id})" if subagent_id else ""
        self.logger.debug(f"🔧 TOOL CALL{context} | {tool_name} | {json.dumps(inputs)[:200]}")
        return self._log_event("tool_call", {
            "tool_name": tool_name,
            "inputs": inputs,
            "subagent_id": subagent_id
        })
    
    def tool_result(self, tool_name: str, result_preview: str, subagent_id: Optional[str] = None):
        """Log tool result."""
        context = f" (subagent: {subagent_id})" if subagent_id else ""
        self.logger.debug(f"📤 TOOL RESULT{context} | {tool_name} | {result_preview[:200]}")
        return self._log_event("tool_result", {
            "tool_name": tool_name,
            "result_preview": result_preview[:500],
            "subagent_id": subagent_id
        })
    
    def synthesis_start(self, subagent_outputs: list[str]):
        """Log start of result synthesis."""
        self.logger.info(f"🔄 SYNTHESIS START | Combining {len(subagent_outputs)} outputs")
        return self._log_event("synthesis_start", {
            "num_outputs": len(subagent_outputs)
        })
    
    def response_complete(self, total_duration_ms: int):
        """Log complete response."""
        self.logger.info(f"✅ RESPONSE COMPLETE | Total: {total_duration_ms}ms")
        return self._log_event("response_complete", {
            "total_duration_ms": total_duration_ms
        })
    
    def export_events(self, filepath: Optional[str] = None) -> str:
        """Export all events to JSON."""
        filepath = filepath or str(self.log_dir / f"{self.session_id}_events.json")
        with open(filepath, 'w') as f:
            json.dump(self.events, f, indent=2)
        return filepath


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@dataclass
class Message:
    """A single message in conversation history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """Conversation session with history and state."""
    session_id: str
    created_at: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    active_skill: Optional[str] = None
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
    
    def get_langchain_messages(self, max_messages: int = 50) -> list:
        """Convert to LangChain message format."""
        lc_messages = []
        for msg in self.messages[-max_messages:]:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages
    
    def to_dict(self) -> dict:
        """Serialize session to dict."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp, "metadata": m.metadata}
                for m in self.messages
            ],
            "metadata": self.metadata,
            "active_skill": self.active_skill
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Deserialize session from dict."""
        session = cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            active_skill=data.get("active_skill")
        )
        for m in data.get("messages", []):
            session.messages.append(Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", ""),
                metadata=m.get("metadata", {})
            ))
        return session


class SessionManager:
    """Manages conversation sessions with persistence."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session_dir = Path(config.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create a new session."""
        session_id = session_id or str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            created_at=datetime.now().isoformat()
        )
        self._sessions[session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID, loading from disk if needed."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Try loading from disk
        session_file = self.session_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
            session = Session.from_dict(data)
            self._sessions[session_id] = session
            return session
        
        return None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(session_id)
    
    def _save_session(self, session: Session):
        """Persist session to disk."""
        session_file = self.session_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def save_session(self, session: Session):
        """Public method to save session."""
        self._save_session(session)
    
    def list_sessions(self) -> list[str]:
        """List all available session IDs."""
        return [f.stem for f in self.session_dir.glob("*.json")]


# ============================================================================
# SKILL SYSTEM
# ============================================================================

class ExecutionMode(Enum):
    """How to execute tools in a skill step."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class SkillStep:
    """A single step in a skill's execution DAG."""
    step_id: str
    description: str
    tools: list[str]  # Tool names to bind
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    depends_on: list[str] = field(default_factory=list)  # step_ids this depends on
    prompt_template: str = ""  # Additional instructions for subagent


@dataclass
class Skill:
    """
    A skill defines:
    - When it should activate (trigger patterns)
    - What tools it needs
    - How to execute (DAG of steps)
    """
    name: str
    description: str
    trigger_patterns: list[str]  # Keywords/phrases that activate this skill
    steps: list[SkillStep]
    priority: int = 0  # Higher priority wins on conflicts
    system_prompt: str = ""  # Additional context for the skill
    
    def get_all_tools(self) -> set[str]:
        """Get all tools required by this skill."""
        tools = set()
        for step in self.steps:
            tools.update(step.tools)
        return tools


class SkillRegistry:
    """Registry of all available skills."""
    
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    def register(self, skill: Skill):
        """Register a skill."""
        self._skills[skill.name] = skill
    
    def get(self, name: str) -> Optional[Skill]:
        """Get skill by name."""
        return self._skills.get(name)
    
    def all_skills(self) -> list[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())
    
    def get_skill_descriptions(self) -> str:
        """Get formatted descriptions of all skills for discovery prompt."""
        descriptions = []
        for skill in sorted(self._skills.values(), key=lambda s: -s.priority):
            descriptions.append(f"""
<skill name="{skill.name}" priority="{skill.priority}">
  <description>{skill.description}</description>
  <triggers>{', '.join(skill.trigger_patterns)}</triggers>
  <steps>
    {chr(10).join(f'    <step id="{s.step_id}">{s.description}</step>' for s in skill.steps)}
  </steps>
</skill>""")
        return "\n".join(descriptions)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """
    Registry of all available tools.
    Tools are registered globally but bound dynamically per skill.
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def get_tools(self, names: list[str]) -> list[BaseTool]:
        """Get multiple tools by name."""
        tools = []
        for name in names:
            if name in self._tools:
                tools.append(self._tools[name])
        return tools
    
    def all_tool_names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())


# ============================================================================
# SUBAGENT FACTORY
# ============================================================================

class SubagentFactory:
    """Factory for creating skill-specific subagents."""
    
    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry, logger: AgentLogger):
        self.config = config
        self.tool_registry = tool_registry
        self.logger = logger
        self.llm = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=8192
        )
    
    def create_subagent(
        self,
        subagent_id: str,
        task: str,
        tools: list[str],
        system_prompt: str = ""
    ):
        """
        Create a subagent for a specific task.
        
        Returns a callable that executes the subagent and returns results.
        """
        # Get tools for this subagent
        bound_tools = self.tool_registry.get_tools(tools)
        
        self.logger.subagent_spawned(subagent_id, task, tools)
        
        # Create the subagent prompt
        subagent_system = f"""You are a specialized subagent executing a specific task.

<task>
{task}
</task>

<instructions>
{system_prompt if system_prompt else "Execute the task using the available tools. Be thorough and precise."}
</instructions>

<output_format>
Provide a clear, structured output that can be synthesized with other subagent outputs.
Focus on facts and results, not process narration.
</output_format>
"""
        
        # Create react agent with tools
        checkpointer = MemorySaver()
        agent = create_react_agent(
            self.llm,
            bound_tools,
            checkpointer=checkpointer
        )
        
        return agent, subagent_system, subagent_id


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

SKILL_DISCOVERY_PROMPT = """You are a skill discovery system. Your task is to analyze the user's query and determine which single skill should be activated.

<available_skills>
{skill_descriptions}
</available_skills>

<instructions>
1. Analyze the user's query to understand their intent
2. Match the query against skill triggers and descriptions
3. Select EXACTLY ONE skill that best matches the query
4. If no skill clearly matches, respond with skill_name: "none"
</instructions>

<output_format>
Respond in this exact JSON format:
{{
  "skill_name": "<name of the selected skill or 'none'>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<brief explanation of why this skill was selected>"
}}
</output_format>

<user_query>
{user_query}
</user_query>

Analyze and respond with the JSON:"""


SYNTHESIS_PROMPT = """You are synthesizing results from multiple subagent executions to create a coherent response.

<original_query>
{original_query}
</original_query>

<subagent_outputs>
{subagent_outputs}
</subagent_outputs>

<instructions>
1. Combine all subagent outputs into a unified, coherent response
2. Resolve any conflicts or redundancies
3. Present the information in a clear, organized manner
4. Do not mention subagents or internal processes - present as a direct answer
</instructions>

Synthesize the results:"""


class DeepAgent:
    """
    Main orchestrator for the deep agent system.
    
    Handles:
    - Skill discovery via progressive disclosure
    - Dynamic tool binding based on activated skill
    - DAG-based subagent orchestration
    - Result synthesis
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.skill_registry = SkillRegistry()
        self.tool_registry = ToolRegistry()
        self.session_manager = SessionManager(self.config)
        
        # LLM for skill discovery and synthesis
        self.llm = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=8192
        )
    
    def register_skill(self, skill: Skill):
        """Register a skill."""
        self.skill_registry.register(skill)
    
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tool_registry.register(tool)
    
    async def _discover_skill(
        self,
        query: str,
        session: Session,
        logger: AgentLogger
    ) -> Optional[tuple[Skill, float, str]]:
        """
        Progressive skill discovery.
        Returns (skill, confidence, reasoning) or None if no match.
        """
        logger.skill_discovery_start(query)
        
        # Build discovery prompt
        skill_descriptions = self.skill_registry.get_skill_descriptions()
        prompt = SKILL_DISCOVERY_PROMPT.format(
            skill_descriptions=skill_descriptions,
            user_query=query
        )
        
        # Get skill recommendation
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse response
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            skill_name = result.get("skill_name", "none")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            if skill_name == "none" or confidence < 0.3:
                logger.logger.info(f"❌ NO SKILL MATCHED | confidence: {confidence}")
                return None
            
            skill = self.skill_registry.get(skill_name)
            if skill:
                logger.skill_discovered(skill_name, confidence, reasoning)
                return skill, confidence, reasoning
            
            return None
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.logger.error(f"Failed to parse skill discovery response: {e}")
            return None
    
    async def _execute_skill_step(
        self,
        step: SkillStep,
        context: dict,
        session: Session,
        logger: AgentLogger,
        factory: SubagentFactory
    ) -> dict:
        """Execute a single skill step, potentially spawning subagents."""
        
        step_results = {}
        
        if step.mode == ExecutionMode.PARALLEL and len(step.tools) > 1:
            # Spawn parallel subagents for each tool
            tasks = []
            for tool_name in step.tools:
                subagent_id = f"{step.step_id}_{tool_name}_{uuid.uuid4().hex[:4]}"
                task_description = f"{step.description} using {tool_name}"
                
                agent, system_prompt, sid = factory.create_subagent(
                    subagent_id=subagent_id,
                    task=task_description,
                    tools=[tool_name],
                    system_prompt=step.prompt_template
                )
                
                tasks.append(self._run_subagent(
                    agent, system_prompt, sid, context, logger
                ))
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                tool_name = step.tools[i]
                if isinstance(result, Exception):
                    logger.subagent_failed(f"{step.step_id}_{tool_name}", str(result))
                    step_results[tool_name] = f"Error: {result}"
                else:
                    step_results[tool_name] = result
        else:
            # Sequential execution - single subagent with all tools
            subagent_id = f"{step.step_id}_{uuid.uuid4().hex[:4]}"
            
            agent, system_prompt, sid = factory.create_subagent(
                subagent_id=subagent_id,
                task=step.description,
                tools=step.tools,
                system_prompt=step.prompt_template
            )
            
            result = await self._run_subagent(agent, system_prompt, sid, context, logger)
            step_results["result"] = result
        
        return step_results
    
    async def _run_subagent(
        self,
        agent,
        system_prompt: str,
        subagent_id: str,
        context: dict,
        logger: AgentLogger
    ) -> str:
        """Run a subagent and return its output."""
        start_time = datetime.now()
        
        try:
            # Build the input message
            input_message = f"{system_prompt}\n\nContext from previous steps:\n{json.dumps(context, indent=2)}"
            
            # Run the agent
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=input_message)]},
                config=config
            )
            
            # Extract the final response
            messages = result.get("messages", [])
            if messages:
                final_message = messages[-1]
                output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                output = "No output generated"
            
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.subagent_completed(subagent_id, duration_ms, output[:200])
            
            return output
            
        except Exception as e:
            logger.subagent_failed(subagent_id, str(e))
            raise
    
    async def _execute_skill_dag(
        self,
        skill: Skill,
        query: str,
        session: Session,
        logger: AgentLogger
    ) -> list[dict]:
        """Execute the skill's DAG of steps."""
        
        # Bind tools for this skill
        all_tools = list(skill.get_all_tools())
        logger.skill_activated(skill.name, all_tools)
        
        # Create subagent factory
        factory = SubagentFactory(self.config, self.tool_registry, logger)
        
        # Build execution order (topological sort)
        completed_steps: dict[str, dict] = {}
        pending_steps = {step.step_id: step for step in skill.steps}
        step_outputs = []
        
        context = {
            "original_query": query,
            "skill_name": skill.name,
            "conversation_history": [
                {"role": m.role, "content": m.content} 
                for m in session.messages[-10:]  # Last 10 messages for context
            ]
        }
        
        while pending_steps:
            # Find steps with all dependencies satisfied
            ready_steps = [
                step for step_id, step in pending_steps.items()
                if all(dep in completed_steps for dep in step.depends_on)
            ]
            
            if not ready_steps:
                # Circular dependency or error
                logger.logger.error("No ready steps - possible circular dependency")
                break
            
            # Execute ready steps (could be parallel at this level too)
            for step in ready_steps:
                # Add outputs from dependencies to context
                step_context = context.copy()
                for dep_id in step.depends_on:
                    step_context[f"step_{dep_id}_output"] = completed_steps[dep_id]
                
                # Execute the step
                result = await self._execute_skill_step(
                    step, step_context, session, logger, factory
                )
                
                completed_steps[step.step_id] = result
                step_outputs.append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "output": result
                })
                del pending_steps[step.step_id]
        
        return step_outputs
    
    async def _synthesize_results(
        self,
        query: str,
        step_outputs: list[dict],
        logger: AgentLogger
    ) -> str:
        """Synthesize outputs from all steps into final response."""
        
        logger.synthesis_start([str(o) for o in step_outputs])
        
        # Format outputs for synthesis
        formatted_outputs = []
        for output in step_outputs:
            formatted_outputs.append(f"""
<step id="{output['step_id']}">
  <description>{output['description']}</description>
  <output>{json.dumps(output['output'], indent=2)}</output>
</step>
""")
        
        prompt = SYNTHESIS_PROMPT.format(
            original_query=query,
            subagent_outputs="\n".join(formatted_outputs)
        )
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    
    async def _handle_no_skill(
        self,
        query: str,
        session: Session,
        logger: AgentLogger
    ) -> str:
        """Handle queries that don't match any skill (direct LLM response)."""
        
        # Use conversation history for context
        messages = session.get_langchain_messages()
        messages.append(HumanMessage(content=query))
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    async def run(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> tuple[str, Session]:
        """
        Main entry point for processing a user query.
        
        Returns (response, session).
        """
        start_time = datetime.now()
        
        # Get or create session
        session = self.session_manager.get_or_create_session(session_id)
        
        # Initialize logger for this session
        logger = AgentLogger(self.config, session.session_id)
        
        # Add user message to session
        session.add_message("user", query)
        
        try:
            # Step 1: Discover appropriate skill
            discovery_result = await self._discover_skill(query, session, logger)
            
            if discovery_result is None:
                # No skill matched - direct response
                response = await self._handle_no_skill(query, session, logger)
            else:
                skill, confidence, reasoning = discovery_result
                session.active_skill = skill.name
                
                # Step 2: Execute skill DAG
                step_outputs = await self._execute_skill_dag(skill, query, session, logger)
                
                # Step 3: Synthesize results
                response = await self._synthesize_results(query, step_outputs, logger)
            
            # Add assistant response to session
            session.add_message("assistant", response)
            
            # Calculate total duration
            total_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.response_complete(total_ms)
            
            # Save session
            self.session_manager.save_session(session)
            
            # Export event log
            logger.export_events()
            
            return response, session
            
        except Exception as e:
            logger.logger.error(f"Error processing query: {e}")
            error_response = f"I encountered an error processing your request: {str(e)}"
            session.add_message("assistant", error_response, error=str(e))
            self.session_manager.save_session(session)
            return error_response, session
    
    def run_sync(self, query: str, session_id: Optional[str] = None) -> tuple[str, Session]:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(query, session_id))


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Interactive CLI for the deep agent."""
    import sys
    
    print("=" * 60)
    print("  DEEP AGENT SYSTEM")
    print("  Type 'quit' to exit, 'new' for new session")
    print("=" * 60)
    
    agent = DeepAgent()
    
    # Register example skills and tools (see skills.py and tools.py)
    from skills import register_default_skills
    from tools import register_default_tools
    
    register_default_skills(agent)
    register_default_tools(agent)
    
    session_id = None
    
    while True:
        try:
            query = input("\n> ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'new':
                session_id = None
                print("Starting new session...")
                continue
            elif not query:
                continue
            
            response, session = agent.run_sync(query, session_id)
            session_id = session.session_id
            
            print(f"\n[Session: {session_id}]")
            print("-" * 40)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
