"""Skill executor for isolated skill execution.

This module provides the SkillExecutor class for running skills in isolated
contexts with support for parallel execution, timeouts, and error recovery.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel

from deepagents.skills.base import Skill

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from a skill execution.

    Attributes:
        skill_name: Name of the skill that was executed.
        success: Whether the execution succeeded.
        result: The result string if successful.
        error: Error message if execution failed.
        execution_time: Time taken to execute in seconds.
    """

    skill_name: str
    success: bool
    result: str | None = None
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class SkillExecutor:
    """Executor for running skills in isolated contexts.

    The executor provides:
    - Isolated execution: Each skill runs with its own message history
    - Parallel execution: Multiple skills can run concurrently
    - Timeout handling: Skills are terminated if they exceed timeout
    - Error recovery: Failures in one skill don't affect others

    Example:
        ```python
        executor = SkillExecutor(model=my_model)

        # Execute a single skill
        result = await executor.execute(market_skill, "Research AAPL")

        # Execute multiple skills in parallel
        results = await executor.execute_parallel([
            (market_skill, "Research AAPL"),
            (quant_skill, "Analyze AAPL technicals"),
        ])
        ```
    """

    model: BaseChatModel
    """Default model for skill execution."""

    default_timeout: float = 300.0
    """Default timeout in seconds for skill execution."""

    max_parallel: int = 5
    """Maximum number of skills to run in parallel."""

    context: dict[str, Any] = field(default_factory=dict)
    """Shared context passed to all skills."""

    def execute(
        self,
        skill: Skill,
        task: str,
        *,
        timeout: float | None = None,
        model: BaseChatModel | None = None,
    ) -> SkillResult:
        """Execute a skill synchronously.

        Args:
            skill: The skill to execute.
            task: Task description for the skill.
            timeout: Optional timeout override.
            model: Optional model override.

        Returns:
            SkillResult with execution outcome.
        """
        import time

        start_time = time.time()
        timeout = timeout or skill.config.timeout or self.default_timeout
        model = model or self.model

        try:
            # Use thread pool for timeout support
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    skill.invoke,
                    task,
                    model,
                    context=self.context,
                )
                result = future.result(timeout=timeout)

            return SkillResult(
                skill_name=skill.name,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
            )

        except FuturesTimeoutError:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"Skill '{skill.name}' timed out after {timeout}s",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.exception("Error executing skill %s", skill.name)
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def aexecute(
        self,
        skill: Skill,
        task: str,
        *,
        timeout: float | None = None,
        model: BaseChatModel | None = None,
    ) -> SkillResult:
        """Execute a skill asynchronously.

        Args:
            skill: The skill to execute.
            task: Task description for the skill.
            timeout: Optional timeout override.
            model: Optional model override.

        Returns:
            SkillResult with execution outcome.
        """
        import time

        start_time = time.time()
        timeout = timeout or skill.config.timeout or self.default_timeout
        model = model or self.model

        try:
            result = await asyncio.wait_for(
                skill.ainvoke(task, model, context=self.context),
                timeout=timeout,
            )

            return SkillResult(
                skill_name=skill.name,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"Skill '{skill.name}' timed out after {timeout}s",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.exception("Error executing skill %s", skill.name)
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def execute_parallel(
        self,
        tasks: list[tuple[Skill, str]],
        *,
        timeout: float | None = None,
        model: BaseChatModel | None = None,
    ) -> list[SkillResult]:
        """Execute multiple skills in parallel (synchronous).

        Args:
            tasks: List of (skill, task_description) tuples.
            timeout: Optional timeout for each skill.
            model: Optional model override.

        Returns:
            List of SkillResult objects in same order as input.
        """
        import time

        start_time = time.time()
        results: list[SkillResult] = []

        # Limit concurrency
        max_workers = min(len(tasks), self.max_parallel)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for skill, task in tasks:
                future = executor.submit(
                    self.execute,
                    skill,
                    task,
                    timeout=timeout,
                    model=model,
                )
                futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # This shouldn't happen since execute() catches exceptions
                    results.append(
                        SkillResult(
                            skill_name="unknown",
                            success=False,
                            error=str(e),
                            execution_time=time.time() - start_time,
                        )
                    )

        return results

    async def aexecute_parallel(
        self,
        tasks: list[tuple[Skill, str]],
        *,
        timeout: float | None = None,
        model: BaseChatModel | None = None,
    ) -> list[SkillResult]:
        """Execute multiple skills in parallel (asynchronous).

        Args:
            tasks: List of (skill, task_description) tuples.
            timeout: Optional timeout for each skill.
            model: Optional model override.

        Returns:
            List of SkillResult objects in same order as input.
        """
        # Create tasks for asyncio
        async_tasks = [
            self.aexecute(skill, task, timeout=timeout, model=model)
            for skill, task in tasks
        ]

        # Limit concurrency with semaphore
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def limited_execute(coro: Any) -> SkillResult:
            async with semaphore:
                return await coro

        limited_tasks = [limited_execute(t) for t in async_tasks]

        # Gather results
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Convert exceptions to SkillResult
        final_results: list[SkillResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                skill_name = tasks[i][0].name if i < len(tasks) else "unknown"
                final_results.append(
                    SkillResult(
                        skill_name=skill_name,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results


def create_executor(
    model: BaseChatModel,
    *,
    timeout: float = 300.0,
    max_parallel: int = 5,
    context: dict[str, Any] | None = None,
) -> SkillExecutor:
    """Create a skill executor.

    Args:
        model: The language model to use.
        timeout: Default timeout for skill execution.
        max_parallel: Maximum parallel skill executions.
        context: Optional shared context for skills.

    Returns:
        Configured SkillExecutor instance.
    """
    return SkillExecutor(
        model=model,
        default_timeout=timeout,
        max_parallel=max_parallel,
        context=context or {},
    )


__all__ = [
    "SkillExecutor",
    "SkillResult",
    "create_executor",
]
