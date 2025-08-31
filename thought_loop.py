from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import time
import threading

from utils.notes_manager import NotesManager
from services.provider_registry import get_llm


@dataclass
class ThoughtStep:
    step: int
    thought: str
    critique: str
    action: str
    outcome: str


class ThoughtLoop:
    """Iterative neural-style thought loop with reflection and working memory.

    Steps per iteration:
      1) Think: propose approach
      2) Critique: evaluate risks/assumptions
      3) Act (mental simulation/tool invocation placeholder)
      4) Outcome: summarize effect and decide to continue/stop

    Persists traces into brain_memory via NotesManager.
    """

    def __init__(self, project_root: str):
        self.notes = NotesManager(project_root)
        try:
            self.llm = get_llm()
        except Exception:
            self.llm = None

    def run(self, prompt: str, max_iters: int = 3) -> Dict[str, Any]:
        start = time.time()
        steps: List[ThoughtStep] = []
        if not self.llm:
            return {"final": prompt, "steps": []}

        working_summary: str = ""
        for i in range(1, max_iters + 1):
            think = self.llm.generate_response(
                f"Context (summary): {working_summary}\n\nTask: {prompt}\n\nThink step {i}: Outline an approach succinctly.")

            # Simulate human parallel thinking: critique and action concurrently
            self.critique_result = None
            critique_thread = threading.Thread(target=self._generate_critique, args=(think, i))
            critique_thread.start()

            action = self.llm.generate_response(
                f"Given the plan and critique, propose a concrete next micro-action (no code execution, just a description).\nPlan: {think}\nCritique: {self.critique_result or 'Processing...'}")

            # Wait for critique to complete
            critique_thread.join()
            critique = self.critique_result

            outcome = self.llm.generate_response(
                f"Simulate the likely outcome of executing: {action}. If adequate to answer the task, provide final answer; else say CONTINUE and what to refine.")

            steps.append(ThoughtStep(i, think, critique, action, outcome))

            # Update working memory
            working_summary = self.llm.generate_response(
                "Summarize the conversation so far in 2-3 bullets, carrying forward only essential details for the next step.\n" +
                "\n".join([s.thought for s in steps[-2:]])[:2000]
            )

            if "FINAL" in outcome.upper() or "STOP" in outcome.upper():
                break

        final_answer = self.llm.generate_response(
            "Provide the final result succinctly for the original task, based on the last outcome above."
        )

        trace = {
            "type": "thought_loop_trace",
            "started": start,
            "ended": time.time(),
            "prompt": prompt,
            "steps": [asdict(s) for s in steps],
            "final": final_answer,
        }
        # Persist trace for learning
        self.notes.save_json('notes', 'thought_loop', trace)
        return trace

    def _generate_critique(self, think: str, i: int):
        """Generate critique in parallel thread to simulate human concurrent thinking"""
        self.critique_result = self.llm.generate_response(
            f"Critique this plan step {i} realistically. Identify risks, missing info, and simpler alternatives.\nPlan: {think}")
