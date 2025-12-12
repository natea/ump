"""
LLM Service for Voice Referee - Intervention Generation

Uses Anthropic Claude for context-aware mediation interventions.
Returns the actual AnthropicLLMService for pipeline integration.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Optional

from pipecat.services.anthropic import AnthropicLLMService

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 150  # Keep responses concise
    timeout_seconds: float = 5.0
    temperature: float = 0.7


REFEREE_SYSTEM_PROMPT = """You are a neutral AI mediator (ump.ai) helping startup founders resolve conflict.

RULES:
- Keep responses under 25 words
- Never take sides on substance
- Acknowledge emotions, redirect to solutions
- Use calm, measured tone
- Address both founders by name if known

INTERVENTION TYPES:
1. SYSTEM INTERVENTION - General session guidance
2. PROTOCOL VIOLATION (Rule X) - Specific rule broken with corrective guidance
3. PROTOCOL WARNING (Rule X) - Warning with suggested alternative

PROTOCOLS TO ENFORCE:
1. No Interruptions - Allow complete thoughts
2. Data Over Opinion - Cite specific metrics or evidence
3. Future Focused - No dredging up resolved past issues
4. Binary Outcome - Commit to a decision by session end

EXAMPLES:
- "PROTOCOL VIOLATION (Rule 3): Future Focused. Please restate using forward-looking impact."
- "PROTOCOL WARNING (Rule 2): Data Over Opinion. Support that claim with metrics."
- "Let's pause here. What would help you both move forward?"
"""

# Fallback templates when LLM is unavailable
FALLBACK_TEMPLATES = {
    "high_tension": [
        "Let's take a breath. What outcome would satisfy both of you?",
        "I'm sensing strong emotions. Let's focus on solutions.",
        "Pause. What specific decision needs to be made here?",
    ],
    "interruption": [
        "PROTOCOL VIOLATION (Rule 1): No Interruptions. Please let them finish.",
        "Hold on. Let's hear the complete thought first.",
    ],
    "opinion_based": [
        "PROTOCOL WARNING (Rule 2): Data Over Opinion. Can you cite specific metrics?",
        "What data supports that view?",
    ],
    "past_focused": [
        "PROTOCOL VIOLATION (Rule 3): Future Focused. How does this impact going forward?",
        "Let's redirect to future impact, not past grievances.",
    ],
    "default": [
        "Let's stay focused on reaching a decision.",
        "What would help move this forward?",
    ],
}


def create_llm_service(config: LLMConfig) -> AnthropicLLMService:
    """
    Factory function to create Anthropic LLM service for pipeline integration.

    Args:
        config: LLMConfig with API key and model settings

    Returns:
        Configured AnthropicLLMService instance (proper FrameProcessor)
    """
    try:
        service = AnthropicLLMService(
            api_key=config.api_key,
            model=config.model,
        )
        logger.info(f"Anthropic LLM service initialized with model: {config.model}")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic LLM service: {e}")
        raise RuntimeError(f"LLM service initialization failed: {e}") from e


class InterventionGenerator:
    """
    Helper class for generating context-aware interventions.
    Used by RefereeMonitorProcessor to generate interventions.
    NOT a pipeline processor - use create_llm_service() for the pipeline.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate_intervention(self, context: dict) -> str:
        """Generate context-aware intervention using direct API call."""
        import anthropic

        try:
            client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
            prompt = self._build_prompt(context)

            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    system=REFEREE_SYSTEM_PROMPT,
                ),
                timeout=self.config.timeout_seconds
            )

            if response.content:
                intervention = response.content[0].text.strip()
                logger.info(f"Generated intervention: {intervention}")
                return intervention

        except asyncio.TimeoutError:
            logger.warning("LLM generation timed out, using fallback")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

        # Fallback to templates
        return self._generate_fallback(context)

    def _build_prompt(self, context: dict) -> str:
        """Build context-aware prompt for LLM"""
        recent_transcript = context.get("recent_transcript", [])
        tension_score = context.get("tension_score", 0.0)
        reason = context.get("reason", "")
        intervention_type = context.get("intervention_type", "SYSTEM")
        founder_names = context.get("founder_names", [])

        # Build transcript context
        transcript_text = "\n".join([
            f"{utt.get('speaker', 'Unknown')}: {utt.get('text', '')}"
            for utt in recent_transcript[-5:]
        ])

        # Build speaker context
        speaker_context = ""
        if founder_names:
            speaker_context = f"Founders: {', '.join(founder_names)}\n"

        return f"""CONTEXT:
{speaker_context}
Recent conversation:
{transcript_text}

SITUATION:
Tension Level: {tension_score:.2f}/1.0
Trigger: {reason}
Intervention Type: {intervention_type}

Generate a brief intervention (under 25 words) that:
1. Addresses the situation directly
2. Uses the appropriate intervention format (PROTOCOL VIOLATION/WARNING if applicable)
3. Redirects toward constructive resolution
4. Maintains neutral, calm tone

INTERVENTION:"""

    def _generate_fallback(self, context: dict) -> str:
        """Generate fallback intervention from templates"""
        reason = context.get("reason", "").lower()
        tension_score = context.get("tension_score", 0.0)

        if "interruption" in reason:
            templates = FALLBACK_TEMPLATES["interruption"]
        elif "opinion" in reason or "data" in reason:
            templates = FALLBACK_TEMPLATES["opinion_based"]
        elif "past" in reason or "history" in reason:
            templates = FALLBACK_TEMPLATES["past_focused"]
        elif tension_score > 0.7:
            templates = FALLBACK_TEMPLATES["high_tension"]
        else:
            templates = FALLBACK_TEMPLATES["default"]

        intervention = random.choice(templates)
        logger.info(f"Using fallback intervention: {intervention}")
        return intervention
