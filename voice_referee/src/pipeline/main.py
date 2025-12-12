"""
Main Pipecat pipeline assembly for Voice Referee system.

Pipeline flow:
transport.input() → stt → referee_monitor → llm → tts → transport.output()

The pipeline processes audio from players/coaches, uses Deepgram for diarization
and transcription, monitors for referee calls via the RefereeMonitorProcessor,
generates responses using Anthropic Claude, and outputs audio via ElevenLabs TTS.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TTSSpeakFrame

from ..config.settings import Settings, get_settings, TTSConfig
from ..config.daily_config import DailyConfig
from ..processors.referee_monitor import RefereeMonitorProcessor
from ..services.daily_transport import create_daily_transport
from ..services.deepgram_stt import create_deepgram_stt, DeepgramConfig
from ..services.llm_service import create_llm_service, LLMConfig
from ..services.tts_service import create_tts_service

logger = logging.getLogger(__name__)


def create_pipeline(settings: Settings) -> tuple[Pipeline, PipelineTask]:
    """
    Create and configure the complete Pipecat pipeline.

    Args:
        settings: Application settings containing API keys and configuration

    Returns:
        Tuple of (Pipeline, PipelineTask) ready to run

    Pipeline stages:
        1. DailyTransport input - receives audio from participants
        2. DeepgramSTTService - transcribes speech with speaker diarization
        3. RefereeMonitorProcessor - monitors for referee calls and rule violations
        4. AnthropicLLMService - generates contextual referee responses
        5. ElevenLabsTTSService - converts responses to speech
        6. DailyTransport output - sends audio back to participants
    """
    logger.info("Creating Voice Referee pipeline...")

    # Build config objects from settings
    daily_config = DailyConfig(
        api_key=settings.daily_token,  # Daily uses token for auth
        room_url=settings.daily_room_url,
        token=settings.daily_token,
        bot_name="Voice Referee",
    )

    deepgram_config = DeepgramConfig(
        api_key=settings.deepgram_api_key,
        model=settings.deepgram_model,
        diarize=settings.deepgram_diarize,
        language=settings.deepgram_language,
    )

    llm_config = LLMConfig(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
    )

    tts_config = TTSConfig(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.tts_voice_id,
        model=settings.tts_model,
    )

    # Create transport (Daily.co WebRTC)
    logger.info("Initializing Daily transport...")
    transport = create_daily_transport(daily_config)

    # Create STT service (Deepgram with diarization)
    logger.info("Initializing Deepgram STT service...")
    stt_service = create_deepgram_stt(deepgram_config)

    # Create referee monitor processor
    logger.info("Initializing Referee Monitor processor...")
    referee_monitor = RefereeMonitorProcessor(settings)

    # Wire up participant events from Daily to the referee monitor
    # This allows the referee to use actual participant names instead of "Founder A"/"Founder B"
    transport.set_participant_callbacks(
        on_joined=referee_monitor.register_participant,
        on_left=referee_monitor.unregister_participant
    )
    logger.info("Participant callbacks wired to referee monitor")

    # Create LLM service (Anthropic Claude)
    logger.info("Initializing Anthropic LLM service...")
    llm_service = create_llm_service(llm_config)

    # Create TTS service (ElevenLabs)
    logger.info("Initializing ElevenLabs TTS service...")
    tts_service = create_tts_service(tts_config)

    # Assemble the pipeline
    logger.info("Assembling pipeline stages...")
    pipeline = Pipeline(
        [
            transport.input(),  # Audio input from Daily
            stt_service,  # Speech-to-text with diarization
            referee_monitor,  # Monitor for referee calls
            llm_service,  # Generate responses
            tts_service,  # Text-to-speech
            transport.output(),  # Audio output to Daily
        ]
    )

    # Configure pipeline parameters
    params = PipelineParams(
        allow_interruptions=True,  # Referee can interrupt ongoing speech
        enable_metrics=True,  # Track performance metrics
        enable_usage_metrics=True,  # Track API usage
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=params,
    )

    logger.info("Pipeline created successfully")
    logger.info(f"Pipeline stages: {len(pipeline._processors)} processors")
    logger.info(f"Interruptions enabled: {params.allow_interruptions}")
    logger.info(f"Metrics enabled: {params.enable_metrics}")

    return pipeline, task


async def run_referee(settings: Optional[Settings] = None) -> None:
    """
    Run the Voice Referee system.

    This is the main entry point that:
    1. Creates the pipeline
    2. Sets up graceful shutdown handlers
    3. Runs the pipeline task
    4. Handles cleanup on exit

    Args:
        settings: Optional settings object. If not provided, will load from environment.
    """
    if settings is None:
        settings = get_settings()

    logger.info("=" * 60)
    logger.info("🏀 Starting Voice Referee System")
    logger.info("=" * 60)
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Daily room: {settings.daily_room_url if settings.daily_room_url else 'Not configured'}")
    logger.info(f"STT: Deepgram {settings.deepgram_model} (diarization: {settings.deepgram_diarize})")
    logger.info(f"LLM: {settings.llm_model}")
    logger.info(f"TTS: ElevenLabs {settings.tts_voice_id}")
    logger.info("=" * 60)

    # Create pipeline and task
    pipeline, task = create_pipeline(settings)

    # Create pipeline runner
    runner = PipelineRunner()

    # Set up graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Introduction message for the referee
    REFEREE_INTRODUCTION = """Hello, I'm the ump.ai referee. Welcome to your mediation session.
I'll be monitoring our conversation to ensure productive dialogue.
Please remember our four protocols: No interruptions. Data over opinion. Stay future focused. And commit to a binary outcome.
When you're ready, please begin your discussion."""

    try:
        # Run the pipeline
        logger.info("🚀 Pipeline starting...")
        logger.info("Waiting for participants to join Daily room...")

        # Run pipeline in background
        pipeline_task = asyncio.create_task(runner.run(task))

        # Wait a moment for pipeline to initialize, then send introduction
        await asyncio.sleep(3)
        logger.info("📢 Sending referee introduction...")
        await task.queue_frame(TTSSpeakFrame(text=REFEREE_INTRODUCTION))

        # Wait for shutdown signal or pipeline completion
        done, pending = await asyncio.wait(
            [pipeline_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If shutdown was triggered, cancel pipeline
        if shutdown_event.is_set():
            logger.info("Shutdown signal received, stopping pipeline...")
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                logger.info("Pipeline task cancelled successfully")

        # Cancel any remaining tasks
        for pending_task in pending:
            pending_task.cancel()
            try:
                await pending_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")

        # Allow time for graceful shutdown
        await asyncio.sleep(1)

        logger.info("=" * 60)
        logger.info("🏀 Voice Referee System stopped")
        logger.info("=" * 60)


async def main():
    """Main entry point for the Voice Referee system."""
    try:
        settings = get_settings()
        await run_referee(settings)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("voice_referee.log"),
        ],
    )

    # Run the async main function
    asyncio.run(main())
