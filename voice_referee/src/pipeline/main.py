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

from src.config.settings import Settings, get_settings, TTSConfig
from src.config.daily_config import DailyConfig
from src.processors.referee_monitor import RefereeMonitorProcessor
from src.services.daily_transport import create_daily_transport
from src.services.deepgram_stt import create_deepgram_stt, DeepgramConfig
from src.services.llm_service import create_llm_service, LLMConfig
from src.services.tts_service import create_tts_service
from src.services.vision_service import create_vision_service
from src.processors.screen_analyzer import create_screen_analyzer
from src.processors.commentary_processor import create_commentary_processor

logger = logging.getLogger(__name__)


async def handle_screen_share_started(transport, participant_id: str):
    """Handle when a participant starts screen sharing.

    This enables screen capture for vision analysis.

    Args:
        transport: VoiceRefereeTransport instance
        participant_id: ID of participant who started sharing
    """
    logger.info(f"📺 Screen share detected from {participant_id}, enabling capture...")
    try:
        await transport.enable_screen_capture(participant_id, framerate=1)
    except Exception as e:
        logger.error(f"Failed to enable screen capture: {e}")


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

    # Set up screen share detection callback if vision is enabled
    if settings.vision_enabled:
        transport._on_screen_share_started = lambda pid: handle_screen_share_started(transport, pid)
        logger.info("Screen share detection callback wired")

    # Create LLM service (Anthropic Claude)
    logger.info("Initializing Anthropic LLM service...")
    llm_service = create_llm_service(llm_config)

    # Create TTS service (ElevenLabs)
    logger.info("Initializing ElevenLabs TTS service...")
    tts_service = create_tts_service(tts_config)

    # Create vision services if enabled
    screen_analyzer = None
    commentary_processor = None

    if settings.vision_enabled:
        logger.info("📺 Vision analysis ENABLED - initializing vision components...")

        # Create vision service
        vision_service = create_vision_service(
            provider=settings.vision_provider,
            model=settings.vision_model,
            api_key=settings.vision_api_key or settings.anthropic_api_key,
        )
        logger.info(f"Vision service created: {settings.vision_provider}/{settings.vision_model}")

        # Create screen analyzer
        screen_analyzer = create_screen_analyzer(
            vision_service=vision_service,
            analysis_interval=settings.vision_analysis_interval,
            cost_limit=settings.vision_max_cost,
        )
        logger.info(f"Screen analyzer created: interval={settings.vision_analysis_interval}s, cost_limit=${settings.vision_max_cost}")

        # Create commentary processor
        commentary_processor = create_commentary_processor(
            commentary_style=settings.vision_commentary_style,
            trigger_threshold=0.6,
            cooldown_seconds=5.0,
        )
        logger.info(f"Commentary processor created: style={settings.vision_commentary_style}")
    else:
        logger.info("📺 Vision analysis DISABLED")

    # Assemble the pipeline
    logger.info("Assembling pipeline stages...")

    # Build processor list dynamically
    processors = [
        transport.input(),  # Audio input from Daily
        stt_service,  # Speech-to-text with diarization
    ]

    # Add screen analyzer if vision is enabled
    if screen_analyzer:
        processors.append(screen_analyzer)
        logger.info("  + Screen analyzer added to pipeline")

    processors.append(referee_monitor)  # Monitor for referee calls

    # Add commentary processor if vision is enabled
    if commentary_processor:
        processors.append(commentary_processor)
        logger.info("  + Commentary processor added to pipeline")

    processors.extend([
        llm_service,  # Generate responses
        tts_service,  # Text-to-speech
        transport.output(),  # Audio output to Daily
    ])

    pipeline = Pipeline(processors)

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
    logger.info(f"Vision: {'ENABLED' if settings.vision_enabled else 'DISABLED'}")
    if settings.vision_enabled:
        logger.info(f"  Provider: {settings.vision_provider}")
        logger.info(f"  Model: {settings.vision_model}")
        logger.info(f"  Analysis interval: {settings.vision_analysis_interval}s")
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

    # Introduction message for the referee (matches the OPENING from the system prompt)
    REFEREE_INTRODUCTION = """Hi, I'm your AI mediator—completely neutral, here to help you work through this.
Founders, are you both here?
Quick ground rules: one at a time, stay respectful, focus on understanding.
Let's start—what's the situation?"""

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
