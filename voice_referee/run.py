#!/usr/bin/env python3
"""
Voice Referee System - Entry Point

Simple entry point script for running the Voice Referee system.

Usage:
    python run.py                    # Run with default settings from .env
    python -m voice_referee.run      # Run as module

Requirements:
    - .env file configured with API keys
    - Daily.co room URL configured
    - Python 3.11+
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.pipeline.main import run_referee


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("voice_referee.log"),
        ],
    )


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load settings
        settings = get_settings()

        # Validate required configuration
        if not settings.daily_token:
            logger.error("DAILY_TOKEN not configured in .env file")
            sys.exit(1)

        if not settings.daily_room_url:
            logger.error("DAILY_ROOM_URL not configured in .env file")
            logger.info("Create a room at: https://dashboard.daily.co/")
            sys.exit(1)

        if not settings.deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY not configured in .env file")
            sys.exit(1)

        if not settings.anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not configured in .env file")
            sys.exit(1)

        if not settings.elevenlabs_api_key:
            logger.error("ELEVENLABS_API_KEY not configured in .env file")
            sys.exit(1)

        if not settings.tts_voice_id:
            logger.error("TTS_VOICE_ID not configured in .env file")
            sys.exit(1)

        logger.info("✅ All configuration validated")
        logger.info(f"📍 Room URL: {settings.daily_room_url}")
        logger.info(f"🎤 STT Model: {settings.deepgram_model}")
        logger.info(f"🤖 LLM Model: {settings.llm_model}")
        logger.info(f"🔊 TTS Voice: {settings.tts_voice_id}")

        # Run the voice referee system
        asyncio.run(run_referee(settings))

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
