# GOAP Action Templates for Voice Referee Implementation

This document provides code templates and implementation patterns for each action in the GOAP plan.

---

## Phase 1: Foundation Setup

### ACTION 1.1: setup_environment

**Bash Script Template:**
```bash
#!/bin/bash
# setup_environment.sh

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required, found $python_version"
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Create .env file from template
cat > .env << EOF
# Daily.co Configuration
DAILY_ROOM_URL=https://example.daily.co/referee-test
DAILY_TOKEN=

# Deepgram Configuration
DEEPGRAM_API_KEY=

# LLM Configuration (Choose one)
ANTHROPIC_API_KEY=
# OPENAI_API_KEY=

# ElevenLabs TTS Configuration
ELEVENLABS_API_KEY=
ELEVENLABS_REFEREE_VOICE_ID=

# Processing Configuration
INTERVENTION_TENSION_THRESHOLD=0.7
INTERVENTION_COOLDOWN_SECONDS=30
TRANSCRIPT_BUFFER_SIZE=50
LOG_LEVEL=INFO
EOF

echo "Environment setup complete!"
echo "Please edit .env file with your API keys"
```

**Validation Script:**
```python
# scripts/validate_foundation.py
import sys
import os
from pathlib import Path

def validate_environment():
    """Validate Phase 1 setup"""
    checks = []

    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        checks.append(("Python version", True, f"{version.major}.{version.minor}"))
    else:
        checks.append(("Python version", False, f"{version.major}.{version.minor} < 3.10"))

    # Check venv
    venv_exists = Path("venv").exists()
    checks.append(("Virtual environment", venv_exists, "venv/ directory"))

    # Check .env
    env_exists = Path(".env").exists()
    checks.append((".env file", env_exists, ".env"))

    # Print results
    print("\n=== Foundation Validation ===\n")
    all_pass = True
    for name, passed, detail in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}: {detail}")
        all_pass = all_pass and passed

    print("\n" + ("="*30))
    if all_pass:
        print("✅ All foundation checks passed!")
        return 0
    else:
        print("❌ Some checks failed. Please fix and retry.")
        return 1

if __name__ == "__main__":
    sys.exit(validate_environment())
```

---

### ACTION 1.4: create_config_module

**Configuration Module Template:**
```python
# src/config/settings.py
from pydantic import BaseModel, Field, validator
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class DailyConfig(BaseModel):
    """Daily.co transport configuration"""
    room_url: str = Field(..., env="DAILY_ROOM_URL")
    token: str = Field(..., env="DAILY_TOKEN")

    @validator("room_url")
    def validate_room_url(cls, v):
        if not v.startswith("https://"):
            raise ValueError("DAILY_ROOM_URL must start with https://")
        return v

class DeepgramConfig(BaseModel):
    """Deepgram STT configuration"""
    api_key: str = Field(..., env="DEEPGRAM_API_KEY")
    model: str = "nova-2"
    language: str = "en"
    diarize: bool = True
    punctuate: bool = True
    interim_results: bool = True
    smart_format: bool = True
    utterance_end_ms: int = 1000

class LLMConfig(BaseModel):
    """LLM configuration for intervention generation"""
    provider: str = Field(default="anthropic", env="LLM_PROVIDER")
    api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-3-5-sonnet-20241022")
    max_tokens: int = 150
    temperature: float = 0.7

    @validator("provider")
    def validate_provider(cls, v):
        if v not in ["anthropic", "openai"]:
            raise ValueError("LLM provider must be 'anthropic' or 'openai'")
        return v

class TTSConfig(BaseModel):
    """ElevenLabs TTS configuration"""
    api_key: str = Field(..., env="ELEVENLABS_API_KEY")
    voice_id: str = Field(..., env="ELEVENLABS_REFEREE_VOICE_ID")
    model: str = "eleven_flash_v2_5"
    stability: float = 0.5
    similarity_boost: float = 0.75
    output_format: str = "pcm_16000"

class ProcessorConfig(BaseModel):
    """Referee processor configuration"""
    tension_threshold: float = Field(default=0.7, env="INTERVENTION_TENSION_THRESHOLD")
    cooldown_seconds: int = Field(default=30, env="INTERVENTION_COOLDOWN_SECONDS")
    transcript_buffer_size: int = Field(default=50, env="TRANSCRIPT_BUFFER_SIZE")
    speaker_imbalance_threshold: float = 0.8
    argument_repetition_threshold: int = 3

    @validator("tension_threshold")
    def validate_tension_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Tension threshold must be between 0.0 and 1.0")
        return v

class Settings(BaseModel):
    """Main settings aggregator"""
    daily: DailyConfig
    deepgram: DeepgramConfig
    llm: LLMConfig
    tts: TTSConfig
    processor: ProcessorConfig
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment"""
        return cls(
            daily=DailyConfig(
                room_url=os.getenv("DAILY_ROOM_URL"),
                token=os.getenv("DAILY_TOKEN")
            ),
            deepgram=DeepgramConfig(
                api_key=os.getenv("DEEPGRAM_API_KEY")
            ),
            llm=LLMConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            ),
            tts=TTSConfig(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=os.getenv("ELEVENLABS_REFEREE_VOICE_ID")
            ),
            processor=ProcessorConfig()
        )

# Global settings instance
settings = Settings.load()
```

---

## Phase 2: Core Services Configuration

### ACTION 2.1: configure_daily_transport

**Daily Transport Setup:**
```python
# src/services/daily_transport.py
from pipecat.transports.daily import DailyTransport, DailyParams, DailyDialinSettings
from pipecat.vad.silero import SileroVADAnalyzer
from src.config.settings import settings
import asyncio
import logging

logger = logging.getLogger(__name__)

class DailyTransportService:
    """Manages Daily.co WebRTC connection"""

    def __init__(self):
        self.transport: Optional[DailyTransport] = None
        self.vad: Optional[SileroVADAnalyzer] = None

    async def initialize(self) -> DailyTransport:
        """Initialize Daily transport with VAD"""

        # Configure VAD
        self.vad = SileroVADAnalyzer(
            min_volume=0.6,
            sample_rate=16000
        )

        # Configure Daily params
        params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=self.vad,
            transcription_enabled=False  # Using Deepgram instead
        )

        # Create transport
        self.transport = DailyTransport(
            settings.daily.room_url,
            settings.daily.token,
            "VoiceReferee",  # Bot name
            params
        )

        logger.info(f"Daily transport initialized for room: {settings.daily.room_url}")
        return self.transport

    async def test_connection(self) -> bool:
        """Test Daily.co connection"""
        try:
            await self.transport.start()
            logger.info("✅ Daily transport connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Daily transport connection failed: {e}")
            return False
```

---

### ACTION 2.3: configure_deepgram_stt

**Deepgram STT Service:**
```python
# src/services/deepgram_stt.py
from pipecat.services.deepgram import DeepgramSTTService
from deepgram import DeepgramClient, LiveOptions
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DiarizedDeepgramSTT:
    """Deepgram STT with diarization enabled"""

    def __init__(self):
        self.client = DeepgramClient(settings.deepgram.api_key)
        self.service: Optional[DeepgramSTTService] = None

    def create_service(self) -> DeepgramSTTService:
        """Create Deepgram STT service with diarization"""

        options = LiveOptions(
            model=settings.deepgram.model,
            language=settings.deepgram.language,
            diarize=settings.deepgram.diarize,
            punctuate=settings.deepgram.punctuate,
            interim_results=settings.deepgram.interim_results,
            smart_format=settings.deepgram.smart_format,
            utterance_end_ms=settings.deepgram.utterance_end_ms
        )

        self.service = DeepgramSTTService(
            api_key=settings.deepgram.api_key,
            live_options=options
        )

        logger.info("Deepgram STT service created with diarization enabled")
        return self.service

    async def test_diarization(self, audio_file: str) -> dict:
        """Test diarization with sample audio"""
        # Implementation for testing
        pass
```

---

## Phase 3: Processing Logic Implementation

### ACTION 3.1a: implement_speaker_mapper

**Speaker Mapper:**
```python
# src/processors/speaker_mapper.py
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SpeakerMapper:
    """Maps Deepgram speaker IDs to founder identities"""

    def __init__(self):
        self.speaker_map: Dict[int, str] = {}
        self.first_speaker_id: Optional[int] = None

    def assign_identity(self, speaker_id: int) -> str:
        """Assign identity to speaker ID (Founder A or B)"""

        # First speaker becomes Founder A
        if self.first_speaker_id is None:
            self.first_speaker_id = speaker_id
            self.speaker_map[speaker_id] = "Founder A"
            logger.info(f"Speaker {speaker_id} assigned as Founder A")
            return "Founder A"

        # Check if already mapped
        if speaker_id in self.speaker_map:
            return self.speaker_map[speaker_id]

        # Second speaker becomes Founder B
        if speaker_id != self.first_speaker_id:
            self.speaker_map[speaker_id] = "Founder B"
            logger.info(f"Speaker {speaker_id} assigned as Founder B")
            return "Founder B"

        return self.speaker_map[speaker_id]

    def get_identity(self, speaker_id: int) -> str:
        """Get identity for speaker ID"""
        if speaker_id not in self.speaker_map:
            return self.assign_identity(speaker_id)
        return self.speaker_map[speaker_id]

    def reset(self):
        """Reset speaker mapping (for new session)"""
        self.speaker_map.clear()
        self.first_speaker_id = None
        logger.info("Speaker mapping reset")
```

---

### ACTION 3.1b: implement_conversation_state_tracker

**Conversation State Tracker:**
```python
# src/processors/conversation_state.py
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class Utterance:
    """Single utterance in conversation"""
    text: str
    speaker: str
    timestamp: float
    duration: float
    speaker_id: int

@dataclass
class SpeakerStats:
    """Statistics for individual speaker"""
    total_time: float = 0.0
    utterance_count: int = 0
    avg_sentiment: float = 0.0
    last_utterance_time: float = 0.0

class ConversationState:
    """Tracks conversation state and statistics"""

    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.transcript_buffer: deque = deque(maxlen=buffer_size)
        self.per_speaker_stats: Dict[str, SpeakerStats] = {}
        self.interruption_count: int = 0
        self.last_intervention_time: float = 0.0
        self.session_start_time: float = time.time()

    def add_utterance(self, text: str, speaker: str, speaker_id: int,
                     timestamp: float, duration: float):
        """Add new utterance to conversation"""

        utterance = Utterance(
            text=text,
            speaker=speaker,
            timestamp=timestamp,
            duration=duration,
            speaker_id=speaker_id
        )

        self.transcript_buffer.append(utterance)

        # Initialize speaker stats if needed
        if speaker not in self.per_speaker_stats:
            self.per_speaker_stats[speaker] = SpeakerStats()

        # Update stats
        stats = self.per_speaker_stats[speaker]
        stats.total_time += duration
        stats.utterance_count += 1
        stats.last_utterance_time = timestamp

        logger.debug(f"Added utterance: {speaker} - {text[:50]}...")

    def get_recent_transcript(self, n: int = 10) -> List[Utterance]:
        """Get last N utterances"""
        return list(self.transcript_buffer)[-n:]

    def calculate_speaker_balance(self) -> float:
        """Calculate speaker time balance (0.0 = balanced, 1.0 = one dominates)"""
        if len(self.per_speaker_stats) < 2:
            return 0.0

        times = [stats.total_time for stats in self.per_speaker_stats.values()]
        total_time = sum(times)

        if total_time == 0:
            return 0.0

        # Calculate max speaker percentage
        max_percentage = max(times) / total_time

        # Return imbalance (0.5 = balanced, 1.0 = one speaker only)
        return max_percentage

    def track_interruption(self, speaker: str):
        """Track speaker interruption"""
        self.interruption_count += 1
        logger.debug(f"Interruption detected: {speaker}")

    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.session_start_time

    def record_intervention(self):
        """Record that intervention occurred"""
        self.last_intervention_time = time.time()

    def time_since_last_intervention(self) -> float:
        """Get time since last intervention"""
        if self.last_intervention_time == 0:
            return float('inf')
        return time.time() - self.last_intervention_time
```

---

### ACTION 3.1c: implement_analysis_engine

**Conversation Analyzer:**
```python
# src/processors/analyzer.py
from typing import List
from src.processors.conversation_state import Utterance, ConversationState
import re
import logging

logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """Analyzes conversation for tension and patterns"""

    def __init__(self):
        # Simple keyword-based sentiment (can be replaced with ML model)
        self.negative_keywords = {
            'no', 'never', 'wrong', 'disagree', 'impossible', 'terrible',
            'awful', 'hate', 'stupid', 'ridiculous', 'absurd'
        }
        self.positive_keywords = {
            'yes', 'agree', 'good', 'great', 'excellent', 'perfect',
            'right', 'love', 'brilliant', 'amazing'
        }

    def detect_sentiment(self, text: str) -> float:
        """
        Detect sentiment of text
        Returns: -1.0 (very negative) to 1.0 (very positive)
        """
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)

        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total
        return sentiment

    def calculate_tension_score(self, state: ConversationState) -> float:
        """
        Calculate overall tension score (0.0 - 1.0)

        Components:
        - Sentiment negativity (30%)
        - Interruption rate (30%)
        - Speaker imbalance (20%)
        - Argument repetition (20%)
        """

        recent_utterances = state.get_recent_transcript(10)

        # 1. Sentiment negativity
        sentiments = [self.detect_sentiment(u.text) for u in recent_utterances]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        sentiment_negativity = max(0, -avg_sentiment)  # 0 to 1

        # 2. Interruption rate
        session_duration = state.get_session_duration()
        interruption_rate = min(1.0, state.interruption_count / max(1, session_duration / 60))

        # 3. Speaker imbalance
        speaker_imbalance = state.calculate_speaker_balance()

        # 4. Argument repetition
        repetition_score = self.detect_argument_repetition(recent_utterances)

        # Weighted combination
        tension = (
            sentiment_negativity * 0.3 +
            interruption_rate * 0.3 +
            speaker_imbalance * 0.2 +
            repetition_score * 0.2
        )

        logger.debug(f"Tension score: {tension:.2f} (sentiment={sentiment_negativity:.2f}, "
                    f"interruptions={interruption_rate:.2f}, imbalance={speaker_imbalance:.2f}, "
                    f"repetition={repetition_score:.2f})")

        return min(1.0, tension)

    def detect_argument_repetition(self, utterances: List[Utterance]) -> float:
        """
        Detect if same arguments are being repeated
        Returns score 0.0-1.0 (higher = more repetition)
        """
        if len(utterances) < 4:
            return 0.0

        # Simple approach: check for keyword overlap between utterances
        keyword_sets = []
        for u in utterances:
            words = set(re.findall(r'\w+', u.text.lower()))
            # Filter common words
            meaningful_words = {w for w in words if len(w) > 4}
            keyword_sets.append(meaningful_words)

        # Calculate overlap between consecutive pairs
        overlaps = []
        for i in range(len(keyword_sets) - 1):
            if not keyword_sets[i] or not keyword_sets[i+1]:
                continue
            overlap = len(keyword_sets[i] & keyword_sets[i+1]) / len(keyword_sets[i] | keyword_sets[i+1])
            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        return avg_overlap

    def calculate_interruption_rate(self, state: ConversationState) -> float:
        """Calculate interruptions per minute"""
        duration_minutes = state.get_session_duration() / 60
        if duration_minutes == 0:
            return 0.0
        return state.interruption_count / duration_minutes
```

---

### ACTION 3.1d: implement_intervention_decision_logic

**Intervention Decider:**
```python
# src/processors/decider.py
from typing import Tuple, Dict
from src.processors.conversation_state import ConversationState
from src.processors.analyzer import ConversationAnalyzer
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class InterventionDecider:
    """Decides when referee should intervene"""

    def __init__(self, analyzer: ConversationAnalyzer, state: ConversationState):
        self.analyzer = analyzer
        self.state = state
        self.config = settings.processor

    def should_intervene(self) -> Tuple[bool, str]:
        """
        Determine if intervention is needed
        Returns: (should_intervene, reason)
        """

        # Check cooldown first
        if not self.check_cooldown():
            return False, "Cooldown active"

        # Calculate metrics
        tension_score = self.analyzer.calculate_tension_score(self.state)
        speaker_balance = self.state.calculate_speaker_balance()
        session_duration = self.state.get_session_duration()

        # Rule 1: High tension
        if tension_score > self.config.tension_threshold:
            logger.info(f"🚨 Intervention triggered: High tension ({tension_score:.2f})")
            return True, f"High tension detected (score: {tension_score:.2f})"

        # Rule 2: Speaker imbalance (after 5 minutes)
        if (speaker_balance > self.config.speaker_imbalance_threshold and
            session_duration > 300):
            logger.info(f"🚨 Intervention triggered: Speaker imbalance ({speaker_balance:.2f})")
            return True, f"One speaker dominating ({speaker_balance:.2f})"

        # Rule 3: Argument repetition
        recent = self.state.get_recent_transcript(10)
        repetition = self.analyzer.detect_argument_repetition(recent)
        if repetition > 0.7:  # High overlap suggests circular argument
            logger.info(f"🚨 Intervention triggered: Circular argument ({repetition:.2f})")
            return True, "Circular argument detected"

        return False, "No intervention needed"

    def check_cooldown(self) -> bool:
        """Check if cooldown period has elapsed"""
        time_since_last = self.state.time_since_last_intervention()
        cooldown_elapsed = time_since_last >= self.config.cooldown_seconds

        if not cooldown_elapsed:
            logger.debug(f"Cooldown active: {self.config.cooldown_seconds - time_since_last:.0f}s remaining")

        return cooldown_elapsed

    def get_intervention_context(self) -> Dict:
        """Get context for LLM intervention generation"""
        recent = self.state.get_recent_transcript(10)
        tension = self.analyzer.calculate_tension_score(self.state)

        context = {
            "tension_score": tension,
            "recent_transcript": [
                {
                    "speaker": u.speaker,
                    "text": u.text,
                    "timestamp": u.timestamp
                }
                for u in recent
            ],
            "speaker_stats": {
                name: {
                    "total_time": stats.total_time,
                    "utterance_count": stats.utterance_count
                }
                for name, stats in self.state.per_speaker_stats.items()
            },
            "session_duration": self.state.get_session_duration(),
            "interruption_count": self.state.interruption_count
        }

        return context
```

---

## Phase 4: Pipeline Integration

### ACTION 4.1: assemble_pipeline

**Main Pipeline:**
```python
# src/pipeline/main.py
from pipecat.pipeline import Pipeline
from pipecat.processors.frameworks.asyncio import PipelineTask
from src.services.daily_transport import DailyTransportService
from src.services.deepgram_stt import DiarizedDeepgramSTT
from src.processors.referee_monitor import RefereeMonitorProcessor
from src.services.llm_service import LLMInterventionService
from src.services.tts_service import TTSService
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceRefereePipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.daily_service = DailyTransportService()
        self.stt_service = DiarizedDeepgramSTT()
        self.processor = RefereeMonitorProcessor()
        self.llm_service = LLMInterventionService()
        self.tts_service = TTSService()

        self.transport = None
        self.pipeline = None
        self.task = None

    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Voice Referee Pipeline...")

        # Initialize transport
        self.transport = await self.daily_service.initialize()

        # Create STT service
        stt = self.stt_service.create_service()

        # Create LLM service
        llm = await self.llm_service.create_service()

        # Create TTS service
        tts = await self.tts_service.create_service()

        # Assemble pipeline
        self.pipeline = Pipeline([
            self.transport.input(),    # Audio input from Daily
            stt,                        # Speech-to-text with diarization
            self.processor,             # Referee monitoring & decision
            llm,                        # Intervention text generation
            tts,                        # Text-to-speech
            self.transport.output()     # Audio output to Daily
        ])

        logger.info("✅ Pipeline initialized successfully")

    async def run(self):
        """Run the pipeline"""
        logger.info("Starting Voice Referee...")

        # Create pipeline task
        self.task = PipelineTask(self.pipeline)

        try:
            # Start pipeline
            await self.task.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        if self.task:
            await self.task.stop()
        if self.transport:
            await self.transport.stop()
        logger.info("✅ Cleanup complete")

async def main():
    """Main entry point"""
    pipeline = VoiceRefereePipeline()
    await pipeline.initialize()
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing Templates

### Unit Test Example

```python
# tests/unit/test_speaker_mapper.py
import pytest
from src.processors.speaker_mapper import SpeakerMapper

class TestSpeakerMapper:

    def test_first_speaker_assigned_founder_a(self):
        """First speaker should be assigned Founder A"""
        mapper = SpeakerMapper()

        identity = mapper.assign_identity(speaker_id=0)

        assert identity == "Founder A"
        assert mapper.first_speaker_id == 0
        assert mapper.speaker_map[0] == "Founder A"

    def test_second_speaker_assigned_founder_b(self):
        """Second speaker should be assigned Founder B"""
        mapper = SpeakerMapper()

        mapper.assign_identity(speaker_id=0)
        identity = mapper.assign_identity(speaker_id=1)

        assert identity == "Founder B"
        assert mapper.speaker_map[1] == "Founder B"

    def test_speaker_identity_persistence(self):
        """Speaker identity should persist across multiple calls"""
        mapper = SpeakerMapper()

        mapper.assign_identity(speaker_id=0)
        mapper.assign_identity(speaker_id=1)

        # Call multiple times
        assert mapper.get_identity(0) == "Founder A"
        assert mapper.get_identity(1) == "Founder B"
        assert mapper.get_identity(0) == "Founder A"

    def test_reset_clears_mapping(self):
        """Reset should clear all mappings"""
        mapper = SpeakerMapper()

        mapper.assign_identity(speaker_id=0)
        mapper.assign_identity(speaker_id=1)
        mapper.reset()

        assert len(mapper.speaker_map) == 0
        assert mapper.first_speaker_id is None
```

---

This template document provides concrete code implementations for key actions in the GOAP plan. Each template includes:
- Working Python code
- Proper error handling
- Logging statements
- Type hints
- Docstrings
- Validation logic

Teams can copy these templates directly into their project structure and customize as needed.
