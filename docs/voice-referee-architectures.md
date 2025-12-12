# Voice AI Referee: Two Architecture Approaches

## Overview

Both architectures use **Daily.co** for WebRTC transport and **ElevenLabs TTS** for the referee's voice output. The key difference is how we handle speech-to-text and speaker identification.

---

## Architecture A: Mixed Audio + Deepgram Diarization

Uses a single mixed audio stream with Deepgram's real-time diarization to identify speakers.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DAILY.CO WEBRTC ROOM                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  Founder A   │    │  Founder B   │    │  AI Referee  │                 │
│  │  (browser)   │    │  (browser)   │    │  (Pipecat)   │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────▲───────┘                 │
│         │                   │                   │                          │
│         └─────────┬─────────┘                   │                          │
│                   │ mixed audio                 │ TTS audio                │
│                   ▼                             │                          │
└───────────────────┼─────────────────────────────┼──────────────────────────┘
                    │                             │
┌───────────────────┼─────────────────────────────┼──────────────────────────┐
│                   │      PIPECAT CLOUD          │                          │
│                   ▼                             │                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      DailyTransport                                  │  │
│  │  • Receives mixed audio from room                                   │  │
│  │  • Sends TTS audio back to room                                     │  │
│  │  • Silero VAD for voice activity detection                          │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              Deepgram STT (with diarization)                        │  │
│  │  • Model: nova-2                                                    │  │
│  │  • diarize=true                                                     │  │
│  │  • Outputs: TranscriptionFrame with speaker labels                  │  │
│  │  • Example: "[Speaker 0]: I think we should pivot..."              │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   RefereeMonitorProcessor                           │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ Speaker Mapping                                              │   │  │
│  │  │ • Map Speaker 0/1 → "Founder A" / "Founder B"               │   │  │
│  │  │ • Track based on join order or voice enrollment              │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ Conversation State                                           │   │  │
│  │  │ • Rolling transcript buffer (last 5 min)                    │   │  │
│  │  │ • Per-speaker word count / speaking time                    │   │  │
│  │  │ • Interruption tracking                                      │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ Analysis Engine                                              │   │  │
│  │  │ • Sentiment analysis per utterance                          │   │  │
│  │  │ • Tension score calculation                                  │   │  │
│  │  │ • Pattern detection (circular arguments, dominance)         │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │ Intervention Decision                                        │   │  │
│  │  │ • IF tension > 0.7 → trigger interjection                   │   │  │
│  │  │ • IF same_argument_count > 3 → trigger interjection         │   │  │
│  │  │ • IF speaker_imbalance > 0.8 → trigger interjection         │   │  │
│  │  │ • ELSE → continue listening (no output)                     │   │  │
│  │  └──────────────────────────────┬──────────────────────────────┘   │  │
│  └─────────────────────────────────┼───────────────────────────────────┘  │
│                                    │                                       │
│                    [Only when intervention triggered]                      │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM (Claude / GPT-4o)                       │  │
│  │  • System prompt: Referee persona & intervention rules             │  │
│  │  • Context: Recent transcript + analysis scores                    │  │
│  │  • Output: Brief mediation statement (< 20 words)                  │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    ElevenLabs TTS (Flash v2.5)                      │  │
│  │  • WebSocket streaming for lowest latency                          │  │
│  │  • Custom referee voice (calm, authoritative)                      │  │
│  │  • optimize_streaming_latency=4                                    │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│                    [Audio sent back via DailyTransport]                    │
└────────────────────────────────────────────────────────────────────────────┘
```

### Code Implementation

```python
import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame, TranscriptionFrame, TextFrame, LLMMessagesFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

class RefereeMonitorProcessor(FrameProcessor):
    """Monitors conversation and triggers interventions when needed."""
    
    def __init__(self):
        super().__init__()
        self.transcript_buffer = []
        self.speaker_stats = {"speaker_0": {"words": 0, "turns": 0}, 
                              "speaker_1": {"words": 0, "turns": 0}}
        self.tension_threshold = 0.7
        self.last_intervention_time = 0
        self.intervention_cooldown = 30  # seconds
        
        # Map Deepgram speaker IDs to founder names (set during session init)
        self.speaker_map = {"speaker_0": "Founder A", "speaker_1": "Founder B"}
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame):
            # Extract speaker info from Deepgram diarization
            speaker_id = getattr(frame, 'speaker', 'unknown')
            text = frame.text
            
            # Update conversation state
            self.transcript_buffer.append({
                "speaker": self.speaker_map.get(f"speaker_{speaker_id}", f"Speaker {speaker_id}"),
                "text": text,
                "timestamp": frame.timestamp
            })
            
            # Keep buffer manageable (last 50 utterances)
            if len(self.transcript_buffer) > 50:
                self.transcript_buffer.pop(0)
            
            # Analyze conversation
            should_intervene, reason = await self._analyze_conversation()
            
            if should_intervene and self._cooldown_expired():
                # Generate intervention
                intervention_prompt = self._build_intervention_prompt(reason)
                await self.push_frame(LLMMessagesFrame(messages=[
                    {"role": "system", "content": self._referee_system_prompt()},
                    {"role": "user", "content": intervention_prompt}
                ]), direction)
                self.last_intervention_time = frame.timestamp
            
        # Always pass frames through
        await self.push_frame(frame, direction)
    
    async def _analyze_conversation(self) -> tuple[bool, str]:
        """Analyze conversation for intervention triggers."""
        
        if len(self.transcript_buffer) < 3:
            return False, ""
        
        recent = self.transcript_buffer[-10:]
        
        # Check 1: Speaker imbalance
        speaker_counts = {}
        for entry in recent:
            speaker_counts[entry["speaker"]] = speaker_counts.get(entry["speaker"], 0) + 1
        
        if speaker_counts:
            max_count = max(speaker_counts.values())
            total = sum(speaker_counts.values())
            if max_count / total > 0.8 and total >= 5:
                dominant = max(speaker_counts, key=speaker_counts.get)
                return True, f"speaker_imbalance:{dominant}"
        
        # Check 2: Repeated arguments (simple keyword overlap)
        recent_texts = [e["text"].lower() for e in recent[-6:]]
        if len(recent_texts) >= 4:
            # Naive circular argument detection
            keywords_per_turn = [set(t.split()) for t in recent_texts]
            overlap_count = 0
            for i in range(len(keywords_per_turn) - 1):
                overlap = len(keywords_per_turn[i] & keywords_per_turn[i+1])
                if overlap > 3:
                    overlap_count += 1
            if overlap_count >= 3:
                return True, "circular_argument"
        
        # Check 3: Tension keywords (simplified - use ML model in production)
        tension_words = ["never", "always", "wrong", "fault", "stupid", "ridiculous"]
        recent_text = " ".join(recent_texts[-3:])
        tension_count = sum(1 for word in tension_words if word in recent_text)
        if tension_count >= 2:
            return True, "high_tension"
        
        return False, ""
    
    def _cooldown_expired(self) -> bool:
        import time
        return (time.time() - self.last_intervention_time) > self.intervention_cooldown
    
    def _build_intervention_prompt(self, reason: str) -> str:
        recent_transcript = "\n".join([
            f"{e['speaker']}: {e['text']}" 
            for e in self.transcript_buffer[-8:]
        ])
        
        return f"""Recent conversation:
{recent_transcript}

Intervention trigger: {reason}

Generate a brief, calming intervention (under 20 words) to redirect this conversation productively."""
    
    def _referee_system_prompt(self) -> str:
        return """You are a neutral AI mediator helping startup founders resolve conflict.

RULES:
- Keep responses under 20 words
- Never take sides on substance
- Acknowledge emotions, redirect to solutions
- Use calm, measured tone
- Address both founders

EXAMPLES:
- "I'm hearing frustration. Let's pause and each share one concrete need."
- "We've circled this point. What would help you both move forward?"
- "Alex, you've made that point. Sam, what's your response?"""


async def main():
    # Transport: Daily.co WebRTC
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token=os.getenv("DAILY_TOKEN"),
        bot_name="AI Referee",
        params=DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=SileroVADAnalyzer.InputParams(
                min_volume=0.5,
            )),
        ),
    )
    
    # STT: Deepgram with diarization
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        params=DeepgramSTTService.InputParams(
            model="nova-2",
            language="en",
            diarize=True,           # Key: Enable speaker diarization
            smart_format=True,
            interim_results=True,
        ),
    )
    
    # Custom referee logic
    referee_monitor = RefereeMonitorProcessor()
    
    # LLM: Claude for intervention generation
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-20250514",
    )
    
    # TTS: ElevenLabs for referee voice
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_REFEREE_VOICE_ID"),
        model="eleven_flash_v2_5",
        params=ElevenLabsTTSService.InputParams(
            stability=0.7,
            similarity_boost=0.75,
            optimize_streaming_latency=4,
        ),
    )
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),      # Audio from Daily room
        stt,                    # Deepgram transcription with diarization
        referee_monitor,        # Custom analysis & intervention logic
        llm,                    # Generate intervention text
        tts,                    # Convert to speech
        transport.output(),     # Send back to Daily room
    ])
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
    
    # Run
    await task.run()
```

### Latency Breakdown

| Component | Latency |
|-----------|---------|
| Daily WebRTC transport | ~10-15ms |
| Deepgram Nova-2 STT | ~100-200ms |
| Silero VAD (local) | <5ms |
| Analysis processing | <10ms |
| Claude Sonnet TTFT | ~300-400ms |
| ElevenLabs Flash v2.5 TTFB | ~75-100ms |
| **Total (when intervention triggers)** | **~500-730ms** |

### Cost Estimate (100 x 1-hour sessions/month)

| Service | Rate | Monthly Cost |
|---------|------|--------------|
| Daily.co (3 participants) | $0.00099/participant-min | ~$18 |
| Deepgram Nova-2 | $0.0043/min | ~$26 |
| ElevenLabs TTS (~5% speak time) | $0.10/min spoken | ~$30 |
| Claude API (~100 interventions/session) | ~$0.003/intervention | ~$30 |
| Pipecat Cloud | $0.04/active-min | ~$240 |
| **Total** | | **~$344/month** |

---

## Architecture B: Separate Streams + ElevenLabs Scribe Per Participant

Uses Daily's per-participant audio tracks, running separate ElevenLabs Scribe instances for each founder.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DAILY.CO WEBRTC ROOM                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  Founder A   │    │  Founder B   │    │  AI Referee  │                 │
│  │  (browser)   │    │  (browser)   │    │  (Pipecat)   │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────▲───────┘                 │
│         │                   │                   │                          │
│         │ track A           │ track B           │ TTS audio                │
│         ▼                   ▼                   │                          │
└─────────┼───────────────────┼───────────────────┼──────────────────────────┘
          │                   │                   │
┌─────────┼───────────────────┼───────────────────┼──────────────────────────┐
│         │   PIPECAT CLOUD   │                   │                          │
│         │                   │                   │                          │
│         ▼                   ▼                   │                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              Custom MultiTrackDailyTransport                        │  │
│  │  • Subscribes to individual participant audio tracks                │  │
│  │  • Routes Track A → STT Instance A                                  │  │
│  │  • Routes Track B → STT Instance B                                  │  │
│  │  • Sends TTS output to room                                         │  │
│  └──────────┬───────────────────────┬──────────────────────────────────┘  │
│             │                       │                                      │
│             ▼                       ▼                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐                         │
│  │ ElevenLabs Scribe   │  │ ElevenLabs Scribe   │                         │
│  │ Instance A          │  │ Instance B          │                         │
│  │ (scribe_v2_realtime)│  │ (scribe_v2_realtime)│                         │
│  │                     │  │                     │                         │
│  │ Output: Transcript  │  │ Output: Transcript  │                         │
│  │ labeled "Founder A" │  │ labeled "Founder B" │                         │
│  └──────────┬──────────┘  └──────────┬──────────┘                         │
│             │                        │                                     │
│             └───────────┬────────────┘                                     │
│                         │ merged transcripts                               │
│                         ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                TranscriptMergerProcessor                            │  │
│  │  • Receives transcripts from both Scribe instances                  │  │
│  │  • Orders by timestamp                                              │  │
│  │  • Creates unified conversation timeline                            │  │
│  │  • Handles overlapping speech                                       │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   RefereeMonitorProcessor                           │  │
│  │  (Same logic as Architecture A)                                     │  │
│  │  • Speaker already labeled - no diarization needed                  │  │
│  │  • Analyze for tension, imbalance, circular arguments               │  │
│  │  • Trigger intervention when thresholds crossed                     │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                    [Only when intervention triggered]                      │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM (Claude / GPT-4o)                       │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    ElevenLabs TTS (Flash v2.5)                      │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│                    [Audio sent back via DailyTransport]                    │
└────────────────────────────────────────────────────────────────────────────┘
```

### Code Implementation

```python
import os
import asyncio
from typing import Dict
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame, AudioRawFrame, TranscriptionFrame, TextFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class LabeledTranscriptionFrame(TranscriptionFrame):
    """TranscriptionFrame with explicit speaker label."""
    def __init__(self, speaker_label: str, **kwargs):
        super().__init__(**kwargs)
        self.speaker_label = speaker_label


class ParticipantSTTProcessor(FrameProcessor):
    """Runs ElevenLabs Scribe for a specific participant and labels output."""
    
    def __init__(self, participant_id: str, speaker_label: str, api_key: str):
        super().__init__()
        self.participant_id = participant_id
        self.speaker_label = speaker_label
        
        # Create dedicated Scribe instance for this participant
        self.stt = ElevenLabsRealtimeSTTService(
            api_key=api_key,
            model="scribe_v2_realtime",
            params=ElevenLabsRealtimeSTTService.InputParams(
                language="en",
                tag_audio_events=True,
            ),
        )
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Only process audio from our assigned participant
        if isinstance(frame, AudioRawFrame):
            if getattr(frame, 'participant_id', None) == self.participant_id:
                # Run through our Scribe instance
                async for result_frame in self.stt.run_stt(frame.audio):
                    if isinstance(result_frame, TranscriptionFrame):
                        # Re-emit with speaker label
                        labeled = LabeledTranscriptionFrame(
                            speaker_label=self.speaker_label,
                            text=result_frame.text,
                            timestamp=result_frame.timestamp,
                        )
                        await self.push_frame(labeled, direction)
        else:
            await self.push_frame(frame, direction)


class TranscriptMergerProcessor(FrameProcessor):
    """Merges transcripts from multiple speakers into unified timeline."""
    
    def __init__(self):
        super().__init__()
        self.pending_transcripts: Dict[str, list] = {}
        self.merge_window_ms = 100  # Merge transcripts within 100ms
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LabeledTranscriptionFrame):
            # Buffer and emit in chronological order
            await self._buffer_and_emit(frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    async def _buffer_and_emit(self, frame: LabeledTranscriptionFrame, direction: FrameDirection):
        """Simple passthrough for now - can add sophisticated merging later."""
        # In production: buffer briefly to handle simultaneous speech,
        # then emit in timestamp order
        await self.push_frame(frame, direction)


class MultiParticipantRefereeMonitor(FrameProcessor):
    """Monitors labeled transcripts and triggers interventions."""
    
    def __init__(self):
        super().__init__()
        self.transcript_buffer = []
        self.tension_threshold = 0.7
        self.last_intervention_time = 0
        self.intervention_cooldown = 30
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, LabeledTranscriptionFrame):
            # Speaker is already labeled - no diarization needed!
            self.transcript_buffer.append({
                "speaker": frame.speaker_label,
                "text": frame.text,
                "timestamp": frame.timestamp,
            })
            
            # Keep buffer manageable
            if len(self.transcript_buffer) > 50:
                self.transcript_buffer.pop(0)
            
            # Analyze and potentially intervene
            should_intervene, reason = await self._analyze_conversation()
            
            if should_intervene and self._cooldown_expired():
                intervention_prompt = self._build_intervention_prompt(reason)
                from pipecat.frames.frames import LLMMessagesFrame
                await self.push_frame(LLMMessagesFrame(messages=[
                    {"role": "system", "content": self._referee_system_prompt()},
                    {"role": "user", "content": intervention_prompt}
                ]), direction)
                import time
                self.last_intervention_time = time.time()
        
        await self.push_frame(frame, direction)
    
    async def _analyze_conversation(self) -> tuple[bool, str]:
        """Same analysis logic as Architecture A."""
        if len(self.transcript_buffer) < 3:
            return False, ""
        
        recent = self.transcript_buffer[-10:]
        
        # Speaker imbalance check
        speaker_counts = {}
        for entry in recent:
            speaker_counts[entry["speaker"]] = speaker_counts.get(entry["speaker"], 0) + 1
        
        if speaker_counts:
            max_count = max(speaker_counts.values())
            total = sum(speaker_counts.values())
            if max_count / total > 0.8 and total >= 5:
                dominant = max(speaker_counts, key=speaker_counts.get)
                return True, f"speaker_imbalance:{dominant}"
        
        # Tension keyword check
        tension_words = ["never", "always", "wrong", "fault", "stupid", "ridiculous"]
        recent_text = " ".join([e["text"].lower() for e in recent[-3:]])
        tension_count = sum(1 for word in tension_words if word in recent_text)
        if tension_count >= 2:
            return True, "high_tension"
        
        return False, ""
    
    def _cooldown_expired(self) -> bool:
        import time
        return (time.time() - self.last_intervention_time) > self.intervention_cooldown
    
    def _build_intervention_prompt(self, reason: str) -> str:
        recent_transcript = "\n".join([
            f"{e['speaker']}: {e['text']}" 
            for e in self.transcript_buffer[-8:]
        ])
        return f"""Recent conversation:
{recent_transcript}

Intervention trigger: {reason}

Generate a brief, calming intervention (under 20 words)."""
    
    def _referee_system_prompt(self) -> str:
        return """You are a neutral AI mediator helping startup founders resolve conflict.
Keep responses under 20 words. Never take sides. Acknowledge emotions, redirect to solutions."""


async def main():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    # Transport: Daily.co with per-participant track access
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token=os.getenv("DAILY_TOKEN"),
        bot_name="AI Referee",
        params=DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # Enable per-participant audio tracks
            transcription_enabled=False,  # We handle our own STT
        ),
    )
    
    # We'll need to dynamically create STT processors when participants join
    # This is a simplified version - production code would use event handlers
    
    # Participant STT processors (created when participants join)
    founder_a_stt = ParticipantSTTProcessor(
        participant_id="founder-a-session-id",  # Set dynamically
        speaker_label="Founder A",
        api_key=api_key,
    )
    
    founder_b_stt = ParticipantSTTProcessor(
        participant_id="founder-b-session-id",  # Set dynamically
        speaker_label="Founder B", 
        api_key=api_key,
    )
    
    # Merge transcripts from both streams
    transcript_merger = TranscriptMergerProcessor()
    
    # Referee monitor
    referee_monitor = MultiParticipantRefereeMonitor()
    
    # LLM for intervention generation
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-20250514",
    )
    
    # TTS: ElevenLabs for referee voice
    tts = ElevenLabsTTSService(
        api_key=api_key,
        voice_id=os.getenv("ELEVENLABS_REFEREE_VOICE_ID"),
        model="eleven_flash_v2_5",
        params=ElevenLabsTTSService.InputParams(
            stability=0.7,
            similarity_boost=0.75,
            optimize_streaming_latency=4,
        ),
    )
    
    # Build pipeline with parallel STT processing
    # Note: This is simplified - real implementation needs ParallelPipeline
    pipeline = Pipeline([
        transport.input(),
        # In production: ParallelPipeline([founder_a_stt, founder_b_stt])
        founder_a_stt,
        founder_b_stt,
        transcript_merger,
        referee_monitor,
        llm,
        tts,
        transport.output(),
    ])
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
    
    await task.run()


# Event handler for dynamic participant management
class ParticipantManager:
    """Manages STT instances as participants join/leave."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.stt_instances: Dict[str, ParticipantSTTProcessor] = {}
        self.participant_labels: Dict[str, str] = {}
        self.label_counter = 0
    
    def on_participant_joined(self, participant_id: str, participant_name: str = None):
        """Create STT instance for new participant."""
        if participant_id in self.stt_instances:
            return
        
        # Assign label
        label = participant_name or f"Founder {chr(65 + self.label_counter)}"
        self.label_counter += 1
        
        self.participant_labels[participant_id] = label
        self.stt_instances[participant_id] = ParticipantSTTProcessor(
            participant_id=participant_id,
            speaker_label=label,
            api_key=self.api_key,
        )
        
        print(f"Created STT instance for {label} ({participant_id})")
    
    def on_participant_left(self, participant_id: str):
        """Clean up STT instance."""
        if participant_id in self.stt_instances:
            del self.stt_instances[participant_id]
            del self.participant_labels[participant_id]
            print(f"Removed STT instance for {participant_id}")
```

### Latency Breakdown

| Component | Latency |
|-----------|---------|
| Daily WebRTC transport | ~10-15ms |
| ElevenLabs Scribe v2 Realtime (x2) | ~150ms |
| Transcript merging | <5ms |
| Analysis processing | <10ms |
| Claude Sonnet TTFT | ~300-400ms |
| ElevenLabs Flash v2.5 TTFB | ~75-100ms |
| **Total (when intervention triggers)** | **~550-680ms** |

### Cost Estimate (100 x 1-hour sessions/month)

| Service | Rate | Monthly Cost |
|---------|------|--------------|
| Daily.co (3 participants) | $0.00099/participant-min | ~$18 |
| ElevenLabs Scribe v2 (2 streams x 60 min) | ~$0.004/min | ~$48 |
| ElevenLabs TTS (~5% speak time) | $0.10/min spoken | ~$30 |
| Claude API (~100 interventions/session) | ~$0.003/intervention | ~$30 |
| Pipecat Cloud | $0.04/active-min | ~$240 |
| **Total** | | **~$366/month** |

**With your existing ElevenLabs credits:** If you have significant credits, the Scribe + TTS costs (~$78/mo) may be covered, bringing effective cost to ~$288/month.

---

## Comparison Summary

| Factor | Architecture A (Deepgram) | Architecture B (ElevenLabs) |
|--------|---------------------------|----------------------------|
| **STT Provider** | Deepgram Nova-2 | ElevenLabs Scribe v2 |
| **Speaker ID Method** | Diarization (single stream) | Separate tracks (pre-labeled) |
| **STT Latency** | ~100-200ms | ~150ms (x2 parallel) |
| **Total E2E Latency** | ~500-730ms | ~550-680ms |
| **Accuracy** | Excellent + proven diarization | Excellent, no diarization needed |
| **Uses ElevenLabs Credits** | TTS only | STT + TTS |
| **Monthly Cost (100 hrs)** | ~$344 | ~$366 (or ~$288 with credits) |
| **Complexity** | Simpler pipeline | Parallel stream management |
| **Scaling to 3+ founders** | Easy (diarization scales) | More streams to manage |

## Recommendation

**If maximizing ElevenLabs credits is priority:** Go with **Architecture B**. You get 100% ElevenLabs usage (STT + TTS), slightly better latency, and guaranteed speaker accuracy (no diarization errors).

**If simplicity and proven reliability is priority:** Go with **Architecture A**. Deepgram's diarization is battle-tested, single-stream is easier to manage, and it scales better if you add more participants later.

**Hybrid option:** Start with Architecture B to burn through credits, then evaluate switching to A if you need to scale beyond 2 founders or encounter issues with parallel stream management.
