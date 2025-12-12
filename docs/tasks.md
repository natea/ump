# Voice Referee Implementation Tasks

**Source**: [GOAP Plan](goap/voice-referee-goap.md)
**Architecture**: A - Mixed Audio + Deepgram Diarization
**Target Latency**: 500-730ms end-to-end

---

## Status Legend
- ⬜ Not Started
- 🔄 In Progress
- ✅ Completed
- ❌ Blocked
- ⏸️ Paused

---

## Phase 1: Foundation Setup (Days 1-2)
**Milestone**: M1 - Foundation Ready
**Cost**: 6 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 1.1 | Setup Python environment (venv, .env template) | ⬜ | - | None | Python 3.10+ required |
| 1.2 | Install dependencies (pipecat, daily, deepgram, etc.) | ⬜ | - | 1.1 | Pin versions in requirements.txt |
| 1.3 | Create project structure | ⬜ | - | 1.1 | See structure below |
| 1.4 | Create config module with Pydantic validation | ⬜ | - | 1.3 | Validate all API keys |

### Project Structure
```
voice_referee/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── speaker_mapper.py
│   │   ├── conversation_state.py
│   │   ├── analyzer.py
│   │   ├── decider.py
│   │   └── referee_monitor.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── daily_transport.py
│   │   ├── deepgram_stt.py
│   │   ├── llm_service.py
│   │   └── tts_service.py
│   └── pipeline/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Phase 2: Core Services Configuration (Days 3-5)
**Milestone**: M2 - Core Services Configured
**Cost**: 9 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 2.1 | Configure DailyTransport with WebRTC | ⬜ | - | 1.2, 1.4 | audio_in/out enabled, no camera |
| 2.2 | Configure Silero VAD | ⬜ | - | 2.1 | min_volume: 0.6, latency < 50ms |
| 2.3 | Configure Deepgram STT with diarization | ⬜ | - | 1.2, 1.4, 2.2 | **HIGH RISK** - Nova-2, diarize=true |

### Key Configuration
```python
# Deepgram Settings
model: "nova-2"
language: "en"
diarize: true
punctuate: true
interim_results: true
smart_format: true
utterance_end_ms: 1000
```

---

## Phase 3: Processing Logic Implementation (Days 6-12)
**Milestone**: M3 - Processing Logic Complete
**Cost**: 20 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 3.1a | Implement SpeakerMapper | ⬜ | - | 2.3 | Map speaker IDs to Founder A/B |
| 3.1b | Implement ConversationState tracker | ⬜ | - | 3.1a | Buffer 50 utterances, track stats |
| 3.1c | Implement ConversationAnalyzer | ⬜ | - | 3.1b | **HIGH RISK** - Tension scoring |
| 3.1d | Implement InterventionDecider | ⬜ | - | 3.1c | Thresholds + cooldown logic |
| 3.2 | Integrate LLM (Claude/GPT-4o) | ⬜ | - | 3.1d, 1.4 | Intervention text generation |
| 3.3 | Integrate ElevenLabs TTS | ⬜ | - | 3.2, 1.4 | Flash v2.5, WebSocket streaming |

### Intervention Thresholds (from UI screenshots)
Based on the ump.ai interface protocols:
1. **No Interruptions** - Allow complete thoughts
2. **Data Over Opinion** - Cite specific metrics
3. **Future Focused** - No dredging past issues
4. **Binary Outcome** - Commit to decision by session end

```python
# Decision Rules
IF tension_score > 0.7 AND cooldown_elapsed:
    → INTERVENE ("High tension detected")
IF same_argument_count > 3 AND cooldown_elapsed:
    → INTERVENE ("Circular argument detected")
IF speaker_imbalance > 0.8 AND duration > 5min AND cooldown_elapsed:
    → INTERVENE ("One speaker dominating")
ELSE:
    → OBSERVE
```

---

## Phase 4: Pipeline Integration (Days 13-16)
**Milestone**: M4 - Pipeline Integrated
**Cost**: 9 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 4.1 | Assemble Pipecat pipeline | ⬜ | - | 2.1, 2.3, 3.1d, 3.3 | **HIGH RISK** - Integration |
| 4.2 | Write unit tests (>80% coverage) | ⬜ | - | 3.x | Mock external services |

### Pipeline Flow
```
DailyTransport (audio input)
    ↓
SileroVADAnalyzer (voice detection)
    ↓
DeepgramSTTService (transcription + diarization)
    ↓
RefereeMonitorProcessor (analysis + decision)
    ↓ (if intervention needed)
AnthropicLLMService (generate intervention text)
    ↓
ElevenLabsTTSService (text → audio)
    ↓
DailyTransport (audio output to room)
```

---

## Phase 5: Validation & Testing (Days 17-20)
**Milestone**: M5a - Validated
**Cost**: 10 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 5.1 | Integration test full pipeline | ⬜ | - | 4.1, 4.2 | **HIGH RISK** - Real Daily.co room |
| 5.2 | Performance validation | ⬜ | - | 5.1 | Target: 500-730ms latency |

### Test Scenarios
1. Two-speaker calm conversation → No intervention
2. High-tension conversation → Intervention at threshold
3. Speaker imbalance → Intervention after 5 minutes
4. Edge cases: single speaker, rapid switching, background noise

### Performance Targets
| Component | Target Latency |
|-----------|---------------|
| Daily WebRTC | ~10-15ms |
| Deepgram STT | < 300ms |
| Analysis | < 10ms |
| Claude LLM | < 200ms |
| ElevenLabs TTS | < 300ms |
| **Total** | **500-730ms** |

---

## Phase 6: Deployment (Days 21-23)
**Milestone**: M5 - Deployed & Monitored
**Cost**: 5 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 6.1 | Deploy to production | ⬜ | - | 5.2 | Docker + staging first |
| 6.2 | Setup monitoring | ⬜ | - | 6.1 | Prometheus + Grafana |

---

## UI Integration Requirements

Based on screenshots from `docs/screenshots/`:

### Session UI Components
- **Talk Balance Meter**: Shows YOU vs COUNTERPARTY speaking percentage
- **Protocol Sidebar**: Lists active rules (No Interruptions, Data Over Opinion, etc.)
- **System Intervention Panel**: Displays referee messages with monospace font
- **Session Timer**: Shows elapsed time (00:29 format)
- **Microphone Button**: Large circular button for voice input
- **End Session**: Red text button in top right

### Intervention Message Types
1. **SYSTEM INTERVENTION** (Blue header)
   - Session initialization message
   - Protocol enforcement reminders

2. **PROTOCOL VIOLATION** (Red, Rule X)
   - Specific rule violated
   - Corrective guidance

3. **PROTOCOL WARNING** (Red, Rule X)
   - Warning before violation
   - Suggested alternative phrasing

### Example Interventions (from screenshots)
```
PROTOCOL VIOLATION (Rule 3): Future Focused.
Referring to 'last year' is not relevant to the Q3 sprint decision.
Please restate your objection using forward-looking impact.
```

```
PROTOCOL WARNING (Rule 2): Data Over Opinion.
Avoid starting sentences with 'I feel'. Counter the 20% drop-off
statistic with data about Enterprise contract value.
```

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Diarization accuracy < 70% | HIGH | Test with diverse audio, fallback to Architecture B |
| Latency > 1500ms | HIGH | Profile pipeline, optimize bottlenecks |
| False positive rate > 20% | MEDIUM | Increase thresholds, add cooldown |
| LLM API unavailable | MEDIUM | Pre-defined fallback templates |

---

## Replanning Triggers

1. **Diarization < 70%** → Switch to Architecture B (separate tracks)
2. **Latency > 1500ms** → Profile and optimize
3. **False positives > 20%** → Tune thresholds
4. **LLM down** → Use template fallbacks
5. **Production errors > 8%** → Rollback and debug

---

## Progress Log

| Date | Update |
|------|--------|
| 2025-12-11 | Task list created from GOAP plan |

---

## Environment Variables Required

```bash
# Daily.co
DAILY_ROOM_URL=https://example.daily.co/referee-room
DAILY_TOKEN=<bot_token>

# Deepgram
DEEPGRAM_API_KEY=<api_key>

# LLM
ANTHROPIC_API_KEY=<api_key>

# TTS
ELEVENLABS_API_KEY=<api_key>
ELEVENLABS_REFEREE_VOICE_ID=<voice_id>

# Configuration
INTERVENTION_TENSION_THRESHOLD=0.7
INTERVENTION_COOLDOWN_SECONDS=30
TRANSCRIPT_BUFFER_SIZE=50
```
