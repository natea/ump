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
| 1.1 | Setup Python environment (venv, .env template) | ✅ | Claude | None | Python 3.10+ with pipecat |
| 1.2 | Install dependencies (pipecat, daily, deepgram, etc.) | ✅ | Claude | 1.1 | requirements.txt created |
| 1.3 | Create project structure | ✅ | Claude | 1.1 | See structure below |
| 1.4 | Create config module with Pydantic validation | ✅ | Claude | 1.3 | settings.py with env validation |

### Project Structure ✅
```
voice_referee/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          ✅
│   │   └── daily_config.py      ✅
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── speaker_mapper.py    ✅ (with Dynamic participant names)
│   │   ├── conversation_state.py ✅
│   │   ├── analyzer.py          ✅
│   │   ├── decider.py           ✅
│   │   └── referee_monitor.py   ✅
│   ├── analysis/
│   │   └── conversation_analyzer.py ✅
│   ├── decision/
│   │   └── intervention_decider.py  ✅
│   ├── services/
│   │   ├── __init__.py
│   │   ├── daily_transport.py   ✅ (with participant event handlers)
│   │   ├── deepgram_stt.py      ✅ (with DiarizedTranscriptionFrame)
│   │   ├── llm_service.py       ✅
│   │   └── tts_service.py       ✅
│   └── pipeline/
│       ├── __init__.py
│       └── main.py              ✅
├── tests/
│   ├── unit/                    ✅ (80%+ coverage)
│   └── integration/             ✅ (scaffolding)
├── requirements.txt             ✅
├── .env.example                 ✅
└── run.py                       ✅
```

---

## Phase 2: Core Services Configuration (Days 3-5)
**Milestone**: M2 - Core Services Configured
**Cost**: 9 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 2.1 | Configure DailyTransport with WebRTC | ✅ | Claude | 1.2, 1.4 | Public room support, event handlers |
| 2.2 | Configure Silero VAD | ✅ | Claude | 2.1 | SileroVADAnalyzer integrated |
| 2.3 | Configure Deepgram STT with diarization | ✅ | Claude | 1.2, 1.4, 2.2 | DiarizedDeepgramSTTService with speaker extraction |

### Key Configuration ✅
```python
# Deepgram Settings (from .env)
model: "nova-2"
language: "en-US"
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
| 3.1a | Implement SpeakerMapper | ✅ | Claude | 2.3 | Now uses actual participant names from Daily |
| 3.1b | Implement ConversationState tracker | ✅ | Claude | 3.1a | 50 utterance buffer, speaker stats |
| 3.1c | Implement ConversationAnalyzer | ✅ | Claude | 3.1b | Tension scoring, pattern detection |
| 3.1d | Implement InterventionDecider | ✅ | Claude | 3.1c | Proactive triggers + cooldown |
| 3.2 | Integrate LLM (Claude Sonnet 4) | ✅ | Claude | 3.1d, 1.4 | Full mediation prompt |
| 3.3 | Integrate ElevenLabs TTS | ✅ | Claude | 3.2, 1.4 | Flash v2.5, Rachel voice |

### Intervention Thresholds (Updated)
```python
# Current Settings (.env)
TENSION_THRESHOLD=0.1        # Low threshold for active engagement
COOLDOWN_SECONDS=10          # Short cooldown for frequent check-ins
BUFFER_SIZE=50               # 50 utterance buffer

# Proactive Triggers (intervention_decider.py)
- Every 5 utterances → Check-in
- 3+ consecutive same speaker → Balance prompt
- tension_score > threshold → Intervention
```

### AI Mediator Prompt ✅
Comprehensive mediation facilitator prompt with:
- Two-speaker protocol (confirms presence, names addressee)
- Short responses (1-3 sentences, voice-optimized)
- Reframing techniques ("He never listens" → "Being heard matters to you")
- Intervention strategies (heated/stuck/quiet situations)
- Clear boundaries (legal, safety, impasse)
- Dynamic participant names from Daily.co

---

## Phase 4: Pipeline Integration (Days 13-16)
**Milestone**: M4 - Pipeline Integrated
**Cost**: 9 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 4.1 | Assemble Pipecat pipeline | ✅ | Claude | 2.1, 2.3, 3.1d, 3.3 | Full pipeline assembled |
| 4.2 | Write unit tests (>80% coverage) | ✅ | Claude | 3.x | All processors tested |

### Pipeline Flow ✅
```
DailyTransport (audio input)
    ↓
    ├→ on_participant_joined → SpeakerMapper.register_participant()
    ↓
SileroVADAnalyzer (voice detection)
    ↓
DiarizedDeepgramSTTService (transcription + diarization)
    ↓
    └→ DiarizedTranscriptionFrame (with speaker attribute)
    ↓
RefereeMonitorProcessor (analysis + decision)
    ├→ ConversationAnalyzer.analyze()
    ├→ InterventionDecider.decide()
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
| 5.1 | Integration test full pipeline | 🔄 | - | 4.1, 4.2 | Testing in progress |
| 5.2 | Performance validation | ⬜ | - | 5.1 | Target: 500-730ms latency |

### Test Scenarios
1. ✅ Two-speaker calm conversation → No intervention (working)
2. ✅ Proactive check-in → Triggers every 5 utterances
3. ✅ Speaker imbalance → Triggers after 3 consecutive
4. 🔄 High-tension conversation → Testing in progress
5. ⬜ Edge cases: single speaker, rapid switching, background noise

### Current Issues Being Validated
- [x] LLMService returns proper FrameProcessor
- [x] Daily.co public room authentication
- [x] Deepgram diarization speaker extraction
- [x] Participant name registration from Daily events
- [x] AI mediator prompt updated
- [ ] End-to-end TTS output verification

---

## Phase 6: Deployment (Days 21-23)
**Milestone**: M5 - Deployed & Monitored
**Cost**: 5 units

| ID | Task | Status | Assignee | Dependencies | Notes |
|----|------|--------|----------|--------------|-------|
| 6.1 | Deploy to production | ⬜ | - | 5.2 | Docker + staging first |
| 6.2 | Setup monitoring | ⬜ | - | 6.1 | Prometheus + Grafana |

---

## Recent Features Implemented

### Dynamic Participant Names (2025-12-11)
- SpeakerMapper now registers participants from Daily.co join events
- Uses actual display names instead of "Founder A"/"Founder B"
- Participant callbacks wired from VoiceRefereeTransport to RefereeMonitor

### AI Mediator Prompt (2025-12-11)
- Comprehensive two-speaker mediation protocol
- Dynamic name substitution in prompts
- Short, voice-optimized responses (1-3 sentences)
- Reframing techniques and intervention strategies
- Clear boundaries for legal/safety/impasse situations

### Proactive Engagement (2025-12-11)
- Periodic check-ins every 5 utterances
- Balance prompts after 3 consecutive same-speaker turns
- Lower tension threshold (0.1) for more active engagement
- Shorter cooldown (10s) for frequent interaction

---

## Progress Log

| Date | Update |
|------|--------|
| 2025-12-11 | Task list created from GOAP plan |
| 2025-12-11 | Phase 1-4 completed - full pipeline working |
| 2025-12-11 | Fixed LLMService FrameProcessor issue |
| 2025-12-11 | Fixed Daily.co public room authentication |
| 2025-12-11 | Fixed Deepgram diarization speaker extraction |
| 2025-12-11 | Added referee introduction message |
| 2025-12-11 | Made referee proactive with check-ins |
| 2025-12-11 | Added dynamic participant names from Daily |
| 2025-12-11 | Updated AI mediator prompt for better facilitation |

---

## Environment Variables Required

```bash
# Daily.co
DAILY_ROOM_URL=https://ump.daily.co/founders
DAILY_TOKEN=<api_key_or_meeting_token>

# Deepgram
DEEPGRAM_API_KEY=<api_key>
DEEPGRAM_MODEL=nova-2
DEEPGRAM_DIARIZE=true

# LLM
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=<api_key>

# TTS
ELEVENLABS_API_KEY=<api_key>
TTS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
TTS_MODEL=eleven_flash_v2_5

# Configuration
TENSION_THRESHOLD=0.1
COOLDOWN_SECONDS=10
BUFFER_SIZE=50
LOG_LEVEL=INFO
```

---

## Next Steps

1. **Validate TTS Output** - Confirm referee voice is audible in Daily room
2. **Test Full Mediation Session** - Run through complete scenario with two founders
3. **Performance Profiling** - Measure actual latency against 500-730ms target
4. **Edge Case Testing** - Single speaker, rapid switching, background noise
5. **Production Deployment** - Docker container, monitoring setup

---

## Future Feature Requests

### Authentication & User Management

#### FR-1: Login and Registration System
**Priority**: High | **Complexity**: Medium

Implement user authentication. Choose between:
- **Supabase Auth** - Open source, PostgreSQL-based, includes Row Level Security
- **Clerk** - Managed service, better prebuilt UI components

Requirements:
- Email/password registration with verification
- Password reset flow
- Social login (Google, GitHub)
- Session management
- Protected API endpoints

---

#### FR-2: Invite System for Second Participant
**Priority**: High | **Complexity**: Medium

Allow session initiator to invite co-founder via:
- Unique join codes/links (`voicereferee.com/join/ABC123`)
- Email invites sent from platform
- SMS invites (via Twilio)
- Copy-to-clipboard for manual sharing

Features:
- Time-limited invites (24-48 hours)
- Prevent reuse after session starts
- Track invite status (pending/accepted)

---

### Monetization & Growth

#### FR-3: Stripe Subscription Pricing
**Priority**: High | **Complexity**: High

Implement subscription model:
- **Free Tier**: 1 teaser session (15-30 min), no CC required
- **Pro Tier** ($X/month): X sessions/month, full length, recordings
- **Enterprise**: Unlimited, team accounts, custom integrations

Features:
- Stripe Checkout integration
- Customer portal for subscription management
- Webhook handling for subscription events
- Usage tracking and enforcement
- Failed payment handling

---

#### FR-4: Referral System
**Priority**: Medium | **Complexity**: Medium

"Invite a friend, both get a free session"

Mechanics:
- Unique referral links per user (`voicereferee.com/r/USERCODE`)
- Credit awarded when referee completes first session
- Dashboard showing referral status
- Anti-fraud measures (prevent self-referral)

---

### Communication Channels

#### FR-5: Twilio Phone Integration
**Priority**: Medium | **Complexity**: High

Enable mediation via phone calls:
- System calls both participants
- Bridges calls into conference with AI mediator
- No computer/app required

Technical options:
- Twilio Conference + Daily.co bridge
- Pipecat's Twilio transport directly

Considerations:
- Call status tracking (ringing, answered, disconnected)
- Cost tracking (~$0.013/min US, varies internationally)

---

#### FR-6: Calendar Scheduling with Google Calendar
**Priority**: Medium | **Complexity**: High

Integrate with co-founders' Google Calendars:
- OAuth connection to both calendars
- Analyze free/busy across both participants
- Suggest optimal meeting times
- Book recurring sessions (weekly, bi-weekly)
- Create events with join links
- Send reminders before sessions

Technical options:
- Direct Google Calendar API
- [Composio](https://composio.dev/) for simplified OAuth/integration

---

### AI & Analysis Enhancements

#### FR-7: Video Emotion Detection
**Priority**: Low | **Complexity**: High

Add visual sentiment analysis from video:
- Detect anger, contempt, disgust, fear, sadness, disengagement
- Supplement audio tension scoring
- Real-time processing (< 500ms latency)

Reference implementations:
- https://dev.to/blockopensource/why-i-used-goose-to-build-a-chaotic-emotion-detection-app-3979
- https://github.com/blackgirlbytes/chaotic-emotion-detector

Technical options:
- Browser-based: TensorFlow.js + face-api.js (privacy-friendly)
- Server-side: Python fer/deepface (more accurate)

Integration:
```python
tension_score = weighted_sum(
    audio_sentiment * 0.25,
    interruption_rate * 0.25,
    speaker_imbalance * 0.15,
    argument_repetition * 0.15,
    facial_emotion_negativity * 0.20  # NEW
)
```

---

#### FR-8: HeyGen Video Avatar for AI Mediator
**Priority**: Low | **Complexity**: Medium

Give the AI referee a visual body using HeyGen's streaming avatar:
- Real-time lip-sync to TTS output
- Professional mediator appearance
- Non-verbal communication cues

Reference implementation:
- https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/43a-heygen-video-service.py

Integration with existing pipeline:
```
LLM Response → ElevenLabs TTS → HeyGen Avatar → Daily.co Video Output
```

Considerations:
- Additional latency from avatar rendering
- HeyGen API costs
- User preference (some may prefer audio-only)

---

### Personalization

#### FR-9: Voice Selection (Male/Female)
**Priority**: Medium | **Complexity**: Low

Allow users to choose AI mediator voice:
- Preview samples before selection
- Male and female options
- Remember preference for future sessions

Recommended voices:
- Female: Rachel (calm, professional), Domi (neutral), Bella (warm)
- Male: Adam (professional), Antoni (trustworthy), Josh (clear)

---

#### FR-10: Multi-Language Support & Translation
**Priority**: Medium | **Complexity**: Very High

Support sessions in multiple languages:

**Same language (non-English):**
Both speak Spanish → Session in Spanish

**Translation mode:**
Founder A (English) ↔ Founder B (Mandarin):
- Each hears the other translated
- AI speaks to each in their language

Initial languages: English, Spanish, French, German, Portuguese, Mandarin, Japanese, Korean, Italian, Dutch

Considerations:
- Translation adds 200-400ms latency
- Additional API costs (Google/DeepL ~$20-25/1M chars)
- Cultural adaptation of mediation style

---

### Feature Request Status Legend
- ⬜ Not Started
- 📋 Specified
- 🔄 In Development
- ✅ Completed

| ID | Feature | Priority | Status |
|----|---------|----------|--------|
| FR-1 | Login/Registration | High | ⬜ |
| FR-2 | Invite System | High | ⬜ |
| FR-3 | Stripe Pricing | High | ⬜ |
| FR-4 | Referral System | Medium | ⬜ |
| FR-5 | Twilio Phone | Medium | ⬜ |
| FR-6 | Calendar Scheduling | Medium | ⬜ |
| FR-7 | Video Emotion Detection | Low | ⬜ |
| FR-8 | HeyGen Avatar | Low | ⬜ |
| FR-9 | Voice Selection | Medium | ⬜ |
| FR-10 | Multi-Language | Medium | ⬜ |
