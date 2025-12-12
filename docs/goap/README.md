# Voice Referee System - GOAP Implementation Plan

This directory contains the complete Goal-Oriented Action Plan (GOAP) for implementing the Voice Referee system using Architecture A: Mixed Audio + Deepgram Diarization.

## 📚 Documentation Structure

### 1. [voice-referee-goap.md](./voice-referee-goap.md)
**The Master Plan** - Complete GOAP with:
- State definitions (current → goal)
- 18 detailed actions with preconditions, effects, costs
- Dependency graph
- Optimal execution sequence (A* path)
- Success criteria & validation checkpoints
- Risk assessment & mitigation strategies
- Replanning triggers

**Use this for:** Overall project planning, milestone tracking, decision-making

### 2. [action-templates.md](./action-templates.md)
**Implementation Templates** - Ready-to-use code templates:
- Python module templates for each component
- Configuration classes (Pydantic models)
- Service implementations (Daily, Deepgram, LLM, TTS)
- Processor implementations (speaker mapper, analyzer, decider)
- Pipeline assembly
- Unit test examples

**Use this for:** Copy-paste implementation, coding standards reference

### 3. [execution-guide.md](./execution-guide.md)
**Day-by-Day Execution** - Tactical guide with:
- 23-day implementation timeline
- Daily task breakdowns for 2 developers
- Parallel execution opportunities
- Success criteria checkpoints
- Troubleshooting guide
- Standup templates

**Use this for:** Sprint planning, daily task assignments, progress tracking

## 🎯 Quick Start

### For Project Managers
1. Read: `voice-referee-goap.md` (Section 1-3, 9)
2. Review: Execution sequence and timeline
3. Track: Weekly milestones in `execution-guide.md`

### For Developers
1. Start: `execution-guide.md` → Day 1 tasks
2. Reference: `action-templates.md` → Copy code templates
3. Validate: Use checkpoints from `voice-referee-goap.md`

### For Technical Leads
1. Review: Complete `voice-referee-goap.md`
2. Plan: Resource allocation using dependency graph
3. Monitor: Risk triggers and replanning conditions

## 📊 Project Overview

**Timeline:** 2-3 weeks (with 2 developers)
**Total Actions:** 18 actions across 6 phases
**Total Cost:** 59 action units
**Team Size:** 2 developers (1 senior, 1 mid-level)

### Milestones

| Milestone | Phase | Duration | Success Criteria |
|-----------|-------|----------|------------------|
| M1: Foundation Ready | Phase 1 | 1 day | Env setup, deps installed, config validated |
| M2: Core Services | Phase 2 | 3-4 days | Daily.co + Deepgram working, diarization > 80% |
| M3: Processing Logic | Phase 3 | 5-6 days | Referee logic complete, LLM + TTS integrated |
| M4: Pipeline Integrated | Phase 4 | 2 days | End-to-end flow working, tests passing |
| M5a: Validated | Phase 5 | 3-4 days | Latency < 730ms, accuracy > 90% |
| M5: Deployed | Phase 6 | 2-3 days | Production ready, monitoring active |

## 🔧 Technology Stack

- **Framework:** Pipecat Cloud
- **Language:** Python 3.10+
- **Transport:** Daily.co WebRTC
- **STT:** Deepgram Nova-2 (with diarization)
- **VAD:** Silero VAD
- **LLM:** Claude 3.5 Sonnet / GPT-4o
- **TTS:** ElevenLabs Flash v2.5
- **Testing:** pytest, pytest-asyncio
- **Monitoring:** Prometheus + Grafana

## 📈 Success Metrics

### Performance Targets
- End-to-end latency: **500-730ms** (average)
- STT latency: **< 300ms**
- LLM latency: **< 200ms**
- TTS latency: **< 300ms**
- Diarization accuracy: **> 80%**

### Quality Targets
- Intervention accuracy: **> 90%**
- False positive rate: **< 20%**
- False negative rate: **< 30%**
- Uptime: **> 99%**

## 🚨 Critical Dependencies

### API Keys Required
```bash
DAILY_ROOM_URL=https://example.daily.co/referee-test
DAILY_TOKEN=<bot_token>
DEEPGRAM_API_KEY=<api_key>
ANTHROPIC_API_KEY=<api_key>  # or OPENAI_API_KEY
ELEVENLABS_API_KEY=<api_key>
ELEVENLABS_REFEREE_VOICE_ID=<voice_id>
```

### External Services
- Daily.co (WebRTC rooms)
- Deepgram (STT with diarization)
- Anthropic/OpenAI (LLM for interventions)
- ElevenLabs (TTS for referee voice)

## 🎨 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Voice Referee Pipeline                  │
└─────────────────────────────────────────────────────────────┘

 Daily.co Room (Mixed Audio from 2 Founders)
           ↓
    [DailyTransport]
           ↓
    [Silero VAD] ← Voice Activity Detection
           ↓
    [Deepgram STT] ← Transcription + Diarization
           ↓
    [RefereeMonitorProcessor]
      ├── Speaker Mapper (0 → Founder A, 1 → Founder B)
      ├── Conversation State (transcript buffer, stats)
      ├── Analysis Engine (tension scoring, sentiment)
      └── Intervention Decider (threshold checks, cooldown)
           ↓
    [LLM Service] ← Generate intervention text
           ↓
    [TTS Service] ← Convert to speech
           ↓
    [DailyTransport] → Audio back to room
```

## 🔄 GOAP Planning Process

The plan uses Goal-Oriented Action Planning (GOAP) with A* pathfinding:

1. **State Assessment:** Define current state vs. goal state
2. **Action Analysis:** Identify actions with preconditions & effects
3. **Plan Generation:** Find optimal path using A* search
4. **Execution Monitoring:** Track progress, detect deviations
5. **Dynamic Replanning:** Adjust plan when triggers occur

### Replanning Triggers
- Diarization accuracy < 70% → Consider Architecture B
- Latency > 1500ms → Profile and optimize
- False positive rate > 20% → Tune thresholds
- LLM API down → Use fallback templates

## 📝 Implementation Phases

### Phase 1: Foundation (Days 1-2)
- Setup environment (Python 3.10+, venv)
- Install dependencies (Pipecat, Daily, Deepgram, etc.)
- Create project structure
- Implement configuration module

### Phase 2: Core Services (Days 3-5)
- Configure Daily.co transport
- Setup Silero VAD
- Configure Deepgram STT with diarization

### Phase 3: Processing Logic (Days 6-12)
- Implement speaker mapping
- Build conversation state tracker
- Create analysis engine (tension scoring)
- Implement intervention decision logic
- Integrate LLM for text generation
- Integrate TTS for voice output

### Phase 4: Pipeline Integration (Days 13-14)
- Assemble full pipeline
- Write unit tests (target: 80% coverage)

### Phase 5: Validation (Days 15-19)
- Integration testing with real audio
- Performance validation (latency, accuracy)
- Load testing

### Phase 6: Deployment (Days 20-23)
- Deploy to production
- Setup monitoring & alerting
- Complete documentation

## 🧪 Testing Strategy

### Unit Tests (Continuous)
- Test individual components in isolation
- Mock external services
- Target: 80%+ coverage

### Integration Tests (After each phase)
- Test component interactions
- Use real APIs (dev keys)
- Validate frame flow

### End-to-End Tests (Phase 5)
- Real Daily.co rooms with 2 participants
- Record conversations for consistency
- Manual QA for intervention quality

### Performance Tests (Phase 5)
- Measure latency at each stage
- Load test with multiple concurrent rooms
- 24-hour soak test for memory leaks

## 📞 Support & Resources

### Documentation
- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Daily.co API Docs](https://docs.daily.co/)
- [Deepgram API Docs](https://developers.deepgram.com/)
- [ElevenLabs API Docs](https://docs.elevenlabs.io/)

### Code Repository
- Templates: `docs/goap/action-templates.md`
- Scripts: `scripts/` (validation, testing, profiling)
- Tests: `tests/` (unit, integration)

### Team Communication
- Daily standups: Use template in `execution-guide.md`
- Weekly reviews: Track against milestones
- Issue tracking: Document blockers and solutions

## 🎯 Next Steps

1. **Week 1:** Review this README and master plan
2. **Day 1:** Start with `execution-guide.md` → Day 1 tasks
3. **Daily:** Use checkpoints to validate progress
4. **Weekly:** Review milestones and adjust plan if needed

---

**Generated:** 2025-12-11
**Plan Version:** 1.0
**Architecture:** Mixed Audio + Deepgram Diarization
**Framework:** Pipecat Cloud
