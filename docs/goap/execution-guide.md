# GOAP Execution Guide
# Voice Referee System Implementation

This guide provides day-by-day execution instructions for implementing the Voice Referee system using the GOAP plan.

---

## Week 1: Foundation & Core Services

### Day 1: Environment Setup (Phase 1)

**Developer A Tasks:**
```bash
# Morning (2-3 hours)
1. Clone repository
2. Run setup_environment.sh
3. Install dependencies (requirements.txt)
4. Validate foundation with scripts/validate_foundation.py

# Afternoon (3-4 hours)
5. Create project structure
6. Implement configuration module (src/config/settings.py)
7. Write unit tests for config validation
8. Test .env loading
```

**Success Criteria:**
- ✅ Virtual environment active
- ✅ All dependencies installed
- ✅ Config loads from .env without errors
- ✅ `pytest tests/unit/test_config.py` passes

**Checkpoint:**
```bash
python scripts/validate_foundation.py
# Expected: All checks pass
```

---

### Day 2-3: Daily Transport & VAD (Phase 2, Part 1)

**Developer A Tasks:**

**Day 2 Morning:**
```python
# Implement Daily transport service
1. Create src/services/daily_transport.py
2. Configure DailyParams with Silero VAD
3. Write connection test method
4. Test with Daily.co test room
```

**Day 2 Afternoon:**
```python
# VAD configuration
1. Tune Silero VAD parameters (min_volume, threshold)
2. Test VAD with sample audio
3. Verify VAD triggers on speech, not silence
4. Document optimal settings
```

**Day 3:**
```python
# Integration testing
1. Write tests/integration/test_daily_transport.py
2. Test audio input streaming
3. Verify VAD events (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)
4. Measure VAD latency (target < 50ms)
```

**Success Criteria:**
- ✅ DailyTransport connects to room
- ✅ VAD triggers correctly
- ✅ Audio stream received
- ✅ Integration tests pass

**Checkpoint:**
```bash
pytest tests/integration/test_daily_transport.py -v
# Expected: All tests pass
```

---

### Day 4-5: Deepgram STT with Diarization (Phase 2, Part 2)

**Developer B Tasks:**

**Day 4:**
```python
# Deepgram service implementation
1. Create src/services/deepgram_stt.py
2. Configure LiveOptions with diarize=True
3. Test with sample audio (2 speakers)
4. Verify speaker IDs in transcript (0, 1)
```

**Day 5:**
```python
# Diarization testing & tuning
1. Test with various audio samples:
   - Clear 2-speaker conversation
   - Overlapping speech
   - Background noise
2. Measure diarization accuracy
3. Tune utterance_end_ms for optimal speaker switching
4. Document accuracy metrics
```

**Success Criteria:**
- ✅ Deepgram returns transcripts with speaker IDs
- ✅ Diarization accuracy > 80%
- ✅ STT latency < 300ms
- ✅ Speaker switching detected correctly

**Checkpoint:**
```bash
python scripts/test_deepgram_diarization.py
# Expected: Speaker 0 and Speaker 1 detected, accuracy > 80%
```

---

## Week 2: Processing Logic Implementation

### Day 6-7: Speaker Mapper & Conversation State (Phase 3, Part 1)

**Developer A Tasks:**

**Day 6:**
```python
# Speaker mapper
1. Implement src/processors/speaker_mapper.py
2. Write unit tests (test_speaker_mapper.py)
3. Test identity assignment logic
4. Verify persistence across utterances
```

**Day 7:**
```python
# Conversation state tracker
1. Implement src/processors/conversation_state.py
2. Implement Utterance and SpeakerStats dataclasses
3. Test transcript buffer (max 50 utterances)
4. Test speaker balance calculation
5. Write comprehensive unit tests
```

**Success Criteria:**
- ✅ Speaker mapping works correctly
- ✅ Transcript buffer maintains rolling window
- ✅ Speaker stats calculated accurately
- ✅ Unit tests pass with > 80% coverage

**Checkpoint:**
```bash
pytest tests/unit/test_speaker_mapper.py -v
pytest tests/unit/test_conversation_state.py -v --cov
# Expected: All tests pass, coverage > 80%
```

---

### Day 8-10: Analysis Engine (Phase 3, Part 2)

**Developer A Tasks:**

**Day 8:**
```python
# Sentiment detection
1. Implement basic keyword-based sentiment in analyzer.py
2. Test with positive/negative sample texts
3. Validate sentiment scores (-1.0 to 1.0)
```

**Day 9:**
```python
# Tension scoring
1. Implement calculate_tension_score()
2. Implement component calculations:
   - Sentiment negativity
   - Interruption rate
   - Speaker imbalance
   - Argument repetition
3. Test with mock conversation data
```

**Day 10:**
```python
# Refinement & testing
1. Tune weighted combination of tension components
2. Test with diverse conversation scenarios:
   - Calm conversation (tension < 0.3)
   - Heated argument (tension > 0.7)
   - Circular debate (high repetition)
3. Write comprehensive unit tests
4. Document threshold recommendations
```

**Success Criteria:**
- ✅ Tension score correlates with conversation intensity
- ✅ High-tension conversations score > 0.7
- ✅ Calm conversations score < 0.3
- ✅ Argument repetition detected

**Checkpoint:**
```bash
python scripts/test_tension_scoring.py
# Expected: High-tension sample > 0.7, calm sample < 0.3
```

---

### Day 11-12: Intervention Logic + LLM + TTS (Phase 3, Part 3)

**Developer A:**
```python
# Day 11: Intervention decision logic
1. Implement src/processors/decider.py
2. Implement intervention rules:
   - Tension > threshold
   - Speaker imbalance > 0.8 after 5min
   - Circular argument detection
3. Implement cooldown logic (30 seconds)
4. Test intervention triggers
```

**Developer B:**
```python
# Day 11: LLM integration (parallel)
1. Implement src/services/llm_service.py
2. Configure Claude/GPT-4o API
3. Design intervention prompt template
4. Test with sample contexts
5. Measure response latency (target < 200ms)
```

**Developer B:**
```python
# Day 12: TTS integration
1. Implement src/services/tts_service.py
2. Configure ElevenLabs Flash v2.5
3. Test voice quality
4. Measure TTS latency (target < 300ms)
5. Test WebSocket streaming
```

**Success Criteria:**
- ✅ Intervention triggers at correct thresholds
- ✅ Cooldown prevents spam
- ✅ LLM generates appropriate interventions
- ✅ TTS produces clear audio
- ✅ Combined latency < 500ms

**Checkpoint:**
```bash
python scripts/test_intervention_flow.py
# Expected:
# - Intervention triggered on high tension
# - LLM text generated
# - TTS audio saved
# - Total latency < 500ms
```

---

## Week 3: Integration, Testing & Deployment

### Day 13-14: Pipeline Assembly (Phase 4)

**Developer A + B (Pair Programming):**

**Day 13:**
```python
# Pipeline assembly
1. Implement src/pipeline/main.py
2. Connect components:
   DailyTransport → Deepgram → Processor → LLM → TTS → Daily
3. Test frame flow through pipeline
4. Add error handling at each stage
5. Implement comprehensive logging
```

**Day 14:**
```python
# Unit tests for pipeline
1. Write tests/integration/test_pipeline.py
2. Mock external services (Deepgram, LLM, TTS)
3. Test frame routing
4. Test error propagation
5. Test intervention flow end-to-end
```

**Success Criteria:**
- ✅ All components connected
- ✅ Frames flow correctly
- ✅ No blocking operations
- ✅ Integration tests pass

**Checkpoint:**
```bash
pytest tests/integration/test_pipeline.py -v
# Expected: test_full_pipeline_flow PASSED
```

---

### Day 15-17: Integration Testing (Phase 5, Part 1)

**Developer A:**
```python
# Day 15: Real-world audio testing
1. Set up Daily.co test room
2. Record diverse conversation samples:
   - Calm discussion
   - High-tension argument
   - One speaker dominating
   - Rapid turn-taking
3. Run pipeline with real audio
4. Verify STT diarization accuracy
5. Monitor intervention triggers
```

**Developer B:**
```python
# Day 15: Test automation (parallel)
1. Create automated test suite
2. Implement scripts/integration_test.py
3. Test scenarios:
   - 2-speaker calm conversation
   - High-tension intervention
   - Speaker imbalance detection
4. Collect metrics (latency, accuracy)
```

**Both Developers:**
```python
# Day 16-17: Testing & refinement
1. Run 30-minute conversation tests
2. Measure end-to-end latency
3. Tune intervention thresholds
4. Fix bugs found during testing
5. Validate false positive/negative rates
6. Manual QA of intervention quality
```

**Success Criteria:**
- ✅ End-to-end latency: 500-730ms average
- ✅ Diarization accuracy > 80%
- ✅ Intervention accuracy > 90%
- ✅ False positive rate < 20%

**Checkpoint:**
```bash
python scripts/integration_test.py --duration 1800
# Expected:
# - Latency within targets
# - Interventions appropriate
# - No crashes over 30min
```

---

### Day 18-19: Performance Validation (Phase 5, Part 2)

**Developer A:**
```python
# Performance testing
1. Implement scripts/performance_test.py
2. Measure latency at each pipeline stage:
   - Audio input → STT
   - STT → Processor
   - Processor → LLM
   - LLM → TTS
   - TTS → Audio output
3. Profile CPU/memory usage
4. Run soak test (4+ hours)
5. Check for memory leaks
```

**Developer B:**
```python
# Load testing (parallel)
1. Test multiple concurrent rooms
2. Measure performance degradation
3. Identify bottlenecks
4. Optimize slow components
5. Document performance metrics
```

**Success Criteria:**
- ✅ All latency targets met
- ✅ No memory leaks over extended runtime
- ✅ CPU usage < 50% idle
- ✅ System handles 3+ concurrent rooms

**Checkpoint:**
```bash
python scripts/performance_test.py --duration 14400
# Expected: Stable performance over 4 hours, no memory growth
```

---

### Day 20-21: Deployment (Phase 6, Part 1)

**Developer B:**
```python
# Day 20: Deployment preparation
1. Create Dockerfile
2. Set up production environment (AWS/GCP)
3. Configure environment variables in secrets manager
4. Set up systemd service or k8s deployment
5. Configure network (firewall, WebRTC ports)
```

**Developer A:**
```python
# Day 20: Deployment testing (parallel)
1. Test deployment to staging environment
2. Verify all environment variables load
3. Test Daily.co connection from production network
4. Run smoke tests
```

**Both Developers:**
```python
# Day 21: Production deployment
1. Deploy to production
2. Monitor initial startup
3. Test with production Daily.co room
4. Verify referee joins and responds
5. Monitor logs for errors
6. Prepare rollback plan
```

**Success Criteria:**
- ✅ Service running in production
- ✅ Health checks passing
- ✅ Referee joins Daily.co room
- ✅ No errors in logs

**Checkpoint:**
```bash
curl https://referee.example.com/health
# Expected: {"status": "healthy"}

python scripts/prod_test.py --room-url $PROD_ROOM_URL
# Expected: Referee joins, monitors, intervenes correctly
```

---

### Day 22-23: Monitoring & Documentation (Phase 6, Part 2)

**Developer A:**
```python
# Monitoring setup
1. Configure logging (structured JSON)
2. Set up metrics dashboard (Prometheus + Grafana)
3. Configure alerts:
   - Error rate > 5%
   - Latency > 1000ms
   - Service down
4. Test alerting (trigger test alert)
5. Document monitoring runbook
```

**Developer B:**
```python
# Documentation
1. Write deployment guide
2. Document architecture decisions
3. Create troubleshooting guide
4. Write user guide (how to use referee)
5. Document API keys setup
6. Create maintenance checklist
```

**Success Criteria:**
- ✅ Metrics dashboard operational
- ✅ Alerts configured and tested
- ✅ Complete documentation
- ✅ Runbooks for common issues

**Checkpoint:**
```bash
# Trigger test alert
python scripts/trigger_test_alert.py
# Expected: Alert received via configured channel

# View metrics
curl https://referee.example.com/metrics
# Expected: Prometheus-format metrics
```

---

## Daily Standup Template

**Format:**
```
Developer: [Name]
Date: [YYYY-MM-DD]

Yesterday:
- ✅ [Completed task 1]
- ✅ [Completed task 2]
- ⏳ [In-progress task]

Today:
- [ ] [Planned task 1]
- [ ] [Planned task 2]

Blockers:
- [None / Blocker description]

Metrics:
- Tests passing: [X/Y]
- Coverage: [Z%]
- Latency (if measured): [Xms]
```

---

## Weekly Milestones Checklist

### End of Week 1
- ✅ M1: Foundation Ready
- ✅ M2: Core Services Configured
- ✅ Daily.co connection working
- ✅ Deepgram diarization > 80% accuracy

### End of Week 2
- ✅ M3: Processing Logic Complete
- ✅ Speaker mapping working
- ✅ Tension scoring validated
- ✅ Intervention triggers correct
- ✅ LLM + TTS integrated

### End of Week 3
- ✅ M4: Pipeline Integrated
- ✅ M5a: Validated (performance tests pass)
- ✅ M5: Deployed & Monitored
- ✅ Production ready

---

## Troubleshooting Guide

### Issue: Diarization accuracy < 80%
**Symptoms:** Speaker IDs incorrect, frequent speaker switches
**Diagnosis:**
```bash
python scripts/test_diarization_accuracy.py --audio samples/test_audio.wav
```
**Solutions:**
1. Check audio quality (suppress background noise)
2. Adjust utterance_end_ms (try 800ms or 1200ms)
3. Test with different audio samples
4. Consider Architecture B if accuracy doesn't improve

---

### Issue: End-to-end latency > 1000ms
**Symptoms:** Slow interventions, delayed audio
**Diagnosis:**
```bash
python scripts/profile_pipeline.py
# Shows latency at each stage
```
**Solutions:**
1. If STT > 400ms: Reduce utterance_end_ms
2. If LLM > 300ms: Use streaming, cache prompts, or switch to faster model
3. If TTS > 400ms: Use lower quality, enable caching
4. Check network latency to APIs

---

### Issue: False positive interventions (> 20%)
**Symptoms:** Referee interrupts calm conversations
**Diagnosis:**
```bash
python scripts/analyze_false_positives.py --logs production.log
```
**Solutions:**
1. Increase tension threshold (0.7 → 0.8)
2. Require 2 consecutive high readings
3. Extend cooldown (30s → 60s)
4. Improve sentiment model

---

### Issue: Memory leak over extended runtime
**Symptoms:** Memory usage grows continuously
**Diagnosis:**
```bash
python scripts/memory_profiler.py --duration 3600
```
**Solutions:**
1. Check transcript_buffer cleanup
2. Verify VAD analyzer cleanup
3. Check for unclosed connections
4. Use weakref for circular references

---

## Success Criteria Summary

**Phase 1 (Day 1):**
- ✅ Environment setup complete
- ✅ Dependencies installed
- ✅ Config module validates

**Phase 2 (Days 2-5):**
- ✅ Daily.co connection working
- ✅ Silero VAD triggers correctly
- ✅ Deepgram diarization > 80% accurate

**Phase 3 (Days 6-12):**
- ✅ Speaker mapping persistent
- ✅ Conversation state tracked
- ✅ Tension scoring validated
- ✅ Intervention logic correct
- ✅ LLM generates appropriate text
- ✅ TTS produces clear audio

**Phase 4 (Days 13-14):**
- ✅ Pipeline assembled
- ✅ Integration tests pass

**Phase 5 (Days 15-19):**
- ✅ End-to-end latency: 500-730ms
- ✅ Intervention accuracy > 90%
- ✅ False positive rate < 20%
- ✅ Performance stable over 4+ hours

**Phase 6 (Days 20-23):**
- ✅ Deployed to production
- ✅ Monitoring operational
- ✅ Documentation complete

---

## Post-Deployment Checklist

**Week 4+: Monitoring & Iteration**

1. **Daily Monitoring (First Week)**
   - Check metrics dashboard
   - Review error logs
   - Monitor intervention quality
   - Collect user feedback

2. **Weekly Reviews**
   - Analyze intervention patterns
   - Review false positive/negative rates
   - Check performance trends
   - Plan optimizations

3. **Monthly Improvements**
   - Retrain sentiment model with real data
   - Tune intervention thresholds
   - Update documentation
   - Plan feature enhancements

4. **Ongoing Maintenance**
   - Update dependencies monthly
   - Monitor API changes (Deepgram, ElevenLabs, etc.)
   - Review and update tests
   - Optimize based on production metrics

---

**END OF EXECUTION GUIDE**

This guide provides concrete, day-by-day instructions for implementing the Voice Referee system. Each day has clear tasks, success criteria, and checkpoints to ensure progress aligns with the GOAP plan.
