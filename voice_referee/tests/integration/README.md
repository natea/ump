# Voice Referee Integration Tests

Comprehensive integration tests for the Voice Referee pipeline testing end-to-end conversation flow with mocked external services.

## Test Coverage

### Test Scenarios

1. **test_two_speaker_calm_conversation**
   - Balanced, calm conversation between two founders
   - Verifies NO intervention triggered for healthy dialogue
   - Validates speaker mapping to Founder A/B
   - Checks low tension score and balanced speaking time

2. **test_high_tension_conversation**
   - High tension with conflict keywords ("wrong", "fault", "always")
   - Verifies intervention triggered when tension > 0.7
   - Validates LLM receives correct intervention context
   - Confirms system prompt includes referee guidance

3. **test_speaker_imbalance**
   - One speaker dominates with 80%+ talk time
   - Verifies intervention after sustained imbalance
   - Validates "SYSTEM INTERVENTION" type
   - Confirms dominant speaker identification

4. **test_protocol_violation_past_reference**
   - Protocol 3 (Future Focused) violation detection
   - Speaker references past issues ("last year we agreed...")
   - Verifies pattern detection for past-focused language
   - Validates corrective guidance would be provided

5. **test_protocol_warning_opinion**
   - Protocol 2 (Data Over Opinion) warning detection
   - Speaker uses opinion phrases without data ("I feel like...")
   - Verifies detection of opinion vs data indicators
   - Confirms warning mechanism works

6. **test_intervention_cooldown**
   - Cooldown period enforcement between interventions
   - First intervention succeeds
   - Second intervention blocked during cooldown
   - Third intervention allowed after cooldown expires

7. **test_full_pipeline_integration**
   - Complete realistic conversation flow
   - Progression: calm → tension → intervention → recovery
   - Validates entire pipeline coordination
   - Tracks tension scores throughout conversation

## Mocked Services

All external APIs are mocked for offline testing:

- **Daily.co Transport**: WebRTC audio/video connection
- **Deepgram STT**: Speech-to-text with speaker diarization
- **Anthropic Claude**: LLM for intervention generation
- **ElevenLabs TTS**: Text-to-speech output

## Running Tests

### All Integration Tests

```bash
pytest voice_referee/tests/integration/test_pipeline.py -v
```

### Specific Test

```bash
pytest voice_referee/tests/integration/test_pipeline.py::test_two_speaker_calm_conversation -v
```

### With Coverage

```bash
pytest voice_referee/tests/integration/test_pipeline.py --cov=voice_referee/src --cov-report=html
```

### Debug Mode (Show Print Statements)

```bash
pytest voice_referee/tests/integration/test_pipeline.py -v -s
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `mock_deepgram_stt` - Mock STT service
- `mock_llm_service` - Mock Claude LLM
- `mock_tts_service` - Mock ElevenLabs TTS
- `mock_daily_transport` - Mock Daily.co transport
- `create_transcription_frame` - Factory for creating test frames
- `sample_utterances` - Pre-defined conversation scenarios
- `memory_store` - In-memory coordination storage
- `test_config` - Test configuration parameters

## Memory Coordination

Tests store results in memory namespace `voice-referee`:

```python
memory_store['get']('voice-referee', 'phase5-integration-tests')
```

Results include:
- Test pass/fail status
- Tension scores
- Intervention counts
- Speaker balance metrics
- Protocol violation detections

## Architecture Tested

```
TranscriptionFrame (Deepgram)
    ↓
RefereeMonitorProcessor
    ├→ SpeakerMapper (Speaker ID → Founder Name)
    ├→ ConversationState (Transcript Buffer + Stats)
    ├→ ConversationAnalyzer (Tension + Balance Analysis)
    └→ InterventionDecider (Decision Logic)
        ↓
LLMMessagesFrame (Claude)
    ↓
TextFrame (ElevenLabs TTS)
    ↓
Daily.co Transport (Audio Output)
```

## Expected Test Results

All tests should pass with:
- ✓ No false positive interventions
- ✓ All high-tension scenarios caught
- ✓ Protocol violations detected
- ✓ Cooldown properly enforced
- ✓ Speaker mapping accurate
- ✓ Transcript capture complete

## Troubleshooting

### Import Errors

Ensure you're in the correct directory:
```bash
cd /Users/nateaune/Documents/code/ump
export PYTHONPATH="${PYTHONPATH}:/Users/nateaune/Documents/code/ump"
```

### Async Warnings

Tests use `pytest-asyncio`. Install if needed:
```bash
pip install pytest-asyncio
```

### Mock Issues

If mocks aren't working, verify patch paths match actual import structure.

## Next Steps

After integration tests pass:

1. **Phase 5.2**: Add performance benchmarks
2. **Phase 5.3**: Add stress testing (rapid interventions)
3. **Phase 5.4**: Add real API integration tests (optional)
4. **Phase 6**: Deploy to staging environment

## Test Data Sources

Conversation scenarios are based on:
- ump.ai UI screenshots showing real founder conversations
- Protocol definitions from ump.ai documentation
- Tension detection requirements from product specs

---

**Generated**: Phase 5.1 - Integration Tests
**Architecture**: A (Deepgram Diarization)
**Status**: Complete ✓
