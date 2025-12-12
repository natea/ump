"""
Unit tests for RefereeMonitorProcessor

Tests transcription processing, speaker mapping integration,
intervention triggering, and frame handling.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import time

from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMMessagesFrame,
    FrameDirection
)

from processors.referee_monitor import RefereeMonitorProcessor
from config.settings import ProcessorConfig


class TestRefereeMonitorProcessor:
    """Test suite for RefereeMonitorProcessor"""

    @pytest.fixture
    def config(self):
        """Fixture providing processor configuration"""
        return ProcessorConfig(
            tension_threshold=0.7,
            cooldown_seconds=60,
            buffer_size=50
        )

    @pytest.fixture
    def processor(self, config):
        """Fixture providing RefereeMonitorProcessor instance"""
        return RefereeMonitorProcessor(
            config=config,
            founder_names=["Alice", "Bob"]
        )

    @pytest.mark.asyncio
    async def test_initialization(self, processor, config):
        """Test processor initialization"""
        assert processor._config == config
        assert processor._speaker_mapper is not None
        assert processor._conversation_state is not None
        assert processor._analyzer is not None
        assert processor._decider is not None
        assert processor._current_speaker is None
        assert processor._speaking_start_time is None

    @pytest.mark.asyncio
    async def test_transcription_frame_processing(self, processor):
        """Test processing TranscriptionFrame with speaker diarization"""
        # Create TranscriptionFrame with speaker attribute
        frame = TranscriptionFrame(text="Hello world", user_id="user1", timestamp=100)
        frame.speaker = 0  # Deepgram speaker ID

        # Mock push_frame to verify it's called
        processor.push_frame = AsyncMock()

        # Process the frame
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify speaker was mapped
        assert processor._speaker_mapper.is_assigned(0)
        assert processor._speaker_mapper.get_identity(0) == "Alice"

        # Verify utterance was added to state
        stats = processor.get_stats()
        assert stats['total_utterances'] == 1
        assert 'Alice' in stats['speaker_stats']

        # Verify frame was pushed downstream
        processor.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_transcription_frame_missing_speaker_attribute(self, processor):
        """Test handling TranscriptionFrame without speaker attribute"""
        frame = TranscriptionFrame(text="Hello world", user_id="user1", timestamp=100)
        # No speaker attribute set

        processor.push_frame = AsyncMock()

        # Should handle gracefully (log warning and skip)
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # No utterance should be added
        stats = processor.get_stats()
        assert stats['total_utterances'] == 0

    @pytest.mark.asyncio
    async def test_empty_transcription_skipped(self, processor):
        """Test that empty transcription text is skipped"""
        frame = TranscriptionFrame(text="   ", user_id="user1", timestamp=100)
        frame.speaker = 0

        processor.push_frame = AsyncMock()

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # No utterance should be added
        stats = processor.get_stats()
        assert stats['total_utterances'] == 0

    @pytest.mark.asyncio
    async def test_speaker_mapping_integration(self, processor):
        """Test that speaker IDs are consistently mapped to founder names"""
        # First utterance from speaker 0
        frame1 = TranscriptionFrame(text="First message", user_id="user1", timestamp=100)
        frame1.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame1, FrameDirection.DOWNSTREAM)

        # Second utterance from speaker 1
        frame2 = TranscriptionFrame(text="Second message", user_id="user1", timestamp=101)
        frame2.speaker = 1
        await processor.process_frame(frame2, FrameDirection.DOWNSTREAM)

        # Third utterance from speaker 0 again
        frame3 = TranscriptionFrame(text="Third message", user_id="user1", timestamp=102)
        frame3.speaker = 0
        await processor.process_frame(frame3, FrameDirection.DOWNSTREAM)

        # Verify consistent mapping
        stats = processor.get_stats()
        assert 'Alice' in stats['speaker_stats']
        assert 'Bob' in stats['speaker_stats']
        assert stats['speaker_stats']['Alice']['utterance_count'] == 2
        assert stats['speaker_stats']['Bob']['utterance_count'] == 1

    @pytest.mark.asyncio
    async def test_intervention_trigger(self, processor):
        """Test that intervention is triggered on high tension"""
        # Mock the analyzer and decider to trigger intervention
        with patch.object(processor._analyzer, 'analyze') as mock_analyze, \
             patch.object(processor._decider, 'decide') as mock_decide:

            # Mock analysis result
            mock_analysis = Mock()
            mock_analysis.tension_score = 0.9
            mock_analysis.detected_patterns = ["high_tension"]
            mock_analyze.return_value = mock_analysis

            # Mock decision to intervene
            mock_decision = Mock()
            mock_decision.should_intervene = True
            mock_decision.reason = "High tension detected"
            mock_decision.confidence = 0.95
            mock_decision.suggested_prompt = "Please calm down and listen to each other"
            mock_decide.return_value = mock_decision

            # Mock push_frame to capture LLMMessagesFrame
            pushed_frames = []
            async def capture_frame(frame, direction=None):
                pushed_frames.append(frame)
            processor.push_frame = AsyncMock(side_effect=capture_frame)

            # Process high-tension transcription
            frame = TranscriptionFrame(text="You never listen to me!", user_id="user1", timestamp=100)
            frame.speaker = 0
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

            # Verify LLMMessagesFrame was pushed
            llm_frames = [f for f in pushed_frames if isinstance(f, LLMMessagesFrame)]
            assert len(llm_frames) > 0

            # Verify intervention was recorded
            stats = processor.get_stats()
            assert stats['last_intervention_time'] is not None

    @pytest.mark.asyncio
    async def test_user_started_speaking_tracking(self, processor):
        """Test tracking when user starts speaking"""
        frame = UserStartedSpeakingFrame()
        processor.push_frame = AsyncMock()

        start_time = time.time()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Should track speaking start time
        assert processor._speaking_start_time is not None
        assert processor._speaking_start_time >= start_time

    @pytest.mark.asyncio
    async def test_user_stopped_speaking_duration(self, processor):
        """Test calculating duration when user stops speaking"""
        # Start speaking
        start_frame = UserStartedSpeakingFrame()
        processor.push_frame = AsyncMock()
        await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)

        # Wait a bit
        time.sleep(0.1)

        # Stop speaking
        stop_frame = UserStoppedSpeakingFrame()
        await processor.process_frame(stop_frame, FrameDirection.DOWNSTREAM)

        # Speaking start time should be cleared
        assert processor._speaking_start_time is None

    @pytest.mark.asyncio
    async def test_duration_calculation_with_tracked_time(self, processor):
        """Test that tracked speaking time is used for duration"""
        # Start speaking
        processor._speaking_start_time = time.time()
        time.sleep(0.1)

        # Process transcription (should use tracked time)
        frame = TranscriptionFrame(text="Test message", user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check that duration was calculated from tracked time
        stats = processor.get_stats()
        speaker_stats = stats['speaker_stats']['Alice']
        assert speaker_stats['total_time'] >= 0.1  # At least 100ms

    @pytest.mark.asyncio
    async def test_duration_estimation_from_word_count(self, processor):
        """Test duration estimation based on word count when no tracked time"""
        # No speaking start time set
        processor._speaking_start_time = None

        # Process transcription with known word count
        # 150 words should take ~60 seconds at 150 words/minute
        text = " ".join(["word"] * 150)
        frame = TranscriptionFrame(text=text, user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Check estimated duration
        stats = processor.get_stats()
        speaker_stats = stats['speaker_stats']['Alice']
        # Should be ~60 seconds (150 words / 150 words per minute * 60)
        assert 50.0 <= speaker_stats['total_time'] <= 70.0

    @pytest.mark.asyncio
    async def test_minimum_duration_applied(self, processor):
        """Test that minimum duration of 0.5s is applied"""
        processor._speaking_start_time = None

        # Single word (would estimate very low duration)
        frame = TranscriptionFrame(text="Hi", user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        stats = processor.get_stats()
        speaker_stats = stats['speaker_stats']['Alice']
        # Should apply minimum 0.5 seconds
        assert speaker_stats['total_time'] >= 0.5

    @pytest.mark.asyncio
    async def test_get_state(self, processor):
        """Test get_state returns conversation state"""
        state = processor.get_state()

        assert state == processor._conversation_state

    @pytest.mark.asyncio
    async def test_get_stats(self, processor):
        """Test get_stats returns comprehensive statistics"""
        # Add some data
        frame = TranscriptionFrame(text="Test", user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        stats = processor.get_stats()

        assert 'session_duration' in stats
        assert 'total_utterances' in stats
        assert 'speaker_stats' in stats
        assert stats['total_utterances'] == 1

    @pytest.mark.asyncio
    async def test_reset(self, processor):
        """Test reset clears all processor state"""
        # Add data
        frame = TranscriptionFrame(text="Test", user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        processor._current_speaker = "Alice"
        processor._speaking_start_time = time.time()

        # Reset
        processor.reset()

        # Verify everything cleared
        stats = processor.get_stats()
        assert stats['total_utterances'] == 0
        assert processor._current_speaker is None
        assert processor._speaking_start_time is None
        assert not processor._speaker_mapper.is_assigned(0)

    @pytest.mark.asyncio
    async def test_repr(self, processor, config):
        """Test __repr__ provides meaningful representation"""
        repr_str = repr(processor)

        assert "RefereeMonitorProcessor" in repr_str
        assert str(config.tension_threshold) in repr_str
        assert str(config.cooldown_seconds) in repr_str

    @pytest.mark.asyncio
    async def test_error_handling_in_transcription(self, processor):
        """Test that errors in transcription handling are caught and logged"""
        # Create frame that will cause an error
        frame = TranscriptionFrame(text="Test", user_id="user1", timestamp=100)
        frame.speaker = 0

        # Mock speaker mapper to raise exception
        with patch.object(processor._speaker_mapper, 'assign_identity', side_effect=Exception("Test error")):
            processor.push_frame = AsyncMock()

            # Should not raise exception (error is caught)
            try:
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
            except Exception:
                pytest.fail("Exception should be caught and logged")

    @pytest.mark.asyncio
    async def test_intervention_creates_llm_messages_frame(self, processor):
        """Test that intervention creates LLMMessagesFrame with correct structure"""
        with patch.object(processor._analyzer, 'analyze') as mock_analyze, \
             patch.object(processor._decider, 'decide') as mock_decide:

            mock_analysis = Mock()
            mock_analysis.tension_score = 0.9
            mock_analysis.detected_patterns = []
            mock_analyze.return_value = mock_analysis

            mock_decision = Mock()
            mock_decision.should_intervene = True
            mock_decision.reason = "Test intervention"
            mock_decision.confidence = 0.95
            mock_decision.suggested_prompt = "Test prompt"
            mock_decide.return_value = mock_decision

            pushed_frames = []
            async def capture_frame(frame, direction=None):
                pushed_frames.append(frame)
            processor.push_frame = AsyncMock(side_effect=capture_frame)

            frame = TranscriptionFrame(text="Test", user_id="user1", timestamp=100)
            frame.speaker = 0
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

            # Find LLMMessagesFrame
            llm_frames = [f for f in pushed_frames if isinstance(f, LLMMessagesFrame)]
            assert len(llm_frames) > 0

            llm_frame = llm_frames[0]
            assert len(llm_frame.messages) == 2  # System and user messages
            assert llm_frame.messages[0]['role'] == 'system'
            assert llm_frame.messages[1]['role'] == 'user'
            assert "Test prompt" in llm_frame.messages[1]['content']

    @pytest.mark.asyncio
    async def test_default_config_when_none_provided(self):
        """Test that default config is used when none provided"""
        processor = RefereeMonitorProcessor()

        assert processor._config is not None
        assert isinstance(processor._config, ProcessorConfig)

    @pytest.mark.asyncio
    async def test_default_founder_names_when_none_provided(self):
        """Test that default founder names are used when none provided"""
        processor = RefereeMonitorProcessor()

        frame = TranscriptionFrame(text="Test", user_id="user1", timestamp=100)
        frame.speaker = 0
        processor.push_frame = AsyncMock()
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Should use default "Founder A"
        assert processor._speaker_mapper.get_identity(0) == "Founder A"
