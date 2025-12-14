"""
Unit tests for vision components.

Tests the vision service, screen analyzer, and commentary processor
for the screen sharing + AI vision + voice commentary feature.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

# Import the components to test
from src.services.vision_service import (
    VisionService,
    VisionAnalysisResult,
    create_vision_service,
)
from src.processors.screen_analyzer import (
    ScreenAnalysisProcessor,
    ScreenAnalysisFrame,
    create_screen_analyzer,
)
from src.processors.commentary_processor import (
    CommentaryProcessor,
    create_commentary_processor,
    COMMENTARY_PROMPTS,
)
from src.config.settings import VisionConfig


# ============================================================================
# VisionConfig Tests
# ============================================================================

class TestVisionConfig:
    """Tests for VisionConfig validation and defaults."""

    def test_default_values(self):
        """Test that VisionConfig has correct defaults."""
        config = VisionConfig(api_key="test-key")

        assert config.enabled == False
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.analysis_interval_seconds == 2.0
        assert config.max_analysis_cost_per_session == 0.30
        assert config.commentary_style == "concise"
        assert config.frame_capture_mode == "on_demand"
        assert config.commentary_trigger_threshold == 0.6
        assert config.max_vision_latency_ms == 500
        assert config.adaptive_frame_rate == True

    def test_valid_providers(self):
        """Test that valid providers are accepted."""
        for provider in ["anthropic", "openai", "google"]:
            config = VisionConfig(api_key="test", provider=provider)
            assert config.provider == provider

    def test_invalid_provider_raises(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider must be one of"):
            VisionConfig(api_key="test", provider="invalid")

    def test_valid_capture_modes(self):
        """Test that valid capture modes are accepted."""
        for mode in ["on_demand", "continuous"]:
            config = VisionConfig(api_key="test", frame_capture_mode=mode)
            assert config.frame_capture_mode == mode

    def test_invalid_capture_mode_raises(self):
        """Test that invalid capture mode raises ValueError."""
        with pytest.raises(ValueError, match="Frame capture mode must be one of"):
            VisionConfig(api_key="test", frame_capture_mode="invalid")

    def test_valid_commentary_styles(self):
        """Test that valid commentary styles are accepted."""
        for style in ["concise", "detailed", "technical"]:
            config = VisionConfig(api_key="test", commentary_style=style)
            assert config.commentary_style == style

    def test_invalid_commentary_style_raises(self):
        """Test that invalid commentary style raises ValueError."""
        with pytest.raises(ValueError, match="Commentary style must be one of"):
            VisionConfig(api_key="test", commentary_style="invalid")


# ============================================================================
# VisionAnalysisResult Tests
# ============================================================================

class TestVisionAnalysisResult:
    """Tests for VisionAnalysisResult dataclass."""

    def test_creation_with_defaults(self):
        """Test creating result with default values."""
        result = VisionAnalysisResult(
            description="Test description",
            confidence=0.9,
            latency_ms=150.0,
            provider="anthropic",
            model="claude-3-5-sonnet"
        )

        assert result.description == "Test description"
        assert result.confidence == 0.9
        assert result.latency_ms == 150.0
        assert result.tokens_used == 0
        assert result.cost_usd == 0.0
        assert result.requires_commentary == False
        assert result.detected_elements == []

    def test_detected_elements_initialized(self):
        """Test that detected_elements is initialized properly."""
        result = VisionAnalysisResult(
            description="Test",
            confidence=0.9,
            latency_ms=100,
            provider="test",
            model="test"
        )
        assert isinstance(result.detected_elements, list)
        assert len(result.detected_elements) == 0


# ============================================================================
# VisionService Tests
# ============================================================================

class TestVisionService:
    """Tests for VisionService."""

    def test_initialization_anthropic(self):
        """Test Anthropic client initialization."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            service = VisionService(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )

            assert service.provider == "anthropic"
            assert service.model == "claude-3-5-sonnet-20241022"
            assert service.api_key == "test-key"
            assert service.max_tokens == 150
            mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_invalid_provider_raises(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vision provider"):
            VisionService(provider="invalid", model="test", api_key="test")

    def test_should_comment_with_notable_keywords(self):
        """Test _should_comment detects notable keywords."""
        service = VisionService.__new__(VisionService)
        service.provider = "anthropic"

        # These should trigger commentary
        assert service._should_comment("There is an error on the screen")
        assert service._should_comment("Warning: disk space low")
        assert service._should_comment("This is a code editor")
        assert service._should_comment("The chart shows significant data")
        assert service._should_comment("Important notification appeared")
        assert service._should_comment("New window opened")

    def test_should_comment_without_notable_keywords(self):
        """Test _should_comment returns False for mundane content."""
        service = VisionService.__new__(VisionService)
        service.provider = "anthropic"

        assert not service._should_comment("Plain desktop background")
        assert not service._should_comment("Static webpage")
        assert not service._should_comment("Empty workspace")

    def test_cost_calculation_anthropic(self):
        """Test cost calculation for Anthropic."""
        service = VisionService.__new__(VisionService)
        service.provider = "anthropic"

        cost = service._calculate_cost(1000, 500)
        # anthropic: input=$3/1M, output=$15/1M
        expected = (1000 / 1_000_000 * 3.0) + (500 / 1_000_000 * 15.0)
        assert cost == pytest.approx(expected)

    def test_cost_calculation_openai(self):
        """Test cost calculation for OpenAI."""
        service = VisionService.__new__(VisionService)
        service.provider = "openai"

        cost = service._calculate_cost(1000, 500)
        # openai: input=$10/1M, output=$30/1M
        expected = (1000 / 1_000_000 * 10.0) + (500 / 1_000_000 * 30.0)
        assert cost == pytest.approx(expected)

    def test_cost_calculation_google(self):
        """Test cost calculation for Google."""
        service = VisionService.__new__(VisionService)
        service.provider = "google"

        cost = service._calculate_cost(1000, 500)
        # google: input=$0.5/1M, output=$1.5/1M
        expected = (1000 / 1_000_000 * 0.5) + (500 / 1_000_000 * 1.5)
        assert cost == pytest.approx(expected)

    def test_create_vision_service_factory(self):
        """Test factory function creates service correctly."""
        with patch('anthropic.Anthropic'):
            service = create_vision_service(
                provider="anthropic",
                model="claude-3-5-sonnet",
                api_key="test-key"
            )

            assert isinstance(service, VisionService)
            assert service.provider == "anthropic"
            assert service.model == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_analyze_image_error_handling(self):
        """Test that analyze_image handles errors gracefully."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            service = VisionService(
                provider="anthropic",
                model="claude-3-5-sonnet",
                api_key="test-key"
            )

            # Mock the client to raise an exception
            mock_client.messages.create.side_effect = Exception("API Error")

            result = await service.analyze_image(
                image_data=b"fake_image_data",
                prompt="Test prompt"
            )

            assert "Analysis failed" in result.description
            assert result.confidence == 0.0
            assert result.latency_ms > 0


# ============================================================================
# ScreenAnalysisProcessor Tests
# ============================================================================

class TestScreenAnalysisProcessor:
    """Tests for ScreenAnalysisProcessor."""

    @pytest.fixture
    def mock_vision_service(self):
        """Create a mock VisionService."""
        service = Mock(spec=VisionService)
        service.analyze_image = AsyncMock(return_value=VisionAnalysisResult(
            description="Test screen content",
            confidence=0.9,
            latency_ms=150,
            provider="anthropic",
            model="test",
            cost_usd=0.001,
            requires_commentary=True
        ))
        return service

    @pytest.fixture
    def processor(self, mock_vision_service):
        """Create a ScreenAnalysisProcessor with mock service."""
        return ScreenAnalysisProcessor(
            vision_service=mock_vision_service,
            analysis_interval=2.0,
            cost_limit=0.30
        )

    def test_initialization(self, processor, mock_vision_service):
        """Test processor initialization."""
        assert processor.vision_service == mock_vision_service
        assert processor.analysis_interval == 2.0
        assert processor.cost_limit == 0.30
        assert processor._frame_count == 0
        assert processor._total_cost == 0.0
        assert processor._analysis_count == 0
        assert processor._cost_exceeded == False

    def test_default_prompt(self, processor):
        """Test that default prompt is set for mediation context."""
        prompt = processor._default_prompt()
        assert "mediation" in prompt.lower()
        assert "brief" in prompt.lower()

    def test_should_analyze_first_call(self, processor):
        """Test that first call should analyze."""
        assert processor._should_analyze() == True

    def test_should_analyze_respects_interval(self, processor):
        """Test that analysis respects interval timing."""
        # First call should analyze
        assert processor._should_analyze() == True

        # Simulate recent analysis
        processor._last_analysis_time = time.time()
        assert processor._should_analyze() == False

        # After interval, should analyze again
        processor._last_analysis_time = time.time() - 3.0
        assert processor._should_analyze() == True

    def test_should_analyze_respects_cost_limit(self, processor):
        """Test that analysis stops when cost limit exceeded."""
        processor._total_cost = 0.35  # Exceeds $0.30 limit
        assert processor._should_analyze() == False
        assert processor._cost_exceeded == True

    def test_get_stats(self, processor):
        """Test statistics retrieval."""
        processor._frame_count = 10
        processor._analysis_count = 5
        processor._total_cost = 0.05
        processor._last_analysis_time = time.time()

        stats = processor.get_stats()

        assert stats["frames_received"] == 10
        assert stats["analyses_performed"] == 5
        assert stats["total_cost_usd"] == 0.05
        assert stats["cost_limit_usd"] == 0.30
        assert stats["cost_exceeded"] == False
        assert "average_interval_actual" in stats

    def test_reset(self, processor):
        """Test processor reset."""
        processor._frame_count = 10
        processor._total_cost = 0.25
        processor._cost_exceeded = True
        processor._analysis_count = 5

        processor.reset()

        assert processor._frame_count == 0
        assert processor._total_cost == 0.0
        assert processor._cost_exceeded == False
        assert processor._analysis_count == 0
        assert processor._last_analysis_time == 0

    def test_get_frame_size_with_size_attr(self, processor):
        """Test getting frame size when size attribute exists."""
        mock_frame = Mock()
        mock_frame.size = (1920, 1080)

        size = processor._get_frame_size(mock_frame)
        assert size == (1920, 1080)

    def test_get_frame_size_with_width_height(self, processor):
        """Test getting frame size when width/height attributes exist."""
        mock_frame = Mock()
        del mock_frame.size  # Remove size attribute
        mock_frame.width = 1280
        mock_frame.height = 720

        size = processor._get_frame_size(mock_frame)
        assert size == (1280, 720)

    def test_get_frame_size_fallback(self, processor):
        """Test getting frame size with no attributes."""
        mock_frame = Mock(spec=[])  # No attributes

        size = processor._get_frame_size(mock_frame)
        assert size == (0, 0)

    def test_repr(self, processor):
        """Test string representation."""
        repr_str = repr(processor)
        assert "ScreenAnalysisProcessor" in repr_str
        assert "2.0s" in repr_str
        assert "$0.30" in repr_str or "0.3" in repr_str

    def test_factory_function(self, mock_vision_service):
        """Test create_screen_analyzer factory."""
        processor = create_screen_analyzer(
            vision_service=mock_vision_service,
            analysis_interval=3.0,
            cost_limit=0.50
        )

        assert isinstance(processor, ScreenAnalysisProcessor)
        assert processor.analysis_interval == 3.0
        assert processor.cost_limit == 0.50


# ============================================================================
# CommentaryProcessor Tests
# ============================================================================

class TestCommentaryProcessor:
    """Tests for CommentaryProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a CommentaryProcessor."""
        return CommentaryProcessor(
            commentary_style="concise",
            trigger_threshold=0.6,
            cooldown_seconds=5.0
        )

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.commentary_style == "concise"
        assert processor.trigger_threshold == 0.6
        assert processor.cooldown_seconds == 5.0
        assert processor._commentary_count == 0
        assert processor._skipped_count == 0
        assert processor._last_commentary_time == 0

    def test_system_prompts_exist_for_all_styles(self):
        """Test that system prompts exist for all styles."""
        assert "concise" in COMMENTARY_PROMPTS
        assert "detailed" in COMMENTARY_PROMPTS
        assert "technical" in COMMENTARY_PROMPTS

        # Verify each prompt has content and is for mediation context
        for style, prompt in COMMENTARY_PROMPTS.items():
            assert len(prompt) > 0
            assert "mediator" in prompt.lower() or "mediation" in prompt.lower()

    def test_system_prompt_selection(self):
        """Test that correct system prompt is selected."""
        processor = CommentaryProcessor(commentary_style="technical")
        assert processor.system_prompt == COMMENTARY_PROMPTS["technical"]

        processor = CommentaryProcessor(commentary_style="detailed")
        assert processor.system_prompt == COMMENTARY_PROMPTS["detailed"]

    def test_invalid_style_falls_back_to_concise(self):
        """Test that invalid style falls back to concise."""
        processor = CommentaryProcessor(commentary_style="invalid_style")
        assert processor.system_prompt == COMMENTARY_PROMPTS["concise"]

    def test_should_generate_commentary_with_flag(self, processor):
        """Test commentary generation when requires_commentary is True."""
        analysis = Mock()
        analysis.requires_commentary = True
        analysis.confidence = 0.5  # Below threshold, but flag overrides
        analysis.description = "Something"

        assert processor._should_generate_commentary(analysis) == True

    def test_should_generate_commentary_above_threshold(self, processor):
        """Test commentary generated above threshold."""
        analysis = Mock()
        analysis.requires_commentary = False
        analysis.confidence = 0.9  # Above 0.6 threshold
        analysis.description = "Interesting content visible"

        assert processor._should_generate_commentary(analysis) == True

    def test_should_generate_commentary_below_threshold(self, processor):
        """Test commentary not generated below threshold."""
        analysis = Mock()
        analysis.requires_commentary = False
        analysis.confidence = 0.3  # Below 0.6 threshold
        analysis.description = "Something notable"

        assert processor._should_generate_commentary(analysis) == False

    def test_should_generate_commentary_nothing_notable(self, processor):
        """Test commentary not generated for nothing notable."""
        analysis = Mock()
        analysis.requires_commentary = False
        analysis.confidence = 0.9  # Above threshold
        analysis.description = "Nothing notable on the screen"

        assert processor._should_generate_commentary(analysis) == False

    def test_should_generate_commentary_analysis_failed(self, processor):
        """Test commentary not generated for failed analysis."""
        analysis = Mock()
        analysis.requires_commentary = False
        analysis.confidence = 0.9
        analysis.description = "Analysis failed: timeout"

        assert processor._should_generate_commentary(analysis) == False

    def test_build_user_prompt(self, processor):
        """Test user prompt building."""
        analysis = Mock()
        analysis.description = "A code editor showing Python"
        analysis.confidence = 0.9
        analysis.detected_elements = ["code", "IDE"]

        prompt = processor._build_user_prompt(analysis)

        assert "A code editor showing Python" in prompt
        assert "code" in prompt
        assert "IDE" in prompt
        assert "spoken commentary" in prompt
        assert "[SKIP]" in prompt

    def test_build_user_prompt_no_elements(self, processor):
        """Test user prompt building without detected elements."""
        analysis = Mock()
        analysis.description = "Desktop background"
        analysis.confidence = 0.7
        analysis.detected_elements = []

        prompt = processor._build_user_prompt(analysis)

        assert "Desktop background" in prompt
        assert "Detected elements:" not in prompt

    def test_get_stats(self, processor):
        """Test statistics retrieval."""
        processor._commentary_count = 5
        processor._skipped_count = 10

        stats = processor.get_stats()

        assert stats["commentaries_generated"] == 5
        assert stats["frames_skipped"] == 10
        assert stats["total_frames_processed"] == 15
        assert stats["commentary_style"] == "concise"
        assert stats["trigger_threshold"] == 0.6

    def test_reset(self, processor):
        """Test processor reset."""
        processor._commentary_count = 5
        processor._last_commentary_time = time.time()
        processor._skipped_count = 10

        processor.reset()

        assert processor._commentary_count == 0
        assert processor._last_commentary_time == 0
        assert processor._skipped_count == 0

    def test_repr(self, processor):
        """Test string representation."""
        repr_str = repr(processor)
        assert "CommentaryProcessor" in repr_str
        assert "concise" in repr_str
        assert "0.6" in repr_str

    def test_factory_function(self):
        """Test create_commentary_processor factory."""
        processor = create_commentary_processor(
            commentary_style="technical",
            trigger_threshold=0.8,
            cooldown_seconds=10.0
        )

        assert isinstance(processor, CommentaryProcessor)
        assert processor.commentary_style == "technical"
        assert processor.trigger_threshold == 0.8
        assert processor.cooldown_seconds == 10.0


# ============================================================================
# ScreenAnalysisFrame Tests
# ============================================================================

class TestScreenAnalysisFrame:
    """Tests for ScreenAnalysisFrame dataclass."""

    def test_creation(self):
        """Test creating ScreenAnalysisFrame."""
        analysis = VisionAnalysisResult(
            description="Test",
            confidence=0.9,
            latency_ms=100,
            provider="anthropic",
            model="test"
        )

        frame = ScreenAnalysisFrame(
            analysis=analysis,
            timestamp=time.time(),
            frame_number=42,
            image_size=(1920, 1080)
        )

        assert frame.analysis == analysis
        assert frame.frame_number == 42
        assert frame.image_size == (1920, 1080)
        assert isinstance(frame.timestamp, float)


# ============================================================================
# Integration Tests (Mock-based)
# ============================================================================

class TestVisionPipelineIntegration:
    """Integration tests for vision pipeline components."""

    @pytest.fixture
    def mock_vision_service(self):
        """Create mock vision service."""
        service = Mock()
        service.analyze_image = AsyncMock(return_value=VisionAnalysisResult(
            description="Code editor with Python",
            confidence=0.95,
            latency_ms=200,
            provider="anthropic",
            model="claude-3-5-sonnet",
            cost_usd=0.002,
            requires_commentary=True,
            detected_elements=["code", "python"]
        ))
        return service

    @pytest.mark.asyncio
    async def test_analyzer_calls_vision_service(self, mock_vision_service):
        """Test that analyzer calls vision service correctly."""
        processor = ScreenAnalysisProcessor(
            vision_service=mock_vision_service,
            analysis_interval=0,  # No interval for test
            cost_limit=1.0
        )

        # Create a mock image frame
        mock_frame = Mock()
        mock_frame.image = b"fake_image_data"
        mock_frame.size = (1920, 1080)

        # Analyze the frame
        result = await processor._analyze_frame(mock_frame)

        # Verify vision service was called
        mock_vision_service.analyze_image.assert_called_once()
        call_args = mock_vision_service.analyze_image.call_args
        assert call_args.kwargs["image_data"] == b"fake_image_data"

        # Verify result
        assert result.description == "Code editor with Python"
        assert result.confidence == 0.95

    def test_commentary_processor_filters_low_confidence(self):
        """Test that commentary processor filters low confidence."""
        processor = CommentaryProcessor(
            commentary_style="concise",
            trigger_threshold=0.8,
            cooldown_seconds=0
        )

        # Low confidence analysis
        low_confidence = Mock()
        low_confidence.requires_commentary = False
        low_confidence.confidence = 0.5
        low_confidence.description = "Some content"

        assert processor._should_generate_commentary(low_confidence) == False

        # High confidence analysis
        high_confidence = Mock()
        high_confidence.requires_commentary = False
        high_confidence.confidence = 0.95
        high_confidence.description = "Important content"

        assert processor._should_generate_commentary(high_confidence) == True

    def test_end_to_end_cost_tracking(self, mock_vision_service):
        """Test cost tracking across multiple analyses."""
        processor = ScreenAnalysisProcessor(
            vision_service=mock_vision_service,
            analysis_interval=0,  # No interval
            cost_limit=0.005  # Low limit for testing
        )

        # Update cost manually (simulating analysis)
        processor._total_cost = 0.003
        assert processor._should_analyze() == True

        processor._total_cost = 0.006  # Exceeds limit
        assert processor._should_analyze() == False
        assert processor._cost_exceeded == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
