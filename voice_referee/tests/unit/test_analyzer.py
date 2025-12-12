"""
Unit tests for ConversationAnalyzer module

Tests the analyze() method which takes ConversationState and returns AnalysisResult.
"""

import pytest
from analysis.conversation_analyzer import ConversationAnalyzer, AnalysisResult
from processors.conversation_state import ConversationState, Utterance


class TestConversationAnalyzer:
    """Test suite for ConversationAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Fixture providing a fresh ConversationAnalyzer instance"""
        return ConversationAnalyzer(tension_threshold=0.7)

    @pytest.fixture
    def state(self):
        """Fixture providing a fresh ConversationState instance"""
        return ConversationState(max_buffer_size=50)

    @pytest.fixture
    def calm_state(self, state):
        """Fixture providing a calm conversation state"""
        # Add balanced, polite utterances
        state.add_utterance("I agree with your point", "Founder A", 100.0, 2.0, sentiment=0.3)
        state.add_utterance("Thank you for understanding", "Founder B", 102.0, 2.0, sentiment=0.4)
        state.add_utterance("Let's move forward together", "Founder A", 104.0, 2.0, sentiment=0.2)
        state.add_utterance("Sounds like a good plan", "Founder B", 106.0, 2.0, sentiment=0.3)
        return state

    @pytest.fixture
    def tense_state(self, state):
        """Fixture providing a tense conversation state with interruptions"""
        state.add_utterance("That's completely wrong", "Founder A", 100.0, 2.0, sentiment=-0.7)
        state.track_interruption("Founder B")
        state.add_utterance("You never listen to me", "Founder B", 102.0, 2.0, sentiment=-0.8)
        state.track_interruption("Founder A")
        state.add_utterance("This is impossible", "Founder A", 104.0, 2.0, sentiment=-0.6)
        state.track_interruption("Founder B")
        state.add_utterance("You always make bad decisions", "Founder B", 106.0, 2.0, sentiment=-0.9)
        return state

    def test_analyzer_initialization(self):
        """Test ConversationAnalyzer initialization with custom threshold"""
        analyzer = ConversationAnalyzer(tension_threshold=0.5)
        assert analyzer._tension_threshold == 0.5

    def test_analyzer_initialization_default_threshold(self):
        """Test ConversationAnalyzer uses default threshold of 0.7"""
        analyzer = ConversationAnalyzer()
        assert analyzer._tension_threshold == 0.7

    def test_analyzer_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ConversationAnalyzer(tension_threshold=1.5)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ConversationAnalyzer(tension_threshold=-0.1)

    def test_analyze_returns_analysis_result(self, analyzer, calm_state):
        """Test analyze returns AnalysisResult dataclass"""
        result = analyzer.analyze(calm_state)

        assert isinstance(result, AnalysisResult)
        assert hasattr(result, 'tension_score')
        assert hasattr(result, 'balance_score')
        assert hasattr(result, 'interruption_rate')
        assert hasattr(result, 'dominant_speaker')
        assert hasattr(result, 'detected_patterns')
        assert hasattr(result, 'requires_intervention')

    def test_analyze_calm_conversation(self, analyzer, calm_state):
        """Test analysis of calm conversation shows low tension"""
        result = analyzer.analyze(calm_state)

        # Calm conversation should have low tension
        assert result.tension_score < 0.5
        assert result.requires_intervention is False
        assert 0.0 <= result.tension_score <= 1.0

    def test_analyze_tense_conversation(self, analyzer, tense_state):
        """Test analysis of tense conversation shows high tension"""
        result = analyzer.analyze(tense_state)

        # Tense conversation should have elevated tension due to interruptions
        # and negative sentiment
        assert result.tension_score > 0.2
        assert 0.0 <= result.tension_score <= 1.0

    def test_analyze_balanced_speakers(self, analyzer, calm_state):
        """Test balance score for balanced speakers"""
        result = analyzer.analyze(calm_state)

        # Both speakers have equal utterances
        assert result.balance_score < 0.3  # Low imbalance

    def test_analyze_imbalanced_speakers(self, analyzer, state):
        """Test balance score for imbalanced speakers"""
        # One speaker dominates
        state.add_utterance("Long speech one", "Founder A", 100.0, 5.0)
        state.add_utterance("Long speech two", "Founder A", 105.0, 5.0)
        state.add_utterance("Long speech three", "Founder A", 110.0, 5.0)
        state.add_utterance("Brief reply", "Founder B", 115.0, 1.0)

        result = analyzer.analyze(state)

        # Should show imbalance
        assert result.balance_score > 0.5

    def test_analyze_detects_dominant_speaker(self, analyzer, state):
        """Test detection of dominant speaker"""
        # Founder A speaks much more
        state.add_utterance("Long speech one", "Founder A", 100.0, 10.0)
        state.add_utterance("Long speech two", "Founder A", 110.0, 10.0)
        state.add_utterance("Long speech three", "Founder A", 120.0, 10.0)
        state.add_utterance("Short reply", "Founder B", 130.0, 2.0)

        result = analyzer.analyze(state)

        # Founder A should be detected as dominant
        assert result.dominant_speaker == "Founder A"

    def test_analyze_balanced_conversation_low_balance_score(self, analyzer, state):
        """Test balance score is low when conversation is truly balanced"""
        # Create truly balanced conversation with equal speaking time
        state.add_utterance("First point here", "Founder A", 100.0, 5.0, sentiment=0.3)
        state.add_utterance("Second point here", "Founder B", 105.0, 5.0, sentiment=0.4)
        state.add_utterance("Third point here", "Founder A", 110.0, 5.0, sentiment=0.2)
        state.add_utterance("Fourth point here", "Founder B", 115.0, 5.0, sentiment=0.3)

        result = analyzer.analyze(state)

        # Balance score should be low when both have equal speaking time
        assert result.balance_score < 0.2  # Near zero for balanced

    def test_analyze_interruption_rate(self, analyzer, tense_state):
        """Test interruption rate calculation"""
        result = analyzer.analyze(tense_state)

        # Should have positive interruption rate
        assert result.interruption_rate > 0

    def test_analyze_no_interruptions(self, analyzer, calm_state):
        """Test analysis with no interruptions"""
        result = analyzer.analyze(calm_state)

        # No interruptions tracked
        assert result.interruption_rate == 0.0

    def test_analyze_detected_patterns(self, analyzer, state):
        """Test pattern detection for imbalanced conversation"""
        # Create very imbalanced conversation
        state.add_utterance("Long speech one", "Founder A", 100.0, 20.0)
        state.add_utterance("Long speech two", "Founder A", 120.0, 20.0)
        state.add_utterance("ok", "Founder B", 140.0, 0.5)

        result = analyzer.analyze(state)

        # Should detect patterns like dominance or imbalance
        assert len(result.detected_patterns) > 0

    def test_analyze_empty_state(self, analyzer, state):
        """Test analysis of empty conversation state"""
        result = analyzer.analyze(state)

        # Empty state should have zero/low values
        assert result.tension_score == 0.0 or result.tension_score < 0.3
        assert result.balance_score == 0.0
        assert result.interruption_rate == 0.0

    def test_set_tension_threshold(self, analyzer):
        """Test updating tension threshold"""
        analyzer.set_tension_threshold(0.5)
        assert analyzer._tension_threshold == 0.5

    def test_set_tension_threshold_invalid_raises_error(self, analyzer):
        """Test invalid threshold update raises ValueError"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            analyzer.set_tension_threshold(1.5)

    def test_tension_threshold_affects_intervention(self, analyzer, tense_state):
        """Test that tension threshold affects requires_intervention"""
        # With high threshold, may not require intervention
        analyzer.set_tension_threshold(0.9)
        result_high = analyzer.analyze(tense_state)

        # With low threshold, more likely to require intervention
        analyzer.set_tension_threshold(0.1)
        result_low = analyzer.analyze(tense_state)

        # Lower threshold should more easily trigger intervention
        if result_high.tension_score < 0.9:
            assert result_high.requires_intervention is False
        if result_low.tension_score >= 0.1:
            assert result_low.requires_intervention is True

    def test_tension_score_bounds(self, analyzer, tense_state):
        """Test that tension score is always bounded 0.0-1.0"""
        result = analyzer.analyze(tense_state)
        assert 0.0 <= result.tension_score <= 1.0

    def test_balance_score_bounds(self, analyzer, state):
        """Test that balance score is always bounded 0.0-1.0"""
        state.add_utterance("test", "A", 100.0, 1.0)
        result = analyzer.analyze(state)
        assert 0.0 <= result.balance_score <= 1.0

    def test_repr(self, analyzer):
        """Test string representation"""
        repr_str = repr(analyzer)
        assert "ConversationAnalyzer" in repr_str
        assert "tension_threshold" in repr_str
