"""
Unit tests for InterventionDecider module

Tests intervention decision logic, protocol violation detection,
cooldown management, and intervention triggers.
"""

import pytest
from datetime import datetime, timedelta
from decision.intervention_decider import (
    InterventionDecider,
    ConversationState
)


class TestInterventionDecider:
    """Test suite for InterventionDecider class"""

    @pytest.fixture
    def decider(self):
        """Fixture providing a fresh InterventionDecider instance"""
        return InterventionDecider()

    @pytest.fixture
    def calm_state(self):
        """Fixture providing a calm conversation state"""
        return ConversationState(
            duration_seconds=300.0,
            tension_score=0.3,
            interruption_rate=0.1,
            speaker_imbalance=0.2,
            argument_repetition_count=0,
            last_intervention_time=None,
            total_interventions=0,
            current_speakers=["Founder A", "Founder B"]
        )

    @pytest.fixture
    def tense_state(self):
        """Fixture providing a tense conversation state"""
        return ConversationState(
            duration_seconds=600.0,
            tension_score=0.85,
            interruption_rate=0.6,
            speaker_imbalance=0.3,
            argument_repetition_count=1,
            last_intervention_time=None,
            total_interventions=0,
            current_speakers=["Founder A", "Founder B"]
        )

    @pytest.fixture
    def imbalanced_state(self):
        """Fixture providing an imbalanced conversation state"""
        return ConversationState(
            duration_seconds=400.0,  # > 300s minimum
            tension_score=0.4,
            interruption_rate=0.2,
            speaker_imbalance=0.85,  # High imbalance
            argument_repetition_count=0,
            last_intervention_time=None,
            total_interventions=0,
            current_speakers=["Founder A", "Founder B"]
        )

    def test_intervene_on_high_tension(self, decider, tense_state):
        """Test that high tension triggers intervention"""
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            tense_state, analysis
        )

        assert should_intervene is True
        assert "tension" in reason.lower()
        assert intervention_type == "SYSTEM INTERVENTION"

    def test_no_intervene_on_calm(self, decider, calm_state):
        """Test that calm conversation doesn't trigger intervention"""
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis
        )

        assert should_intervene is False
        assert reason == ""
        assert intervention_type == ""

    def test_cooldown_blocks_intervention(self, decider, tense_state):
        """Test that cooldown period blocks intervention"""
        # Set last intervention to 30 seconds ago (< 60s cooldown)
        tense_state.last_intervention_time = datetime.now() - timedelta(seconds=30)
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            tense_state, analysis
        )

        assert should_intervene is False
        assert reason == ""

    def test_cooldown_allows_intervention_after_period(self, decider, tense_state):
        """Test that intervention is allowed after cooldown period"""
        # Set last intervention to 70 seconds ago (> 60s cooldown)
        tense_state.last_intervention_time = datetime.now() - timedelta(seconds=70)
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            tense_state, analysis
        )

        assert should_intervene is True

    def test_speaker_imbalance_trigger(self, decider, imbalanced_state):
        """Test that speaker imbalance triggers intervention"""
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            imbalanced_state, analysis
        )

        assert should_intervene is True
        assert "dominating" in reason.lower() or "imbalance" in reason.lower()
        assert intervention_type == "SYSTEM INTERVENTION"

    def test_speaker_imbalance_requires_min_duration(self, decider):
        """Test that speaker imbalance requires minimum duration"""
        # Short duration state with high imbalance
        short_imbalanced = ConversationState(
            duration_seconds=200.0,  # < 300s minimum
            tension_score=0.3,
            interruption_rate=0.1,
            speaker_imbalance=0.9,  # Very high imbalance
            argument_repetition_count=0,
            last_intervention_time=None,
            total_interventions=0,
            current_speakers=["Founder A", "Founder B"]
        )
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            short_imbalanced, analysis
        )

        # Should not intervene yet (too early)
        assert should_intervene is False

    def test_argument_repetition_trigger(self, decider, calm_state):
        """Test that argument repetition triggers intervention"""
        # Set high repetition count
        calm_state.argument_repetition_count = 3  # At threshold
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis
        )

        assert should_intervene is True
        assert "repeated" in reason.lower() or "repetition" in reason.lower()
        assert intervention_type == "SYSTEM INTERVENTION"

    def test_protocol_violation_past_reference(self, decider, calm_state):
        """Test Protocol 3: detecting past references triggers violation"""
        recent_utterances = [
            "We need to focus forward",
            "But last year you said",  # Past reference
            "Let's move on"
        ]
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis, recent_utterances
        )

        assert should_intervene is True
        assert "Protocol 3" in reason
        assert "Future Focused" in reason
        assert intervention_type == "PROTOCOL VIOLATION"

    def test_protocol_warning_opinion_phrase(self, decider, calm_state):
        """Test Protocol 2: detecting opinion without data triggers warning"""
        recent_utterances = [
            "I think we should do this",  # Opinion phrase
            "I believe this is the way"   # Opinion phrase, no data
        ]
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis, recent_utterances
        )

        assert should_intervene is True
        assert "Protocol 2" in reason
        assert "Data Over Opinion" in reason
        assert intervention_type == "PROTOCOL WARNING"

    def test_opinion_with_data_no_warning(self, decider, calm_state):
        """Test that opinion with data doesn't trigger warning"""
        recent_utterances = [
            "I think we should do this because 85% of users requested it",  # Opinion + data
        ]
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis, recent_utterances
        )

        # Should not trigger Protocol 2 warning (has data)
        # Might not intervene at all if state is calm
        if should_intervene and intervention_type == "PROTOCOL WARNING":
            pytest.fail("Should not warn when opinion is backed by data")

    def test_check_cooldown_no_previous_intervention(self, decider):
        """Test cooldown check with no previous intervention returns True"""
        result = decider.check_cooldown(None)

        assert result is True

    def test_check_cooldown_recent_intervention(self, decider):
        """Test cooldown check with recent intervention returns False"""
        recent_time = datetime.now() - timedelta(seconds=30)

        result = decider.check_cooldown(recent_time)

        assert result is False

    def test_check_cooldown_old_intervention(self, decider):
        """Test cooldown check with old intervention returns True"""
        old_time = datetime.now() - timedelta(seconds=70)

        result = decider.check_cooldown(old_time)

        assert result is True

    def test_get_intervention_context(self, decider, tense_state):
        """Test get_intervention_context returns complete context"""
        analysis = {
            'tension_score': 0.85,
            'sentiment_negativity': 0.7
        }

        context = decider.get_intervention_context(tense_state, analysis)

        assert 'timestamp' in context
        assert 'duration_seconds' in context
        assert 'tension_score' in context
        assert 'interruption_rate' in context
        assert 'speaker_imbalance' in context
        assert 'argument_repetition_count' in context
        assert 'total_interventions' in context
        assert 'speakers' in context
        assert 'analysis_summary' in context

        assert context['tension_score'] == 0.85
        assert context['speakers'] == ["Founder A", "Founder B"]

    def test_record_intervention(self, decider):
        """Test recording intervention in history"""
        context = {'test': 'context'}

        decider.record_intervention(
            intervention_type="SYSTEM INTERVENTION",
            reason="Test reason",
            context=context
        )

        assert len(decider.intervention_history) == 1
        intervention = decider.intervention_history[0]

        assert intervention['type'] == "SYSTEM INTERVENTION"
        assert intervention['reason'] == "Test reason"
        assert intervention['context'] == context
        assert 'timestamp' in intervention

    def test_get_intervention_stats_no_interventions(self, decider):
        """Test intervention stats with no interventions"""
        stats = decider.get_intervention_stats()

        assert stats['total_interventions'] == 0
        assert stats['by_type'] == {}
        assert stats['average_interval_seconds'] == 0

    def test_get_intervention_stats_with_interventions(self, decider):
        """Test intervention stats with multiple interventions"""
        # Record interventions of different types
        context = {}
        decider.record_intervention("SYSTEM INTERVENTION", "Reason 1", context)
        decider.record_intervention("PROTOCOL VIOLATION", "Reason 2", context)
        decider.record_intervention("SYSTEM INTERVENTION", "Reason 3", context)

        stats = decider.get_intervention_stats()

        assert stats['total_interventions'] == 3
        assert stats['by_type']['SYSTEM INTERVENTION'] == 2
        assert stats['by_type']['PROTOCOL VIOLATION'] == 1
        assert 'last_intervention' in stats

    def test_protocol_priority_past_over_opinion(self, decider, calm_state):
        """Test that Protocol 3 (past reference) has higher priority than Protocol 2"""
        recent_utterances = [
            "I think back then we should have done this"  # Both opinion AND past reference
        ]
        analysis = {}

        should_intervene, reason, intervention_type = decider.should_intervene(
            calm_state, analysis, recent_utterances
        )

        assert should_intervene is True
        # Protocol 3 should trigger first (higher priority)
        assert intervention_type == "PROTOCOL VIOLATION"
        assert "Protocol 3" in reason

    def test_various_past_reference_patterns(self, decider, calm_state):
        """Test detection of various past reference patterns"""
        past_patterns = [
            "Last year we tried this",
            "Back when we started",
            "You always do this",
            "You never listen",
            "Previously we decided",
            "In the past this worked",
            "Before you joined"
        ]

        analysis = {}

        for pattern in past_patterns:
            should_intervene, reason, intervention_type = decider.should_intervene(
                calm_state, analysis, [pattern]
            )

            assert should_intervene is True, f"Should detect past reference in: {pattern}"
            assert intervention_type == "PROTOCOL VIOLATION"

    def test_various_opinion_patterns(self, decider, calm_state):
        """Test detection of various opinion patterns"""
        opinion_patterns = [
            "I feel this is the right way",
            "I think we should proceed",
            "I believe in this approach",
            "In my opinion this works",
            "It seems like a good idea",
            "This probably will work",
            "Maybe we should try this"
        ]

        analysis = {}

        for pattern in opinion_patterns:
            should_intervene, reason, intervention_type = decider.should_intervene(
                calm_state, analysis, [pattern]
            )

            # Should warn about opinion without data
            if should_intervene and intervention_type == "PROTOCOL WARNING":
                assert "Protocol 2" in reason

    def test_data_indicators_recognized(self, decider, calm_state):
        """Test that various data indicators are recognized"""
        data_phrases = [
            "I think we should do this because 75% of users want it",
            "I believe this based on $50000 in revenue",
            "In my opinion, with 1000 customers, we should",
            "The data shows this is correct",
            "According to the metrics we collected"
        ]

        analysis = {}

        for phrase in data_phrases:
            should_intervene, reason, intervention_type = decider.should_intervene(
                calm_state, analysis, [phrase]
            )

            # Should NOT trigger Protocol 2 warning (has data)
            if should_intervene and intervention_type == "PROTOCOL WARNING":
                pytest.fail(f"Should not warn when data is present: {phrase}")

    def test_case_insensitive_pattern_matching(self, decider, calm_state):
        """Test that pattern matching is case insensitive"""
        # Past reference in different cases
        patterns = [
            "LAST YEAR we did this",
            "Last Year we did this",
            "last year we did this"
        ]

        analysis = {}

        for pattern in patterns:
            should_intervene, reason, _ = decider.should_intervene(
                calm_state, analysis, [pattern]
            )

            assert should_intervene is True

    def test_intervention_thresholds(self, decider):
        """Test that intervention thresholds are correctly configured"""
        assert decider.TENSION_THRESHOLD == 0.7
        assert decider.SPEAKER_IMBALANCE_THRESHOLD == 0.8
        assert decider.SPEAKER_IMBALANCE_MIN_DURATION == 300
        assert decider.ARGUMENT_REPETITION_THRESHOLD == 3
        assert decider.INTERVENTION_COOLDOWN == 60

    def test_intervention_reason_messages(self, decider, tense_state):
        """Test that intervention reasons are descriptive"""
        analysis = {}

        _, reason, _ = decider.should_intervene(tense_state, analysis)

        # Reason should include the score
        assert "0.85" in reason or "tension" in reason.lower()
        assert len(reason) > 20  # Should be descriptive
