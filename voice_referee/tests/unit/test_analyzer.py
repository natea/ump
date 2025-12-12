"""
Unit tests for ConversationAnalyzer module

Tests tension score calculation, sentiment detection, interruption analysis,
and argument repetition detection.
"""

import pytest
from analysis.conversation_analyzer import (
    ConversationAnalyzer,
    Utterance
)


class TestConversationAnalyzer:
    """Test suite for ConversationAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Fixture providing a fresh ConversationAnalyzer instance"""
        return ConversationAnalyzer()

    @pytest.fixture
    def calm_transcript(self):
        """Fixture providing a calm conversation transcript"""
        return [
            Utterance("I agree with your point", "Founder A", 100.0, 2.0, False),
            Utterance("Thank you for understanding", "Founder B", 102.0, 2.0, False),
            Utterance("Let's move forward together", "Founder A", 104.0, 2.0, False)
        ]

    @pytest.fixture
    def tense_transcript(self):
        """Fixture providing a tense conversation transcript"""
        return [
            Utterance("That's completely wrong and stupid", "Founder A", 100.0, 2.0, True),
            Utterance("You never listen to me", "Founder B", 102.0, 2.0, True),
            Utterance("This is impossible and terrible", "Founder A", 104.0, 2.0, True),
            Utterance("You always make the worst decisions", "Founder B", 106.0, 2.0, True)
        ]

    def test_tension_score_high_tension(self, analyzer, tense_transcript):
        """Test tension score calculation with high-tension conversation"""
        tension_score = analyzer.calculate_tension_score(tense_transcript)

        # Should be high tension (> 0.5)
        assert tension_score > 0.5
        assert 0.0 <= tension_score <= 1.0

    def test_tension_score_calm(self, analyzer, calm_transcript):
        """Test tension score calculation with calm conversation"""
        tension_score = analyzer.calculate_tension_score(calm_transcript)

        # Should be low tension (< 0.3)
        assert tension_score < 0.3
        assert 0.0 <= tension_score <= 1.0

    def test_tension_score_empty_transcript(self, analyzer):
        """Test tension score with empty transcript returns 0.0"""
        tension_score = analyzer.calculate_tension_score([])

        assert tension_score == 0.0

    def test_sentiment_detection_negative(self, analyzer):
        """Test sentiment detection for negative text"""
        negative_texts = [
            "This is terrible and wrong",
            "I hate this stupid idea",
            "You never understand anything",
            "This is the worst plan ever"
        ]

        for text in negative_texts:
            sentiment = analyzer.detect_sentiment(text)
            assert sentiment < 0.0, f"Expected negative sentiment for: {text}"
            assert -1.0 <= sentiment <= 1.0

    def test_sentiment_detection_positive(self, analyzer):
        """Test sentiment detection for positive text"""
        positive_texts = [
            "This is great and excellent",
            "I agree with your perfect idea",
            "Thank you for the good work",
            "Yes, that's absolutely right"
        ]

        for text in positive_texts:
            sentiment = analyzer.detect_sentiment(text)
            assert sentiment > 0.0, f"Expected positive sentiment for: {text}"
            assert -1.0 <= sentiment <= 1.0

    def test_sentiment_detection_neutral(self, analyzer):
        """Test sentiment detection for neutral text"""
        neutral_text = "The meeting is at 3pm tomorrow"

        sentiment = analyzer.detect_sentiment(neutral_text)

        # Should be close to 0.0 for neutral text
        assert abs(sentiment) < 0.2
        assert -1.0 <= sentiment <= 1.0

    def test_sentiment_detection_empty_text(self, analyzer):
        """Test sentiment detection with empty text returns 0.0"""
        sentiment = analyzer.detect_sentiment("")

        assert sentiment == 0.0

    def test_sentiment_tension_keywords_count_double(self, analyzer):
        """Test that tension keywords have stronger negative impact"""
        # Text with tension keyword "never"
        tension_text = "You never listen"

        # Text with similar negative keyword "bad"
        negative_text = "You are bad at listening"

        tension_sentiment = analyzer.detect_sentiment(tension_text)
        negative_sentiment = analyzer.detect_sentiment(negative_text)

        # Tension keyword should have more negative impact
        assert tension_sentiment < negative_sentiment

    def test_argument_repetition_detected(self, analyzer):
        """Test detection of repeated arguments"""
        # Create utterances with repeated keywords
        repeated_utterances = [
            Utterance("We need better marketing strategy", "Founder A", 100.0, 2.0, False),
            Utterance("I disagree with that", "Founder B", 102.0, 1.0, False),
            Utterance("Marketing strategy is essential here", "Founder A", 104.0, 2.0, False),
            Utterance("The marketing strategy needs work", "Founder B", 106.0, 2.0, False)
        ]

        repetition_count = analyzer.detect_argument_repetition(repeated_utterances)

        # Should detect repetition (> 0)
        assert repetition_count > 0

    def test_argument_repetition_no_repetition(self, analyzer):
        """Test that different topics don't trigger repetition detection"""
        different_topics = [
            Utterance("Let's discuss the budget", "Founder A", 100.0, 2.0, False),
            Utterance("What about hiring plans", "Founder B", 102.0, 2.0, False),
            Utterance("Product roadmap looks good", "Founder A", 104.0, 2.0, False)
        ]

        repetition_count = analyzer.detect_argument_repetition(different_topics)

        # Should detect minimal or no repetition
        assert repetition_count <= 1

    def test_argument_repetition_too_few_utterances(self, analyzer):
        """Test repetition detection with too few utterances returns 0"""
        single_utterance = [
            Utterance("Hello", "Founder A", 100.0, 1.0, False)
        ]

        repetition_count = analyzer.detect_argument_repetition(single_utterance)

        assert repetition_count == 0

    def test_calculate_interruption_rate(self, analyzer):
        """Test interruption rate calculation"""
        transcript = [
            Utterance("Hello", "Founder A", 100.0, 1.0, False),
            Utterance("Wait", "Founder B", 101.0, 0.5, True),  # Interruption
            Utterance("Let me finish", "Founder A", 101.5, 1.0, False),
            Utterance("Sorry", "Founder B", 102.5, 0.5, True),  # Interruption
        ]

        rate = analyzer.calculate_interruption_rate(transcript)

        assert rate == 0.5  # 2 out of 4 are interruptions

    def test_calculate_interruption_rate_no_interruptions(self, analyzer):
        """Test interruption rate with no interruptions"""
        transcript = [
            Utterance("Hello", "Founder A", 100.0, 1.0, False),
            Utterance("Hi", "Founder B", 101.0, 1.0, False),
        ]

        rate = analyzer.calculate_interruption_rate(transcript)

        assert rate == 0.0

    def test_calculate_interruption_rate_empty(self, analyzer):
        """Test interruption rate with empty transcript"""
        rate = analyzer.calculate_interruption_rate([])

        assert rate == 0.0

    def test_get_analysis_summary(self, analyzer, calm_transcript):
        """Test get_analysis_summary returns complete analysis"""
        summary = analyzer.get_analysis_summary(calm_transcript)

        assert 'tension_score' in summary
        assert 'sentiment_negativity' in summary
        assert 'interruption_rate' in summary
        assert 'speaker_imbalance' in summary
        assert 'argument_repetition' in summary
        assert 'utterance_count' in summary
        assert 'timestamp' in summary

        assert summary['utterance_count'] == len(calm_transcript)
        assert 0.0 <= summary['tension_score'] <= 1.0

    def test_speaker_imbalance_balanced(self, analyzer):
        """Test speaker imbalance calculation with balanced speakers"""
        balanced_transcript = [
            Utterance("Hello world test", "Founder A", 100.0, 2.0, False),  # 3 words
            Utterance("Hi there friend", "Founder B", 102.0, 2.0, False),   # 3 words
            Utterance("Good to see you", "Founder A", 104.0, 2.0, False),   # 4 words
            Utterance("Same here buddy", "Founder B", 106.0, 2.0, False)    # 3 words
        ]

        summary = analyzer.get_analysis_summary(balanced_transcript)

        # Should be relatively balanced (< 0.3)
        assert summary['speaker_imbalance'] < 0.3

    def test_speaker_imbalance_imbalanced(self, analyzer):
        """Test speaker imbalance calculation with imbalanced speakers"""
        imbalanced_transcript = [
            Utterance("Long sentence with many words here", "Founder A", 100.0, 3.0, False),  # 6 words
            Utterance("Another long explanation with details", "Founder A", 103.0, 3.0, False),  # 5 words
            Utterance("More content from the same person", "Founder A", 106.0, 3.0, False),  # 6 words
            Utterance("Hi", "Founder B", 109.0, 0.5, False)  # 1 word
        ]

        summary = analyzer.get_analysis_summary(imbalanced_transcript)

        # Should be significantly imbalanced (> 0.5)
        assert summary['speaker_imbalance'] > 0.5

    def test_tension_score_components_weighted(self, analyzer):
        """Test that tension score properly weights all components"""
        # Create transcript with known characteristics
        transcript = [
            Utterance("This is terrible", "Founder A", 100.0, 1.0, True),  # High negativity + interruption
            Utterance("Wrong", "Founder B", 101.0, 0.5, False),
        ]

        tension = analyzer.calculate_tension_score(transcript)

        # Should be composite of multiple factors
        assert tension > 0.0
        assert tension < 1.0  # Not maximum since not all factors are maxed

    def test_sentiment_keywords_are_case_insensitive(self, analyzer):
        """Test that sentiment detection is case insensitive"""
        lower_text = "this is terrible and wrong"
        upper_text = "THIS IS TERRIBLE AND WRONG"
        mixed_text = "This Is Terrible And Wrong"

        lower_sentiment = analyzer.detect_sentiment(lower_text)
        upper_sentiment = analyzer.detect_sentiment(upper_text)
        mixed_sentiment = analyzer.detect_sentiment(mixed_text)

        # All should have same sentiment
        assert lower_sentiment == upper_sentiment == mixed_sentiment

    def test_keyword_extraction_filters_short_words(self, analyzer):
        """Test that repetition detection filters out short words"""
        # Short common words should be filtered
        utterances = [
            Utterance("I am with you on this", "Founder A", 100.0, 1.0, False),
            Utterance("I am with you on that", "Founder B", 101.0, 1.0, False)
        ]

        # Should not detect high repetition due to "I am with you on"
        # because these are short/common words
        repetition = analyzer.detect_argument_repetition(utterances)

        # Should be 0 or very low
        assert repetition <= 1

    def test_multiple_positive_keywords_increase_sentiment(self, analyzer):
        """Test that multiple positive keywords increase positive sentiment"""
        text_one_positive = "This is good"
        text_many_positive = "This is good great excellent perfect"

        sentiment_one = analyzer.detect_sentiment(text_one_positive)
        sentiment_many = analyzer.detect_sentiment(text_many_positive)

        assert sentiment_many > sentiment_one

    def test_multiple_negative_keywords_decrease_sentiment(self, analyzer):
        """Test that multiple negative keywords decrease sentiment more"""
        text_one_negative = "This is bad"
        text_many_negative = "This is bad terrible awful worst"

        sentiment_one = analyzer.detect_sentiment(text_one_negative)
        sentiment_many = analyzer.detect_sentiment(text_many_negative)

        assert sentiment_many < sentiment_one

    def test_argument_repetition_threshold(self, analyzer):
        """Test that 40% keyword overlap triggers repetition detection"""
        # Create utterances with exactly 40% overlap
        utterances = [
            Utterance("marketing strategy budget planning timeline", "A", 100.0, 2.0, False),
            Utterance("marketing strategy different topics here", "B", 102.0, 2.0, False)
        ]
        # "marketing" and "strategy" overlap = 2/5 = 40%

        repetition = analyzer.detect_argument_repetition(utterances)

        assert repetition >= 1  # Should detect the overlap

    def test_tension_score_bounds(self, analyzer):
        """Test that tension score is always bounded between 0.0 and 1.0"""
        # Extreme cases
        extreme_transcript = [
            Utterance("never always wrong stupid terrible worst hate", "A", 100.0, 2.0, True),
            Utterance("never always wrong stupid terrible worst hate", "A", 102.0, 2.0, True),
            Utterance("never always wrong stupid terrible worst hate", "A", 104.0, 2.0, True)
        ] * 10  # Many extreme utterances

        tension = analyzer.calculate_tension_score(extreme_transcript)

        # Must be bounded
        assert 0.0 <= tension <= 1.0
