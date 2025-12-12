"""
Unit tests for SpeakerMapper module

Tests speaker ID to founder name mapping, persistence, and reset functionality.
"""

import pytest
from processors.speaker_mapper import SpeakerMapper


class TestSpeakerMapper:
    """Test suite for SpeakerMapper class"""

    def test_first_speaker_assigned_founder_a(self):
        """Test that first speaker is assigned 'Founder A' by default"""
        mapper = SpeakerMapper()

        founder_name = mapper.assign_identity(0)

        assert founder_name == "Founder A"
        assert mapper.is_assigned(0)
        assert mapper.get_identity(0) == "Founder A"

    def test_second_speaker_assigned_founder_b(self):
        """Test that second speaker is assigned 'Founder B' by default"""
        mapper = SpeakerMapper()

        # Assign first speaker
        mapper.assign_identity(0)

        # Assign second speaker
        founder_name = mapper.assign_identity(1)

        assert founder_name == "Founder B"
        assert mapper.is_assigned(1)
        assert mapper.get_identity(1) == "Founder B"

    def test_speaker_identity_persistence(self):
        """Test that speaker identity persists across multiple calls"""
        mapper = SpeakerMapper()

        # First call assigns "Founder A"
        first_call = mapper.assign_identity(0)

        # Subsequent calls should return same identity
        second_call = mapper.assign_identity(0)
        third_call = mapper.assign_identity(0)

        assert first_call == "Founder A"
        assert second_call == "Founder A"
        assert third_call == "Founder A"

    def test_reset_clears_mapping(self):
        """Test that reset() clears all speaker mappings"""
        mapper = SpeakerMapper()

        # Assign speakers
        mapper.assign_identity(0)
        mapper.assign_identity(1)

        # Reset should clear mappings
        mapper.reset()

        assert not mapper.is_assigned(0)
        assert not mapper.is_assigned(1)
        assert mapper.get_all_mappings() == {}

    def test_custom_founder_names(self):
        """Test mapper with custom founder names"""
        custom_names = ["Alice", "Bob"]
        mapper = SpeakerMapper(founder_names=custom_names)

        alice = mapper.assign_identity(0)
        bob = mapper.assign_identity(1)

        assert alice == "Alice"
        assert bob == "Bob"

    def test_get_identity_unassigned_raises_key_error(self):
        """Test that get_identity raises KeyError for unassigned speaker"""
        mapper = SpeakerMapper()

        with pytest.raises(KeyError) as exc_info:
            mapper.get_identity(5)

        assert "has not been assigned" in str(exc_info.value)

    def test_too_many_speakers_raises_value_error(self):
        """Test that assigning more speakers than names available raises ValueError"""
        mapper = SpeakerMapper(founder_names=["Founder A", "Founder B"])

        # Assign both available names
        mapper.assign_identity(0)
        mapper.assign_identity(1)

        # Third speaker should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            mapper.assign_identity(2)

        assert "All 2 names already assigned" in str(exc_info.value)

    def test_get_all_mappings(self):
        """Test get_all_mappings returns correct dictionary"""
        mapper = SpeakerMapper()

        mapper.assign_identity(0)
        mapper.assign_identity(1)

        mappings = mapper.get_all_mappings()

        assert mappings == {0: "Founder A", 1: "Founder B"}
        # Verify it's a copy (mutations don't affect internal state)
        mappings[0] = "Changed"
        assert mapper.get_identity(0) == "Founder A"

    def test_is_assigned_returns_false_for_unassigned(self):
        """Test is_assigned returns False for unassigned speaker"""
        mapper = SpeakerMapper()

        assert not mapper.is_assigned(0)
        assert not mapper.is_assigned(99)

    def test_is_assigned_returns_true_for_assigned(self):
        """Test is_assigned returns True for assigned speaker"""
        mapper = SpeakerMapper()

        mapper.assign_identity(0)

        assert mapper.is_assigned(0)

    def test_reset_allows_reassignment(self):
        """Test that reset allows reassigning same speaker IDs"""
        mapper = SpeakerMapper()

        # Initial assignment
        first_assignment = mapper.assign_identity(0)
        assert first_assignment == "Founder A"

        # Reset
        mapper.reset()

        # Should be able to assign again
        second_assignment = mapper.assign_identity(0)
        assert second_assignment == "Founder A"

    def test_repr_shows_mappings(self):
        """Test __repr__ provides meaningful representation"""
        mapper = SpeakerMapper()
        mapper.assign_identity(0)

        repr_str = repr(mapper)

        assert "SpeakerMapper" in repr_str
        assert "mappings" in repr_str
        assert "Founder A" in repr_str

    def test_speaker_ids_can_be_non_sequential(self):
        """Test that speaker IDs don't need to be sequential"""
        mapper = SpeakerMapper()

        # Assign speaker 5 first, then speaker 2
        speaker_5 = mapper.assign_identity(5)
        speaker_2 = mapper.assign_identity(2)

        assert speaker_5 == "Founder A"
        assert speaker_2 == "Founder B"
        assert mapper.get_identity(5) == "Founder A"
        assert mapper.get_identity(2) == "Founder B"

    def test_empty_founder_names_list(self):
        """Test behavior with empty founder names list"""
        # Empty list parameter triggers default ["Founder A", "Founder B"]
        # This is the expected behavior from the implementation
        mapper = SpeakerMapper(founder_names=[])

        # Empty list triggers default assignment
        # The `or` operator in __init__ causes [] to use defaults
        assert mapper._fallback_names == ["Founder A", "Founder B"]

        # Assignment should work with defaults
        identity = mapper.assign_identity(0)
        assert identity == "Founder A"

    def test_single_founder_name(self):
        """Test mapper with only one founder name"""
        mapper = SpeakerMapper(founder_names=["Solo Founder"])

        first = mapper.assign_identity(0)
        assert first == "Solo Founder"

        # Second assignment should fail
        with pytest.raises(ValueError):
            mapper.assign_identity(1)
