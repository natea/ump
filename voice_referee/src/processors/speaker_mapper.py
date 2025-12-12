"""
Speaker Mapper Module

Maps Deepgram speaker IDs (0, 1, 2, ...) to founder names.
Maintains persistent mapping throughout the session.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SpeakerMapper:
    """
    Maps Deepgram speaker IDs to founder identities.

    The first speaker detected is assigned "Founder A" (or custom name from config).
    The second speaker detected is assigned "Founder B".
    Mapping persists throughout the session.

    Attributes:
        _speaker_map: Dictionary mapping speaker IDs to founder names
        _founder_names: List of founder names to assign (default: ["Founder A", "Founder B"])
        _next_assignment_index: Index of next founder name to assign
    """

    def __init__(self, founder_names: Optional[list[str]] = None):
        """
        Initialize the SpeakerMapper.

        Args:
            founder_names: Optional list of founder names. Defaults to ["Founder A", "Founder B"]
        """
        self._speaker_map: Dict[int, str] = {}
        self._founder_names = founder_names or ["Founder A", "Founder B"]
        self._next_assignment_index = 0

        logger.info(f"SpeakerMapper initialized with founder names: {self._founder_names}")

    def assign_identity(self, speaker_id: int) -> str:
        """
        Assign an identity to a speaker ID if not already assigned.

        Args:
            speaker_id: Deepgram speaker ID (0, 1, 2, ...)

        Returns:
            The founder name assigned to this speaker

        Raises:
            ValueError: If all founder names have been assigned and a new speaker is detected
        """
        # Return existing assignment if already mapped
        if speaker_id in self._speaker_map:
            return self._speaker_map[speaker_id]

        # Check if we have founder names available
        if self._next_assignment_index >= len(self._founder_names):
            error_msg = (
                f"Cannot assign identity to speaker {speaker_id}: "
                f"All {len(self._founder_names)} founder names already assigned"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Assign next available founder name
        founder_name = self._founder_names[self._next_assignment_index]
        self._speaker_map[speaker_id] = founder_name
        self._next_assignment_index += 1

        logger.info(f"Assigned speaker {speaker_id} to {founder_name}")
        return founder_name

    def get_identity(self, speaker_id: int) -> str:
        """
        Get the identity assigned to a speaker ID.

        Args:
            speaker_id: Deepgram speaker ID

        Returns:
            The founder name assigned to this speaker

        Raises:
            KeyError: If the speaker ID has not been assigned yet
        """
        if speaker_id not in self._speaker_map:
            error_msg = f"Speaker {speaker_id} has not been assigned an identity yet"
            logger.error(error_msg)
            raise KeyError(error_msg)

        return self._speaker_map[speaker_id]

    def is_assigned(self, speaker_id: int) -> bool:
        """
        Check if a speaker ID has been assigned an identity.

        Args:
            speaker_id: Deepgram speaker ID

        Returns:
            True if assigned, False otherwise
        """
        return speaker_id in self._speaker_map

    def get_all_mappings(self) -> Dict[int, str]:
        """
        Get all speaker ID to founder name mappings.

        Returns:
            Dictionary of speaker IDs to founder names
        """
        return self._speaker_map.copy()

    def reset(self):
        """Reset the speaker mapping (for new sessions)."""
        logger.info("Resetting speaker mappings")
        self._speaker_map.clear()
        self._next_assignment_index = 0

    def __repr__(self) -> str:
        """String representation of the mapper."""
        return f"SpeakerMapper(mappings={self._speaker_map}, available={self._founder_names[self._next_assignment_index:]})"
