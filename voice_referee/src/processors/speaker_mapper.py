"""
Speaker Mapper Module

Maps Deepgram speaker IDs (0, 1, 2, ...) to participant names.
Maintains persistent mapping throughout the session.

Supports dynamic participant registration from Daily.co events.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SpeakerMapper:
    """
    Maps Deepgram speaker IDs to participant identities.

    Supports two modes:
    1. Dynamic: Participants register when they join Daily room (preferred)
    2. Fallback: Assigns generic names ("Founder A", "Founder B") if no participants registered

    The mapping between Deepgram speaker IDs and participant names is established
    based on speaking order - the first speaker detected gets the first registered name.

    Attributes:
        _speaker_map: Dictionary mapping speaker IDs to participant names
        _participant_names: List of participant names (from Daily join events)
        _fallback_names: Default names if no participants registered
        _next_assignment_index: Index of next name to assign
    """

    def __init__(self, founder_names: Optional[list[str]] = None):
        """
        Initialize the SpeakerMapper.

        Args:
            founder_names: Optional list of fallback names. Defaults to ["Founder A", "Founder B"]
        """
        self._speaker_map: Dict[int, str] = {}
        self._participant_names: List[str] = []  # Names from Daily participant events
        self._fallback_names = founder_names or ["Founder A", "Founder B"]
        self._next_assignment_index = 0

        logger.info(f"SpeakerMapper initialized with fallback names: {self._fallback_names}")

    def register_participant(self, participant_id: str, user_name: str) -> None:
        """
        Register a participant from Daily.co when they join the room.

        Called when a participant joins the Daily room. The user_name is the
        display name they set when joining.

        Args:
            participant_id: Daily participant ID (session_id)
            user_name: Display name set by the participant
        """
        # Don't register the bot itself
        if "referee" in user_name.lower() or "bot" in user_name.lower():
            logger.debug(f"Skipping bot registration: {user_name}")
            return

        # Don't register duplicates
        if user_name in self._participant_names:
            logger.debug(f"Participant already registered: {user_name}")
            return

        self._participant_names.append(user_name)
        logger.info(f"📝 Registered participant: {user_name} (Daily ID: {participant_id})")
        logger.info(f"Current participants: {self._participant_names}")

    def unregister_participant(self, participant_id: str, user_name: str) -> None:
        """
        Remove a participant when they leave the room.

        Args:
            participant_id: Daily participant ID
            user_name: Display name of the participant
        """
        if user_name in self._participant_names:
            self._participant_names.remove(user_name)
            logger.info(f"📝 Unregistered participant: {user_name}")

    def get_participant_names(self) -> List[str]:
        """Get list of registered participant names."""
        return self._participant_names.copy()

    def _get_available_names(self) -> List[str]:
        """
        Get the list of names to use for assignment.

        Returns participant names if registered, otherwise fallback names.
        """
        if self._participant_names:
            return self._participant_names
        return self._fallback_names

    def assign_identity(self, speaker_id: int) -> str:
        """
        Assign an identity to a speaker ID if not already assigned.

        Uses participant names from Daily.co if available, otherwise falls back
        to generic names ("Founder A", "Founder B").

        If more speakers are detected than names available (common with diarization),
        extra speakers are mapped to "Unknown Speaker N".

        Args:
            speaker_id: Deepgram speaker ID (0, 1, 2, ...)

        Returns:
            The participant name assigned to this speaker
        """
        # Return existing assignment if already mapped
        if speaker_id in self._speaker_map:
            return self._speaker_map[speaker_id]

        # Get available names (participant names or fallback)
        available_names = self._get_available_names()

        # Check if we have names available
        if self._next_assignment_index >= len(available_names):
            # Diarization detected more speakers than expected
            # This is common - assign a generic name and log a warning
            participant_name = f"Unknown Speaker {speaker_id}"
            self._speaker_map[speaker_id] = participant_name
            logger.warning(
                f"🎤 Extra speaker detected: speaker {speaker_id} → {participant_name} "
                f"(expected max {len(available_names)} speakers)"
            )
            return participant_name

        # Assign next available name
        participant_name = available_names[self._next_assignment_index]
        self._speaker_map[speaker_id] = participant_name
        self._next_assignment_index += 1

        logger.info(f"🎤 Assigned speaker {speaker_id} → {participant_name}")
        return participant_name

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
        Get all speaker ID to participant name mappings.

        Returns:
            Dictionary of speaker IDs to participant names
        """
        return self._speaker_map.copy()

    def reset(self):
        """Reset the speaker mapping (for new sessions)."""
        logger.info("Resetting speaker mappings")
        self._speaker_map.clear()
        self._participant_names.clear()
        self._next_assignment_index = 0

    def __repr__(self) -> str:
        """String representation of the mapper."""
        available = self._get_available_names()[self._next_assignment_index:]
        return f"SpeakerMapper(mappings={self._speaker_map}, participants={self._participant_names}, available={available})"
