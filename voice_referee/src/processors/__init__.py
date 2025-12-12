"""
Voice Referee Processors Module

This module provides processing components for managing speaker identification
and conversation state in the Voice Referee system.
"""

from .speaker_mapper import SpeakerMapper
from .conversation_state import ConversationState, Utterance, SpeakerStats

__all__ = [
    'SpeakerMapper',
    'ConversationState',
    'Utterance',
    'SpeakerStats',
]
