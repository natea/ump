"""Tests for Deepgram STT service with diarization.

These tests verify the configuration and creation of the Deepgram STT
service with proper diarization settings.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from services.deepgram_stt import (
    DeepgramConfig,
    create_deepgram_stt,
    create_default_stt,
    DiarizedDeepgramSTTService,
)


class TestDeepgramConfig:
    """Test suite for DeepgramConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = DeepgramConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model == "nova-2"
        assert config.language == "en"
        assert config.diarize is True  # CRITICAL
        assert config.punctuate is True
        assert config.interim_results is True
        assert config.smart_format is True
        assert config.utterance_end_ms == 1000

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = DeepgramConfig(
            api_key="custom-key",
            model="nova-3",
            language="es",
            diarize=False,
            punctuate=False,
            interim_results=False,
            smart_format=False,
            utterance_end_ms=2000,
        )

        assert config.api_key == "custom-key"
        assert config.model == "nova-3"
        assert config.language == "es"
        assert config.diarize is False
        assert config.punctuate is False
        assert config.interim_results is False
        assert config.smart_format is False
        assert config.utterance_end_ms == 2000

    def test_validate_missing_api_key(self):
        """Test validation fails with missing API key."""
        config = DeepgramConfig(api_key="")

        with pytest.raises(ValueError, match="API key is required"):
            config.validate()

    def test_validate_utterance_end_too_low(self):
        """Test validation fails with utterance_end_ms too low."""
        config = DeepgramConfig(api_key="test-key", utterance_end_ms=50)

        with pytest.raises(ValueError, match="must be at least 100ms"):
            config.validate()

    def test_validate_utterance_end_warning(self, caplog):
        """Test validation warns with very high utterance_end_ms."""
        config = DeepgramConfig(api_key="test-key", utterance_end_ms=15000)

        with caplog.at_level(logging.WARNING):
            config.validate()
        assert "very high" in caplog.text.lower()

    def test_validate_success(self):
        """Test validation succeeds with valid config."""
        config = DeepgramConfig(api_key="test-key")

        # Should not raise
        config.validate()


class TestCreateDeepgramSTT:
    """Test suite for create_deepgram_stt factory function."""

    @patch('services.deepgram_stt.DiarizedDeepgramSTTService')
    def test_create_with_valid_config(self, mock_service_class):
        """Test service creation with valid configuration."""
        config = DeepgramConfig(api_key="test-key")
        mock_instance = Mock()
        mock_service_class.return_value = mock_instance

        service = create_deepgram_stt(config)

        assert service == mock_instance
        # Verify it was called with api_key and live_options
        mock_service_class.assert_called_once()
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs['api_key'] == "test-key"
        assert 'live_options' in call_kwargs
        # Verify live_options has correct values
        live_options = call_kwargs['live_options']
        assert live_options.diarize is True
        assert live_options.model == "nova-2"
        assert live_options.language == "en"

    def test_create_without_diarization_fails(self):
        """Test service creation fails when diarization is disabled."""
        config = DeepgramConfig(api_key="test-key", diarize=False)

        with pytest.raises(ValueError, match="Diarization must be enabled"):
            create_deepgram_stt(config)

    def test_create_with_invalid_config_fails(self):
        """Test service creation fails with invalid configuration."""
        config = DeepgramConfig(api_key="", utterance_end_ms=50)

        with pytest.raises(ValueError):
            create_deepgram_stt(config)

    @patch('services.deepgram_stt.DiarizedDeepgramSTTService')
    def test_create_logs_configuration(self, mock_service_class, caplog):
        """Test service creation logs configuration details."""
        config = DeepgramConfig(api_key="test-key")
        mock_service_class.return_value = Mock()

        with caplog.at_level(logging.INFO):
            create_deepgram_stt(config)

        log_text = caplog.text
        assert "nova-2" in log_text
        assert "Diarization" in log_text


class TestCreateDefaultSTT:
    """Test suite for create_default_stt convenience function."""

    @patch('services.deepgram_stt.create_deepgram_stt')
    def test_create_default_uses_correct_config(self, mock_create):
        """Test default creation uses correct configuration."""
        mock_create.return_value = Mock()

        service = create_default_stt("test-api-key")

        # Verify create_deepgram_stt was called
        assert mock_create.call_count == 1

        # Get the config that was passed
        config = mock_create.call_args[0][0]
        assert isinstance(config, DeepgramConfig)
        assert config.api_key == "test-api-key"
        assert config.diarize is True


class TestDiarizedDeepgramSTTService:
    """Test suite for DiarizedDeepgramSTTService."""

    @patch('services.deepgram_stt.DeepgramSTTService.__init__')
    def test_initialization(self, mock_super_init):
        """Test service initializes with statistics tracking."""
        mock_super_init.return_value = None

        service = DiarizedDeepgramSTTService()

        assert service._transcription_count == 0
        assert service._speaker_stats == {}

    @patch('services.deepgram_stt.DeepgramSTTService.__init__')
    def test_get_statistics(self, mock_super_init):
        """Test statistics retrieval."""
        mock_super_init.return_value = None
        service = DiarizedDeepgramSTTService()

        service._transcription_count = 10
        service._speaker_stats = {0: 6, 1: 4}

        stats = service.get_statistics()

        assert stats['total_transcriptions'] == 10
        assert stats['speaker_counts'] == {0: 6, 1: 4}
        assert stats['unique_speakers'] == 2

    @patch('services.deepgram_stt.DeepgramSTTService.__init__')
    def test_log_statistics(self, mock_super_init, caplog):
        """Test statistics logging."""
        mock_super_init.return_value = None
        service = DiarizedDeepgramSTTService()

        service._transcription_count = 10
        service._speaker_stats = {0: 6, 1: 4}

        with caplog.at_level(logging.INFO):
            service.log_statistics()

        assert "10" in caplog.text
        assert "2" in caplog.text  # unique speakers


class TestDiarizationCritical:
    """Critical tests for diarization functionality."""

    def test_diarization_enabled_by_default(self):
        """CRITICAL: Verify diarization is enabled by default."""
        config = DeepgramConfig(api_key="test-key")
        assert config.diarize is True, "Diarization MUST be enabled by default"

    def test_cannot_create_service_without_diarization(self):
        """CRITICAL: Verify service creation fails without diarization."""
        config = DeepgramConfig(api_key="test-key", diarize=False)

        with pytest.raises(ValueError, match="Diarization must be enabled"):
            create_deepgram_stt(config)

    @patch('services.deepgram_stt.DiarizedDeepgramSTTService')
    def test_diarization_parameter_passed_to_service(self, mock_service_class):
        """CRITICAL: Verify diarization parameter is passed to service."""
        config = DeepgramConfig(api_key="test-key")
        mock_service_class.return_value = Mock()

        create_deepgram_stt(config)

        # Verify diarize=True was passed in live_options
        call_kwargs = mock_service_class.call_args[1]
        live_options = call_kwargs['live_options']
        assert live_options.diarize is True
