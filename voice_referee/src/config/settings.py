"""Configuration settings for Voice Referee using Pydantic."""

import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DailyConfig(BaseModel):
    """Configuration for Daily.co room connection."""

    room_url: str = Field(..., description="Daily.co room URL")
    token: str = Field(..., description="Daily.co authentication token")

    @field_validator("room_url")
    @classmethod
    def validate_room_url(cls, v: str) -> str:
        """Validate room URL format."""
        if not v.startswith("https://"):
            raise ValueError("Room URL must start with https://")
        if ".daily.co/" not in v:
            raise ValueError("Room URL must contain .daily.co/")
        return v


class DeepgramConfig(BaseModel):
    """Configuration for Deepgram STT with diarization."""

    api_key: str = Field(..., description="Deepgram API key")
    model: str = Field(default="nova-2", description="Deepgram model to use")
    diarize: bool = Field(default=True, description="Enable speaker diarization")
    language: str = Field(default="en-US", description="Language code")
    smart_format: bool = Field(default=True, description="Enable smart formatting")
    punctuate: bool = Field(default=True, description="Enable punctuation")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Deepgram model selection."""
        valid_models = ["nova-2", "nova", "enhanced", "base"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM provider (Claude)."""

    provider: str = Field(default="anthropic", description="LLM provider name")
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model identifier"
    )
    api_key: str = Field(..., description="API key for LLM provider")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, description="Temperature for generation")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = ["anthropic", "openai"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech (ElevenLabs)."""

    provider: str = Field(default="elevenlabs", description="TTS provider name")
    voice_id: str = Field(..., description="Voice identifier")
    api_key: str = Field(..., description="API key for TTS provider")
    model: str = Field(
        default="eleven_flash_v2_5",
        description="TTS model to use"
    )
    optimize_streaming_latency: int = Field(
        default=3,
        description="Latency optimization level (0-4)"
    )
    stability: float = Field(default=0.5, description="Voice stability (0.0-1.0)")
    similarity_boost: float = Field(
        default=0.75,
        description="Voice similarity boost (0.0-1.0)"
    )

    @field_validator("optimize_streaming_latency")
    @classmethod
    def validate_latency(cls, v: int) -> int:
        """Validate latency optimization level."""
        if not 0 <= v <= 4:
            raise ValueError("Latency optimization must be between 0 and 4")
        return v

    @field_validator("stability", "similarity_boost")
    @classmethod
    def validate_voice_params(cls, v: float) -> float:
        """Validate voice parameter ranges."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Voice parameters must be between 0.0 and 1.0")
        return v


class ProcessorConfig(BaseModel):
    """Configuration for conversation processors."""

    tension_threshold: float = Field(
        default=0.7,
        description="Threshold for tension detection (0.0-1.0)"
    )
    cooldown_seconds: int = Field(
        default=30,
        description="Cooldown period between interventions"
    )
    buffer_size: int = Field(
        default=50,
        description="Number of conversation turns to keep in buffer"
    )

    @field_validator("tension_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate tension threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Tension threshold must be between 0.0 and 1.0")
        return v

    @field_validator("cooldown_seconds")
    @classmethod
    def validate_cooldown(cls, v: int) -> int:
        """Validate cooldown period."""
        if v < 0:
            raise ValueError("Cooldown seconds must be non-negative")
        return v

    @field_validator("buffer_size")
    @classmethod
    def validate_buffer(cls, v: int) -> int:
        """Validate buffer size."""
        if v < 1:
            raise ValueError("Buffer size must be at least 1")
        return v


class VisionConfig(BaseModel):
    """Configuration for vision-based screen analysis."""

    enabled: bool = Field(default=False, description="Enable vision analysis")
    provider: str = Field(default="anthropic", description="Vision provider")
    model: str = Field(default="claude-3-5-sonnet-20241022", description="Vision model")
    api_key: str = Field(default="", description="Vision API key (uses LLM key if empty)")

    # Analysis settings
    frame_capture_mode: str = Field(default="on_demand", description="on_demand or continuous")
    analysis_interval_seconds: float = Field(default=2.0, description="Seconds between analyses")
    max_analysis_cost_per_session: float = Field(default=0.30, description="Max cost per session")

    # Commentary settings
    commentary_style: str = Field(default="concise", description="concise, detailed, or technical")
    commentary_trigger_threshold: float = Field(default=0.6, description="Threshold for triggering commentary")

    # Performance settings
    max_vision_latency_ms: int = Field(default=500, description="Max vision API latency")
    adaptive_frame_rate: bool = Field(default=True, description="Enable adaptive frame rate")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = ["anthropic", "openai", "google"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @field_validator("frame_capture_mode")
    @classmethod
    def validate_capture_mode(cls, v: str) -> str:
        valid_modes = ["on_demand", "continuous"]
        if v not in valid_modes:
            raise ValueError(f"Frame capture mode must be one of {valid_modes}")
        return v

    @field_validator("commentary_style")
    @classmethod
    def validate_commentary_style(cls, v: str) -> str:
        valid_styles = ["concise", "detailed", "technical"]
        if v not in valid_styles:
            raise ValueError(f"Commentary style must be one of {valid_styles}")
        return v


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Daily.co settings
    daily_room_url: str = Field(..., validation_alias="DAILY_ROOM_URL")
    daily_token: str = Field(..., validation_alias="DAILY_TOKEN")

    # Deepgram settings
    deepgram_api_key: str = Field(..., validation_alias="DEEPGRAM_API_KEY")
    deepgram_model: str = Field(default="nova-2", validation_alias="DEEPGRAM_MODEL")
    deepgram_diarize: bool = Field(default=True, validation_alias="DEEPGRAM_DIARIZE")
    deepgram_language: str = Field(default="en-US", validation_alias="DEEPGRAM_LANGUAGE")
    deepgram_smart_format: bool = Field(default=True, validation_alias="DEEPGRAM_SMART_FORMAT")
    deepgram_punctuate: bool = Field(default=True, validation_alias="DEEPGRAM_PUNCTUATE")

    # LLM settings
    llm_provider: str = Field(default="anthropic", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="claude-3-5-sonnet-20241022", validation_alias="LLM_MODEL")
    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")

    # TTS settings
    tts_provider: str = Field(default="elevenlabs", validation_alias="TTS_PROVIDER")
    tts_voice_id: str = Field(..., validation_alias="TTS_VOICE_ID")
    elevenlabs_api_key: str = Field(..., validation_alias="ELEVENLABS_API_KEY")
    tts_model: str = Field(default="eleven_flash_v2_5", validation_alias="TTS_MODEL")
    tts_optimize_streaming_latency: int = Field(default=3, validation_alias="TTS_OPTIMIZE_STREAMING_LATENCY")
    tts_stability: float = Field(default=0.5, validation_alias="TTS_STABILITY")
    tts_similarity_boost: float = Field(default=0.75, validation_alias="TTS_SIMILARITY_BOOST")

    # Processor settings
    tension_threshold: float = Field(default=0.7, validation_alias="TENSION_THRESHOLD")
    cooldown_seconds: int = Field(default=30, validation_alias="COOLDOWN_SECONDS")
    buffer_size: int = Field(default=50, validation_alias="BUFFER_SIZE")

    # Vision settings
    vision_enabled: bool = Field(default=False, validation_alias="VISION_ENABLED")
    vision_provider: str = Field(default="anthropic", validation_alias="VISION_PROVIDER")
    vision_model: str = Field(default="claude-3-5-sonnet-20241022", validation_alias="VISION_MODEL")
    vision_api_key: Optional[str] = Field(default=None, validation_alias="VISION_API_KEY")
    vision_analysis_interval: float = Field(default=2.0, validation_alias="VISION_ANALYSIS_INTERVAL")
    vision_max_cost: float = Field(default=0.30, validation_alias="VISION_MAX_COST_PER_SESSION")
    vision_commentary_style: str = Field(default="concise", validation_alias="VISION_COMMENTARY_STYLE")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def daily(self) -> DailyConfig:
        """Get Daily.co configuration."""
        return DailyConfig(
            room_url=self.daily_room_url,
            token=self.daily_token
        )

    @property
    def deepgram(self) -> DeepgramConfig:
        """Get Deepgram configuration."""
        return DeepgramConfig(
            api_key=self.deepgram_api_key,
            model=self.deepgram_model,
            diarize=self.deepgram_diarize,
            language=self.deepgram_language,
            smart_format=self.deepgram_smart_format,
            punctuate=self.deepgram_punctuate
        )

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.anthropic_api_key
        )

    @property
    def tts(self) -> TTSConfig:
        """Get TTS configuration."""
        return TTSConfig(
            provider=self.tts_provider,
            voice_id=self.tts_voice_id,
            api_key=self.elevenlabs_api_key,
            model=self.tts_model,
            optimize_streaming_latency=self.tts_optimize_streaming_latency,
            stability=self.tts_stability,
            similarity_boost=self.tts_similarity_boost
        )

    @property
    def processor(self) -> ProcessorConfig:
        """Get processor configuration."""
        return ProcessorConfig(
            tension_threshold=self.tension_threshold,
            cooldown_seconds=self.cooldown_seconds,
            buffer_size=self.buffer_size
        )

    @property
    def vision(self) -> VisionConfig:
        """Get vision configuration."""
        return VisionConfig(
            enabled=self.vision_enabled,
            provider=self.vision_provider,
            model=self.vision_model,
            api_key=self.vision_api_key or self.anthropic_api_key,
            analysis_interval_seconds=self.vision_analysis_interval,
            max_analysis_cost_per_session=self.vision_max_cost,
            commentary_style=self.vision_commentary_style,
        )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create Settings singleton instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
