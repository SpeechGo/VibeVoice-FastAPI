# ABOUTME: Dependency injection functions for FastAPI
# ABOUTME: Provides get_voice_service function for injecting VoiceService singleton
from api.core.voice_service import VoiceService

def get_voice_service() -> VoiceService:
    """Dependency injection function for VoiceService"""
    return VoiceService.instance()