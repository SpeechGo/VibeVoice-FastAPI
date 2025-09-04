# ABOUTME: This file defines custom exception classes for the VoiceService API.
# ABOUTME: These exceptions map to specific HTTP status codes as defined in the Public Interface contract.


class ServiceBusyError(Exception):
    """Raised when the service is at capacity and cannot handle more requests.
    Maps to HTTP 429 Too Many Requests.
    """
    pass


class ModelNotReadyError(Exception):
    """Raised when the model is not loaded or ready for inference.
    Maps to HTTP 503 Service Unavailable.
    """
    pass


class GenerationTimeoutError(Exception):
    """Raised when text-to-speech generation times out.
    Maps to HTTP 408 Request Timeout.
    """
    pass


class InvalidVoiceError(Exception):
    """Raised when a requested voice ID is not found or invalid.
    Maps to HTTP 422 Unprocessable Entity.
    """
    pass