import requests
import base64
import json
import logging
import os
from typing import Optional, Union
import io

logger = logging.getLogger(__name__)

class SarvamClient:
    def __init__(self):
        self.api_key = os.environ.get('SARVAM_API_KEY')
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is required")
        
        self.base_url = "https://api.sarvam.ai"
        self.headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # STT endpoint
        self.stt_url = f"{self.base_url}/speech-to-text"
        # TTS endpoint  
        self.tts_url = f"{self.base_url}/text-to-speech"
        
    def speech_to_text(self, audio_data: bytes, final: bool = False) -> Optional[str]:
        """
        Convert audio to text using Sarvam STT API
        
        Args:
            audio_data: Raw audio bytes (PCM format)
            final: Whether this is the final chunk
            
        Returns:
            Transcribed text or None if error
        """
        try:
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "model": "saaras:v1",  # Sarvam's Malayalam STT model
                "audio": audio_base64,
                "language_code": "ml",  # Malayalam
                "format": "pcm",
                "sample_rate": 8000,
                "encoding": "linear16",
                "with_timestamps": False,
                "enable_speaker_diarization": False
            }
            
            # Make API request
            response = requests.post(
                self.stt_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('transcript', '').strip()
                
                if transcript:
                    logger.info(f"STT Result: {transcript}")
                    return transcript
                else:
                    logger.debug("Empty transcript received")
                    return None
                    
            else:
                logger.error(f"STT API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("STT API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"STT API request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"STT processing error: {str(e)}")
            return None
    
    def text_to_speech(self, text: str, voice: str = "meera") -> Optional[bytes]:
        """
        Convert text to speech using Sarvam TTS API
        
        Args:
            text: Malayalam text to convert
            voice: Voice model to use
            
        Returns:
            Audio bytes or None if error
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided to TTS")
                return None
            
            # Prepare request payload
            payload = {
                "inputs": [text.strip()],
                "target_language_code": "ml",  # Malayalam
                "speaker": voice,  # Available: meera, etc.
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 8000,  # Match Exotel's requirement
                "enable_preprocessing": True,
                "model": "bulbul:v1"  # Sarvam's TTS model
            }
            
            # Make API request
            response = requests.post(
                self.tts_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract audio data
                if 'audios' in result and len(result['audios']) > 0:
                    audio_base64 = result['audios'][0]
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    logger.info(f"TTS successful, audio length: {len(audio_bytes)} bytes")
                    return audio_bytes
                else:
                    logger.error("No audio data in TTS response")
                    return None
                    
            else:
                logger.error(f"TTS API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("TTS API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS API request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"TTS processing error: {str(e)}")
            return None
    
    def get_available_voices(self) -> list:
        """
        Get list of available Malayalam voices
        
        Returns:
            List of available voice names
        """
        try:
            # This endpoint may not exist in Sarvam API
            # Return default voices for now
            return ["meera", "ravi", "arjun"]
            
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return ["meera"]  # fallback
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful
        """
        try:
            # Test with a simple TTS request
            test_text = "പരീക്ഷണം"  # "Test" in Malayalam
            result = self.text_to_speech(test_text)
            return result is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False