import logging
import subprocess
import tempfile
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Generates Text-to-Speech audio from document insights.
    Uses native OS voice synthesis for speed and zero dependencies locally.
    """
    @staticmethod
    def synthesize(text: str) -> str:
        """
        Converts text to an audio file and returns the file path.
        """
        if not text:
            return ""

        temp_dir = Path(tempfile.mkdtemp(prefix="cloudinsight_tts_"))

        # Sanitize text
        clean_text = " ".join(text.replace('"', "").split())
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000]

        logger.info("Initializing fast local TTS synthesis...")

        try:
            if sys.platform == "darwin":
                # macOS native say command
                out_path = temp_dir / "readout.aiff"
                subprocess.run(["say", "-o", str(out_path), clean_text], check=True, capture_output=True)
            elif sys.platform == "linux":
                # Fallback to espeak
                out_path = temp_dir / "readout.wav"
                subprocess.run(["espeak", "-w", str(out_path), clean_text], check=True, capture_output=True)
            else:
                logger.warning("TTS currently simulated on this OS.")
                return ""
        except Exception as e:
            logger.error(f"TTS Synthesis Failed: {e}")
            return ""

        logger.info(f"Audio payload created successfully at {out_path}.")
        return str(out_path)
