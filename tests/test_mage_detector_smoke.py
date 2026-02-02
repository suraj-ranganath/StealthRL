import os
import asyncio
import pytest

from stealthrl.tinker.detectors import MageDetector, DetectorCache


@pytest.mark.slow
def test_mage_detector_smoke():
    if not os.getenv("RUN_MAGE_SMOKE"):
        pytest.skip("Set RUN_MAGE_SMOKE=1 to run MAGE smoke test")

    cache = DetectorCache(":memory:")
    detector = MageDetector(cache=cache)

    human_text = (
        "The quick brown fox jumps over the lazy dog. This sentence is used "
        "to test typing and contains every letter of the English alphabet."
    )
    ai_text = (
        "As an AI language model, I can provide information on a wide range of topics "
        "including history, science, and technology in a helpful manner."
    )

    async def _run():
        scores = await detector.detect_batch([human_text, ai_text], batch_size=2)
        return scores

    p_machine_human, p_machine_ai = asyncio.run(_run())
    p_human_human = 1.0 - p_machine_human
    p_human_ai = 1.0 - p_machine_ai

    assert p_human_human > 0.5
    assert p_human_ai < 0.5
