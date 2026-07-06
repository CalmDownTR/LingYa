"""SettingsService — mind/personality settings get/update/reset.

Extracted from MessageRouter (v0.9.5 router.py split).
"""

from __future__ import annotations

from typing import Any


class SettingsService:
    """Handles settings get/update/reset for OCEAN, tone, identity."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def handle_settings(self, payload: dict) -> dict:
        """Handle settings get/update_ocean/update_identity/update_tone/reset."""
        from lingya.mind.engine import TONE_PRESETS

        action = payload.get("action", "get")
        engine = self._engine

        if action == "get":
            c = engine.config
            return {
                "type": "settings_response",
                "payload": {
                    "ocean": {
                        "openness": c.ocean.openness,
                        "conscientiousness": c.ocean.conscientiousness,
                        "extraversion": c.ocean.extraversion,
                        "agreeableness": c.ocean.agreeableness,
                        "neuroticism": c.ocean.neuroticism,
                    },
                    "tone": {
                        "warmth": c.tone_matrix.warmth,
                        "formality": c.tone_matrix.formality,
                        "humor": c.tone_matrix.humor,
                    },
                    "identity": {
                        "identity": c.identity.identity,
                        "core_belief": c.identity.core_belief,
                    },
                    "available_presets": list(TONE_PRESETS.keys()),
                },
            }

        if action == "update_ocean":
            ocean = payload.get("ocean", {})
            key_map = {
                "O": "openness", "C": "conscientiousness",
                "E": "extraversion", "A": "agreeableness", "N": "neuroticism",
            }
            mapped: dict[str, float] = {}
            for k, v in ocean.items():
                full_key = key_map.get(k, k)
                val = float(v)
                if not (0.0 <= val <= 1.0):
                    return {
                        "type": "error",
                        "payload": {"message": f"{k}={val} 超出 0-1 范围"},
                    }
                mapped[full_key] = val
            await engine.reload_config({"ocean": mapped})
            full_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]
            new_ocean = {k: getattr(engine.config.ocean, k) for k in full_names}
            return {
                "type": "settings_response",
                "payload": {"ok": True, "ocean": new_ocean},
            }

        if action == "update_identity":
            identity_data = payload.get("identity", {})
            update: dict = {}
            if "identity" in identity_data:
                update["identity"] = identity_data["identity"]
            if "core_belief" in identity_data:
                update["core_belief"] = identity_data["core_belief"]
            if update:
                await engine.reload_config({"identity": update})
            return {"type": "settings_response", "payload": {"ok": True}}

        if action == "update_tone":
            preset = payload.get("preset", "")
            if preset not in TONE_PRESETS:
                valid = list(TONE_PRESETS.keys())
                return {
                    "type": "error",
                    "payload": {"message": f"Unknown preset: {preset}. Valid: {valid}"},
                }
            await engine.reload_config({"tone_preset": preset})
            return {
                "type": "settings_response",
                "payload": {"ok": True, "tone_preset": preset},
            }

        if action == "reset":
            await engine.reload_config({"reset": True})
            c = engine.config
            full_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]
            return {
                "type": "settings_response",
                "payload": {
                    "ok": True,
                    "ocean": {k: getattr(c.ocean, k) for k in full_names},
                },
            }

        return {
            "type": "error",
            "payload": {"message": f"Unknown settings action: {action}"},
        }
