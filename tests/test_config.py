from __future__ import annotations

from langchain_aidefense.config import AIDefenseSettings


def test_from_env_normalizes_endpoints_and_fallbacks():
    settings = AIDefenseSettings.from_env(
        {
            "AI_DEFENSE_API_MODE_LLM_ENDPOINT": "https://example.com/api/v1/inspect/chat",
            "AI_DEFENSE_API_MODE_LLM_API_KEY": "llm-key",
            "AGENTSEC_API_MODE_LLM": "monitor",
        }
    )

    assert settings.llm.endpoint == "https://example.com"
    assert settings.tools.endpoint == "https://example.com"
    assert settings.tools.api_key == "llm-key"
    assert settings.tools.mode == "monitor"


def test_from_env_parses_optional_tuning_values():
    settings = AIDefenseSettings.from_env(
        {
            "AI_DEFENSE_API_MODE_LLM_ENDPOINT": "https://example.com/api",
            "AI_DEFENSE_API_MODE_LLM_API_KEY": "llm-key",
            "AGENTSEC_LLM_RULES": '["Prompt Injection", "PII"]',
            "AGENTSEC_LLM_ENTITY_TYPES": "EMAIL,PHONE_NUMBER",
            "AGENTSEC_TIMEOUT": "12",
            "AGENTSEC_RETRY_TOTAL": "4",
            "AGENTSEC_RETRY_BACKOFF_FACTOR": "1.5",
            "AGENTSEC_RETRY_STATUS_FORCELIST": "429,500,503",
            "AGENTSEC_VIOLATION_BEHAVIOR": "replace",
            "AGENTSEC_VIOLATION_MESSAGE": "blocked: {phase}",
        }
    )

    assert settings.timeout_ms == 12000
    assert settings.retry_total == 4
    assert settings.retry_backoff == 1.5
    assert settings.retry_status_codes == (429, 500, 503)
    assert settings.llm_default_rules == ("Prompt Injection", "PII")
    assert settings.llm_entity_types == ("EMAIL", "PHONE_NUMBER")
    assert settings.violation_behavior == "replace"
    assert settings.violation_message == "blocked: {phase}"
