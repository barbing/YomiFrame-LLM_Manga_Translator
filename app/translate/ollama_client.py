# -*- coding: utf-8 -*-
"""Ollama API client."""
from __future__ import annotations
import os
import requests
from typing import Optional
import json

import logging

logger = logging.getLogger(__name__)


class DeepSeekApiKeyError(RuntimeError):
    """Raised when the DeepSeek API key is missing or unreadable."""


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _parse_api_key_text(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    if clean.startswith("{"):
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("DEEPSEEK_API_KEY", "API_KEY", "api_key", "key"):
                value = str(parsed.get(key) or "").strip()
                if value:
                    return _strip_wrapping_quotes(value)

    for line in clean.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"DEEPSEEK_API_KEY", "API_KEY"}:
            return _strip_wrapping_quotes(value)
    return _strip_wrapping_quotes(clean)


def _candidate_api_key_paths(api_key_path: str) -> list[str]:
    path = str(api_key_path or "").strip()
    if not path:
        return []
    paths = [path]
    if os.path.normpath(path) == os.path.normpath("api/API_KEY"):
        paths.append("api/API_KEY.json")
    return paths


def _load_deepseek_api_key(api_key_path: str = "api/API_KEY") -> str:
    env_key = _parse_api_key_text(os.environ.get("DEEPSEEK_API_KEY", ""))
    if env_key:
        return env_key

    for path in _candidate_api_key_paths(api_key_path):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                file_key = _parse_api_key_text(handle.read())
        except OSError as exc:
            raise DeepSeekApiKeyError(f"DeepSeek API key file could not be read: {path}") from exc
        if file_key:
            return file_key
    raise DeepSeekApiKeyError("DeepSeek API key not found. Set DEEPSEEK_API_KEY or create api/API_KEY.")


def _parse_glossary_response(text: str) -> dict[str, str]:
    clean_response = str(text or "").strip()
    if not clean_response:
        return {}
    if "```json" in clean_response:
        clean_response = clean_response.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in clean_response:
        clean_response = clean_response.split("```", 1)[1].split("```", 1)[0]

    try:
        parsed = json.loads(clean_response)
        if isinstance(parsed, dict):
            return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    start = clean_response.find("{")
    end = clean_response.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = clean_response[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {str(k).strip(): str(v).strip() for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    merged = {}
    for line in clean_response.splitlines():
        line = line.strip().rstrip(",")
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                merged[str(key).strip()] = str(value).strip()
    return merged


class DeepSeekClient:
    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        api_key_path: str = "api/API_KEY",
        model_name: str = "deepseek-v4-flash",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key_path = api_key_path
        self.model_name = model_name

    @staticmethod
    def has_configured_key(api_key_path: str = "api/API_KEY") -> bool:
        try:
            return bool(_load_deepseek_api_key(api_key_path).strip())
        except DeepSeekApiKeyError:
            return False

    def _headers(self) -> dict[str, str]:
        api_key = _load_deepseek_api_key(self._api_key_path)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def is_available(self, timeout: int = 5) -> bool:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 1,
        }
        try:
            response = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
            return response.status_code == 200
        except (DeepSeekApiKeyError, requests.RequestException) as exc:
            logger.debug(f"DeepSeek unavailable: {exc}")
            return False

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        url = f"{self._base_url}/chat/completions"
        model_to_use = str(model or "").strip() or self.model_name
        options = dict(options or {})
        payload = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": options.get("temperature", 0.2),
        }
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if "num_predict" in options:
            payload["max_tokens"] = options["num_predict"]
        elif "max_tokens" in options:
            payload["max_tokens"] = options["max_tokens"]

        response = requests.post(url, headers=self._headers(), json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()

    def translate_glossary(self, terms: list[str], source_lang: str, target_lang: str) -> dict[str, str]:
        if not terms:
            return {}

        results = {}
        batch_size = 20
        is_zh = target_lang in ["Simplified Chinese", "Traditional Chinese", "zh", "zh-CN", "zh-TW"]

        for i in range(0, len(terms), batch_size):
            chunk = terms[i:i + batch_size]

            if is_zh:
                prompt_text = (
                    f"将以下日文人名列表翻译成中文。\n"
                    f"请输出严格的JSON格式，格式为 {{\"日文原名\": \"中文译名\"}}。\n"
                    f"不要添加任何解释或Markdown标记。\n\n"
                    f"待翻译列表：\n" + "\n".join([f"- {t}" for t in chunk])
                )
            else:
                prompt_text = (
                    f"Translate the following Japanese names to {target_lang}.\n"
                    f"Output strictly valid JSON format: {{\"Source Name\": \"Translated Name\"}}.\n"
                    f"No explanations or markdown.\n\n"
                    f"List:\n" + "\n".join([f"- {t}" for t in chunk])
                )

            try:
                response = self.generate(
                    model=self.model_name,
                    prompt=prompt_text,
                    options={"num_predict": 1024, "temperature": 0.1}
                )

                chunk_map = _parse_glossary_response(response)
                if chunk_map:
                    results.update(chunk_map)
                else:
                    logger.warning(f"Failed to parse glossary JSON (DeepSeek): {str(response)[:50]}...")
            except Exception as e:
                logger.error(f"Error translating glossary chunk (DeepSeek): {e}")

        final_map = {}
        for term in terms:
            if term in results:
                final_map[term] = str(results[term]).strip()

        return final_map


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    _availability_cache = {"timestamp": 0.0, "status": False}

    def is_available(self, timeout: int = 5) -> bool:
        import time
        now = time.time()
        
        # Check cache (limit checks to once every 3 seconds globally)
        if now - OllamaClient._availability_cache["timestamp"] < 3.0:
            return OllamaClient._availability_cache["status"]

        url = f"{self._base_url}/api/tags"
        available = False
        try:
            response = requests.get(url, timeout=timeout)
            available = response.status_code == 200
            if not available:
                logger.debug(f"Ollama check failed. Status: {response.status_code}")
        except requests.RequestException as e:
            # Downgrade to debug to prevent spam
            logger.debug(f"Ollama unavailable: {e}")
            available = False
            
        # Update cache
        OllamaClient._availability_cache = {"timestamp": now, "status": available}
        return available

    def generate(self, model: str, prompt: str, timeout: int = 600, options: Optional[dict] = None) -> str:
        url = f"{self._base_url}/api/generate"
        default_options = {"temperature": 0.2}
        if options:
            default_options.update(options)
        payload = {"model": model, "prompt": prompt, "stream": False, "options": default_options}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Ollama generation success for model {model}")
        return str(data.get("response", "")).strip()
    def translate_glossary(self, terms: list[str], source_lang: str, target_lang: str) -> dict[str, str]:
        """Translate a batch of terms using the LLM. Requires self.model_name to be set."""
        if not terms:
            return {}
        
        # Default model if not set (fallback)
        model = getattr(self, "model_name", "sakura")
        
        results = {}
        batch_size = 20
        is_zh = target_lang in ["Simplified Chinese", "Traditional Chinese", "zh", "zh-CN", "zh-TW"]
        
        for i in range(0, len(terms), batch_size):
            chunk = terms[i:i+batch_size]
            
            if is_zh:
                prompt_text = (
                    f"将以下日文人名列表翻译成中文。\n"
                    f"请输出严格的JSON格式，格式为 {{\"日文原名\": \"中文译名\"}}。\n"
                    f"不要添加任何解释或Markdown标记。\n\n"
                    f"待翻译列表：\n" + "\n".join([f"- {t}" for t in chunk])
                )
            else:
                prompt_text = (
                    f"Translate the following Japanese names to {target_lang}.\n"
                    f"Output strictly valid JSON format: {{\"Source Name\": \"Translated Name\"}}.\n"
                    f"No explanations or markdown.\n\n"
                    f"List:\n" + "\n".join([f"- {t}" for t in chunk])
                )

            try:
                response = self.generate(
                    model=model,
                    prompt=prompt_text,
                    options={"num_predict": 1024, "temperature": 0.1}
                )

                chunk_map = _parse_glossary_response(response)
                if chunk_map:
                    results.update(chunk_map)
                else:
                    logger.warning(f"Failed to parse glossary JSON (Ollama): {str(response)[:50]}...")
            except Exception as e:
                logger.error(f"Error translating glossary chunk (Ollama): {e}")
                
        # Clean results
        final_map = {}
        for term in terms:
            if term in results:
                final_map[term] = str(results[term]).strip()
        
        return final_map
