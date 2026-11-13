"""Provedores de LLM intercambiáveis. Nenhum é obrigatório: sem LLM o assistente responde de forma extrativa.
- ollama: modelo local e gratuito (ex.: qwen2.5:3b) em http://localhost:11434
- anthropic: API paga; só é usada se ANTHROPIC_API_KEY estiver definida (custo por token, ver README)."""
from __future__ import annotations

import json
import os
import urllib.request


class LLM:
    nome = "base"

    def gerar(self, sistema: str, usuario: str) -> str:
        raise NotImplementedError


class Ollama(LLM):
    nome = "ollama"

    def __init__(self, modelo: str | None = None, url: str | None = None):
        self.modelo = modelo or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")

    def gerar(self, sistema: str, usuario: str) -> str:
        corpo = {"model": self.modelo, "stream": False, "options": {"temperature": 0.1},
                 "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]}
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(corpo).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]


class Anthropic(LLM):
    nome = "anthropic"

    def __init__(self, modelo: str | None = None):
        self.chave = os.environ["ANTHROPIC_API_KEY"]
        self.modelo = modelo or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

    def gerar(self, sistema: str, usuario: str) -> str:
        corpo = {"model": self.modelo, "max_tokens": 800, "temperature": 0.1, "system": sistema,
                 "messages": [{"role": "user", "content": usuario}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(corpo).encode(),
                                     {"content-type": "application/json", "x-api-key": self.chave, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return "".join(b.get("text", "") for b in json.loads(r.read())["content"])


def do_ambiente() -> LLM | None:
    escolha = os.getenv("LLM_PROVEDOR", "").lower()
    if escolha == "ollama":
        return Ollama()
    if escolha == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return Anthropic()
    return None
