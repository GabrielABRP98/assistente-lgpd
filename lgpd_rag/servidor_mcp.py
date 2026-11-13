"""Servidor MCP (SDK oficial, v2): expõe a LGPD como ferramentas para qualquer cliente MCP (Claude Desktop, IDEs, agentes).
Rodar:  python -m lgpd_rag.servidor_mcp     (transporte stdio)"""
from __future__ import annotations

import os
from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from .assistente import Assistente
from .busca import Buscador
from .llm import do_ambiente

mcp = MCPServer("assistente-lgpd", title="Assistente LGPD", version="1.0.0",
                instructions="Consulta ao texto original da LGPD (Lei 13.709/2018) com citações por artigo.")


@lru_cache(maxsize=1)
def buscador() -> Buscador:
    return Buscador(usar_denso=os.getenv("LGPD_SO_BM25") != "1")


@mcp.tool()
def buscar_na_lgpd(pergunta: str, k: int = 5) -> list[dict]:
    """Busca os trechos da LGPD mais relevantes para uma pergunta. Retorna artigo, capítulo e texto."""
    k = max(1, min(int(k), 10))
    return [{"artigo": r.trecho.artigo, "capitulo": r.trecho.capitulo, "texto": r.trecho.texto, "pontuacao": r.pontuacao}
            for r in buscador().buscar(pergunta, k=k)]


@mcp.tool()
def ler_artigo(numero: str) -> str:
    """Retorna o texto integral de um artigo da LGPD (ex.: '18', '48')."""
    partes = buscador().artigo(str(numero).strip().upper())
    return "\n".join(t.texto for t in partes) if partes else f"Artigo {numero} não encontrado na LGPD."


@mcp.tool()
def responder_sobre_lgpd(pergunta: str) -> dict:
    """Responde em linguagem simples com citações [Art. N], usando o LLM configurado ou, sem LLM, os trechos da lei."""
    r = Assistente(buscador(), do_ambiente()).responder(pergunta)
    return {"resposta": r.texto, "fontes": r.fontes, "modo": r.modo, "avisos": r.avisos}


@mcp.resource("lgpd://artigo/{numero}")
def recurso_artigo(numero: str) -> str:
    """Texto de um artigo da LGPD como recurso MCP."""
    return ler_artigo(numero)


if __name__ == "__main__":
    mcp.run("stdio")
