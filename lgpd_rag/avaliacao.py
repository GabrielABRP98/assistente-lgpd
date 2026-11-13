"""Métricas de recuperação: Hit@k (algum artigo correto entre os k primeiros) e MRR (posição do primeiro acerto)."""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def carregar_perguntas(nome: str) -> list[dict]:
    return [json.loads(l) for l in (RAIZ / "avaliacao" / nome).read_text(encoding="utf-8").splitlines() if l.strip()]


def avaliar(buscador, perguntas: list[dict], modo: str, k: int = 5, expandir: bool = True) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    for q in perguntas:
        arts = [r.trecho.artigo for r in buscador.buscar(q["pergunta"], k=k, modo=modo, expandir_consulta=expandir)]
        pos = next((i for i, a in enumerate(arts) if a in q["artigos"]), None)
        if pos is not None:
            rr += 1 / (pos + 1)
            for kk in hits:
                hits[kk] += pos < kk
    n = len(perguntas)
    return {"n": n, "hit@1": round(hits[1] / n, 3), "hit@3": round(hits[3] / n, 3), "hit@5": round(hits[5] / n, 3), "mrr": round(rr / n, 3)}
