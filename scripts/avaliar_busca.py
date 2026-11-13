"""Compara BM25, denso e híbrido (modelo base ou ajustado) nas perguntas de avaliação.
Uso: python scripts/avaliar_busca.py [--modelo caminho_ou_nome] [--saida docs/x.json]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgpd_rag.avaliacao import avaliar, carregar_perguntas
from lgpd_rag.busca import MODELO_BASE, Buscador

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default=MODELO_BASE)
ap.add_argument("--saida")
a = ap.parse_args()
b = Buscador(modelo=a.modelo)
pergs = carregar_perguntas("perguntas_avaliacao.jsonl")
grupos = {"todas": pergs, "artigos_vistos_no_treino": [p for p in pergs if p["artigo_visto_no_treino"]],
          "artigos_ineditos": [p for p in pergs if not p["artigo_visto_no_treino"]]}
res = {}
for exp in (False, True):
    for modo in ("bm25", "denso", "hibrido"):
        res[f"{modo}{'+expansao' if exp else ''}"] = {g: avaliar(b, ps, modo, expandir=exp) for g, ps in grupos.items()}
print(f"modelo: {a.modelo}")
for modo, r in res.items():
    for g, m in r.items():
        print(f"  {modo:18s} {g:26s} n={m['n']:2d} hit@1={m['hit@1']:.2f} hit@3={m['hit@3']:.2f} hit@5={m['hit@5']:.2f} mrr={m['mrr']:.2f}")
if a.saida:
    Path(a.saida).write_text(json.dumps({"modelo": str(a.modelo), "resultados": res}, ensure_ascii=False, indent=2), encoding="utf-8")
