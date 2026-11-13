"""Recuperação híbrida: BM25 (palavras exatas, ex.: 'art. 18', 'multa') + embeddings (significado, ex.: 'vazaram' ≈ 'incidente
de segurança'), combinadas por Reciprocal Rank Fusion. Cada resultado traz o artigo para citação."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Trecho, carregar

MODELO_BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_AJUSTADO = Path(__file__).resolve().parents[1] / "modelos" / "embeddings-lgpd"
STOP = {"a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", "um", "uma", "os", "as", "que", "se", "ao", "aos", "à", "às", "ou", "é", "eu", "meu", "minha", "tenho", "preciso", "posso", "pode", "isso", "esse", "essa", "tipo", "pra"}


def tokens(t: str) -> list[str]:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return [w[:6] for w in re.findall(r"[a-z0-9]+", t) if w not in STOP]  # radical simples: 6 primeiras letras


# Glossário leigo -> termos da lei (vocabulário diferente é a principal causa de erro em RAG jurídico).
GLOSSARIO = {
    r"vaz|hacke|invad|ataque|ransomware": "incidente de segurança acesso não autorizado comunicação à autoridade nacional",
    r"multa|punic|penalidad": "sanções administrativas",
    r"apag|exclu|delet": "eliminação dos dados",
    r"autoriza": "consentimento",
    r"client|consumidor|usuario|usuário": "titular",
    r"terceiriz|fornecedor de sistema": "operador instruções do controlador",
    r"guard|armazen|manter os dados": "conservação término do tratamento",
    r"crian|menor de idade|aluno|adolescen": "criança adolescente consentimento dos pais",
    r"nuvem|exterior|outro pa[ií]s|estados unidos": "transferência internacional de dados",
    r"\bdpo\b|responsável por (proteção de )?dados": "encarregado",
    r"sa[uú]de|religi|biometr|racial|pol[ií]tic": "dado pessoal sensível",
    r"algoritmo|autom[aá]tic|intelig[eê]ncia artificial": "decisões tomadas unicamente com base em tratamento automatizado",
}


def expandir(pergunta: str) -> str:
    extras = [termos for padrao, termos in GLOSSARIO.items() if re.search(padrao, pergunta.lower())]
    return pergunta + (" | " + " ".join(extras) if extras else "")


@dataclass
class Resultado:
    trecho: Trecho
    pontuacao: float


class Buscador:
    def __init__(self, trechos: list[Trecho] | None = None, modelo: str | Path | None = None, usar_denso: bool = True):
        self.trechos = trechos or carregar()
        self.bm25 = BM25Okapi([tokens(t.texto) for t in self.trechos])
        self.encoder = None
        if usar_denso:
            from sentence_transformers import SentenceTransformer
            nome = modelo or (MODELO_AJUSTADO if MODELO_AJUSTADO.exists() else MODELO_BASE)
            self.encoder = SentenceTransformer(str(nome), device="cpu")
            self.matriz = self.encoder.encode([t.texto for t in self.trechos], normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    def _rank_bm25(self, q: str) -> np.ndarray:
        return np.argsort(-self.bm25.get_scores(tokens(q)))

    def _rank_denso(self, q: str) -> np.ndarray:
        v = self.encoder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.argsort(-(self.matriz @ v))

    def artigo(self, numero: str) -> list[Trecho]:
        return [t for t in self.trechos if t.artigo == str(numero)]

    def buscar(self, pergunta: str, k: int = 5, modo: str = "hibrido", k_rrf: int = 60, expandir_consulta: bool = True) -> list[Resultado]:
        citado = re.search(r"\bart(?:igo|\.)?\s*(\d+(?:-[a-z])?)", pergunta.lower())
        if citado and self.artigo(citado.group(1).upper()):  # "o que diz o art. 48?" -> vai direto ao artigo
            fixos = self.artigo(citado.group(1).upper())
            resto = [r for r in self._buscar(expandir(pergunta) if expandir_consulta else pergunta, k, modo, k_rrf) if r.trecho not in fixos]
            return ([Resultado(t, 1.0) for t in fixos] + resto)[:max(k, len(fixos))]
        return self._buscar(expandir(pergunta) if expandir_consulta else pergunta, k, modo, k_rrf)

    def _buscar(self, pergunta: str, k: int, modo: str, k_rrf: int) -> list[Resultado]:
        if modo == "bm25" or self.encoder is None:
            ordem = self._rank_bm25(pergunta)
            return [Resultado(self.trechos[i], float(len(ordem) - r)) for r, i in enumerate(ordem[:k])]
        if modo == "denso":
            ordem = self._rank_denso(pergunta)
            return [Resultado(self.trechos[i], float(len(ordem) - r)) for r, i in enumerate(ordem[:k])]
        placar: dict[int, float] = {}
        for ordem in (self._rank_bm25(pergunta), self._rank_denso(pergunta)):
            for pos, i in enumerate(ordem):
                placar[i] = placar.get(i, 0.0) + 1.0 / (k_rrf + pos + 1)
        melhores = sorted(placar, key=placar.get, reverse=True)[:k]
        return [Resultado(self.trechos[i], round(placar[i], 5)) for i in melhores]
