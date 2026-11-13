"""Converte o texto da Lei 13.709/2018 (LGPD) em artigos e trechos indexáveis, preservando capítulo e número do artigo."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
HTML = RAIZ / "dados" / "bruto" / "lei_13709_2018_original_camara.html"
CORPUS = RAIZ / "dados" / "lgpd_trechos.jsonl"
MAX_CHARS = 1100


@dataclass
class Trecho:
    id: str
    artigo: str          # "7", "55-A"...
    capitulo: str
    texto: str


def texto_da_lei(html: str) -> str:
    from bs4 import BeautifulSoup
    t = BeautifulSoup(html, "html.parser").get_text("\n")
    t = t.replace("\xa0", " ")
    ini = t.find("Art. 1º")
    fim = t.find("Brasília,", ini)
    return re.sub(r"[ \t]+", " ", t[ini:fim if fim > 0 else None])


def artigos(texto: str) -> list[tuple[str, str, str]]:
    """Devolve [(numero, capitulo, texto)] — divide em 'Art. Nº' no início de linha."""
    capitulo, saida = "CAPÍTULO I DISPOSIÇÕES PRELIMINARES", []
    partes = re.split(r"\n\s*(?=Art\. \d+(?:º|°|-[A-Z])?\.?\s)", "\n" + texto)
    for p in partes:
        p = p.strip()
        caps = re.findall(r"(CAPÍTULO [IVXL]+\s*\n\s*[^\n]+)", p)
        m = re.match(r"Art\. (\d+(?:-[A-Z])?)", p)
        if m:
            corpo = re.split(r"\n\s*CAPÍTULO [IVXL]+", p)[0]
            corpo = re.sub(r"\s*\n\s*", "\n", corpo).strip()
            corpo = re.sub(r"\n(?=[a-zà-ú,;.:)\"])", " ", corpo)
            corpo = re.sub(r"\b(do|da|de|no|na|o|a)\n", r"\1 ", corpo)  # quebras do HTML no meio da frase
            saida.append((m.group(1), capitulo, corpo))
        if caps:
            capitulo = re.sub(r"\s+", " ", caps[-1]).strip()
    return saida


def dividir(numero: str, texto: str) -> list[str]:
    """Artigos longos (ex.: art. 5º com definições) viram vários trechos, sempre com o cabeçalho do artigo."""
    if len(texto) <= MAX_CHARS:
        return [texto]
    linhas, blocos, atual = texto.split("\n"), [], ""
    cab = linhas[0]
    for ln in linhas[1:]:
        if len(atual) + len(ln) > MAX_CHARS and atual:
            blocos.append(atual)
            atual = ""
        atual += ("\n" if atual else "") + ln
    blocos.append(atual)
    return [cab] + [f"Art. {numero} (continuação)\n{b}" for b in blocos] if len(cab) > 200 else \
           [f"{cab}\n{blocos[0]}"] + [f"Art. {numero} (continuação)\n{b}" for b in blocos[1:]]


def construir(html_path: Path = HTML, destino: Path = CORPUS) -> list[Trecho]:
    trechos = []
    for numero, cap, texto in artigos(texto_da_lei(html_path.read_text(encoding="utf-8", errors="ignore"))):
        for i, parte in enumerate(dividir(numero, texto)):
            trechos.append(Trecho(f"art{numero}-{i}", numero, cap, parte))
    destino.write_text("\n".join(json.dumps(asdict(t), ensure_ascii=False) for t in trechos), encoding="utf-8")
    return trechos


def carregar(caminho: Path = CORPUS) -> list[Trecho]:
    return [Trecho(**json.loads(l)) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]
