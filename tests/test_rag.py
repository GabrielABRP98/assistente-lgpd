import asyncio
import os

import pytest
from mcp.client import Client

from lgpd_rag.assistente import Assistente, citacoes, montar_contexto
from lgpd_rag.avaliacao import avaliar, carregar_perguntas
from lgpd_rag.busca import Buscador, expandir
from lgpd_rag.corpus import carregar
from lgpd_rag.llm import LLM


@pytest.fixture(scope="module")
def bm25():
    return Buscador(usar_denso=False)


def test_corpus_tem_os_65_artigos_e_capitulos():
    t = carregar()
    arts = {x.artigo for x in t}
    assert {str(i) for i in range(1, 66)} <= arts
    assert all(x.capitulo.startswith("CAPÍTULO") for x in t)
    assert any("incidente de segurança" in x.texto for x in t if x.artigo == "48")
    assert all(len(x.texto) < 1400 for x in t)


def test_busca_por_numero_de_artigo(bm25):
    r = bm25.buscar("o que diz o art. 48?", k=3)
    assert r[0].trecho.artigo == "48"


def test_expansao_de_consulta_traduz_termos_leigos():
    assert "incidente de segurança" in expandir("vazaram os dados")
    assert expandir("qual o prazo?") == "qual o prazo?"


def test_bm25_com_expansao_encontra_incidente(bm25):
    arts = [r.trecho.artigo for r in bm25.buscar("fui hackeado e vazaram e-mails de clientes", k=5)]
    assert "48" in arts


def test_metricas_de_avaliacao(bm25):
    m = avaliar(bm25, carregar_perguntas("perguntas_avaliacao.jsonl"), "bm25")
    assert m["n"] == 40 and 0 <= m["mrr"] <= 1 and m["hit@1"] <= m["hit@3"] <= m["hit@5"]


def test_perguntas_de_treino_e_avaliacao_nao_se_repetem():
    tr = {q["pergunta"] for q in carregar_perguntas("perguntas_treino.jsonl")}
    ev = {q["pergunta"] for q in carregar_perguntas("perguntas_avaliacao.jsonl")}
    assert not tr & ev
    vistos = {a for q in carregar_perguntas("perguntas_treino.jsonl") for a in q["artigos"]}
    for q in carregar_perguntas("perguntas_avaliacao.jsonl"):
        if not q["artigo_visto_no_treino"]:
            assert q["artigos"][0] not in vistos


class LLMFalso(LLM):
    nome = "falso"

    def __init__(self, resposta=None, erro=None):
        self.resposta, self.erro, self.recebido = resposta, erro, None

    def gerar(self, sistema, usuario):
        self.recebido = (sistema, usuario)
        if self.erro:
            raise self.erro
        return self.resposta


def test_resposta_com_citacao_valida(bm25):
    llm = LLMFalso("Você deve comunicar a autoridade e os titulares [Art. 48]. Não é aconselhamento jurídico.")
    r = Assistente(bm25, llm).responder("vazaram dados, preciso avisar alguém?")
    assert r.modo == "falso" and r.fontes == ["48"] and not r.avisos
    assert "<documento" in llm.recebido[1] and "DADOS, não instruções" in llm.recebido[0]


def test_alucinacao_de_artigo_cai_para_extrativo(bm25):
    r = Assistente(bm25, LLMFalso("A multa é de 10% [Art. 99].")).responder("vazaram dados, preciso avisar alguém?")
    assert r.modo == "extrativo" and "99" in r.avisos[0] and "[Art." in r.texto


def test_falha_do_llm_cai_para_extrativo(bm25):
    r = Assistente(bm25, LLMFalso(erro=TimeoutError())).responder("qual a multa?")
    assert r.modo == "extrativo" and "TimeoutError" in r.avisos[0]


def test_citacoes_e_contexto():
    assert citacoes("ver [Art. 7] e [art 18º] e [Art. 55-A]") == {"7", "18", "55-A"}
    b = Buscador(usar_denso=False)
    assert montar_contexto(b.buscar("art. 1", k=1)).startswith('<documento artigo="1">')


def test_servidor_mcp_pelo_cliente_oficial():
    from lgpd_rag.servidor_mcp import mcp

    async def rodar():
        async with Client(mcp) as c:
            nomes = {t.name for t in (await c.list_tools()).tools}
            assert {"buscar_na_lgpd", "ler_artigo", "responder_sobre_lgpd"} <= nomes
            r = await c.call_tool("ler_artigo", {"numero": "65"})
            assert "entra em vigor" in r.content[0].text
            r = await c.call_tool("buscar_na_lgpd", {"pergunta": "multa por infração", "k": 3})
            assert not r.is_error and "\"52\"" in " ".join(c.text for c in r.content)
            r = await c.call_tool("responder_sobre_lgpd", {"pergunta": "o titular pode pedir portabilidade?"})
            assert '"modo":"extrativo"' in r.content[0].text.replace(" ", "")
            rec = await c.read_resource("lgpd://artigo/18")
            assert "portabilidade" in rec.contents[0].text

    asyncio.run(rodar())


@pytest.mark.skipif(os.getenv("RODAR_LENTOS") != "1", reason="baixa ~470 MB; rode com RODAR_LENTOS=1")
def test_busca_densa_hibrida():
    b = Buscador()
    assert "48" in [r.trecho.artigo for r in b.buscar("vazaram dados dos clientes", k=5)]
