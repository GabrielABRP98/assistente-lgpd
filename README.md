# ⚖️ Assistente LGPD — RAG com citações, fine-tuning de embeddings e servidor MCP

![CI](https://github.com/GabrielABRP98/assistente-lgpd/actions/workflows/ci.yml/badge.svg)

Assistente que responde dúvidas de pequenas empresas sobre a **LGPD (Lei 13.709/2018)** com base no texto da lei, sempre **citando o artigo**. O projeto reúne quatro técnicas de IA aplicada:

| Técnica | O que foi feito |
|---|---|
| **RAG** | busca híbrida (BM25 + embeddings, com fusão RRF) e expansão de consulta com um glossário de termos leigos |
| **Fine-tuning** | ajuste do modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` com pares pergunta → artigo, medido antes e depois |
| **LLM** | geração com Ollama (local, gratuito) ou API da Anthropic; verificação das citações e resposta extrativa quando não há LLM |
| **MCP** | servidor com o SDK oficial (v2) que expõe a lei como ferramentas e recursos para Claude Desktop, IDEs e agentes |

```
pergunta ─► expansão de consulta ─► BM25 ─┐
                                          ├─► RRF ─► top-5 trechos ─► LLM (só pode usar os trechos) ─► checa [Art. N]
                     embeddings ajustados ─┘                                    │ citou artigo não recuperado / falhou
                                                                                └──► resposta extrativa com citações
```

## Resultados (40 perguntas de avaliação, escritas em linguagem coloquial)
| Configuração | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0,38 | 0,55 | 0,57 | 0,46 |
| Embeddings base | 0,53 | 0,70 | 0,82 | 0,62 |
| Embeddings **ajustados** | 0,62 | 0,72 | 0,80 | 0,69 |
| Híbrido + expansão (modelo base) | 0,68 | 0,85 | 0,88 | 0,75 |
| **Híbrido + expansão (modelo ajustado)**, padrão do projeto | **0,72** | 0,82 | 0,88 | **0,78** |
| Embeddings ajustados + expansão | 0,70 | 0,82 | **0,93** | **0,78** |

Hit@k: algum artigo correto aparece entre os k primeiros. MRR: média de 1/posição do primeiro acerto. Detalhes por grupo em `docs/metricas_*.json`.

**Como ler esses números com honestidade:**
- **Separação treino/avaliação:** as 56 perguntas de treino e as 40 de avaliação são diferentes (há um teste automatizado que garante isso). Além disso, 16 perguntas de avaliação tratam de artigos que **nunca aparecem no treino**.
- **Efeito do fine-tuning:** sozinho, ele melhorou o Hit@1 (0,53 → 0,62) e o MRR (0,62 → 0,69). Nos artigos inéditos o ganho foi menor, e o Hit@5 caiu de 0,81 para 0,75 (uma pergunta).
- **Viés do glossário:** foi escrito depois de ver erros como "vazaram" ≠ "incidente de segurança" nas perguntas de avaliação, então o ganho dele está otimista.
- **Tamanho da amostra:** com 40 perguntas, cada pergunta vale 2,5 pontos percentuais.
- Um próximo passo seria criar um conjunto de avaliação maior, escrito por outra pessoa.

## Como executar
```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/ajustar_embeddings.py              # fine-tuning (~1–2 min em CPU, < 1 GB de RAM)
python scripts/avaliar_busca.py --modelo modelos/embeddings-lgpd
python perguntar.py "fui hackeado e vazaram e-mails de clientes, preciso avisar alguém?"
pytest -q
```
Sem `modelos/embeddings-lgpd`, o projeto usa automaticamente o modelo base.

### Com um LLM
- **Local e gratuito (Ollama):** instale o [Ollama](https://ollama.com), rode `ollama pull qwen2.5:3b` e depois `set LLM_PROVEDOR=ollama`.
- **API da Anthropic (paga, por token):** `set LLM_PROVEDOR=anthropic` e `set ANTHROPIC_API_KEY=...`. O modelo padrão é o `claude-haiku-5-5`, o mais barato.

O prompt manda o modelo usar **somente** os trechos recuperados e tratá-los como dados, não como instruções (proteção contra *prompt injection*). Se a resposta citar um artigo que não foi recuperado, ela é descartada e o assistente devolve a resposta extrativa.

### Como servidor MCP (ex.: Claude Desktop)
Em `claude_desktop_config.json`:
```json
{ "mcpServers": { "assistente-lgpd": {
    "command": "C:\\projetos\\portfolio-gabriel\\assistente-lgpd\\.venv\\Scripts\\python.exe",
    "args": ["-m", "lgpd_rag.servidor_mcp"],
    "cwd": "C:\\projetos\\portfolio-gabriel\\assistente-lgpd" } } }
```
Ferramentas: `buscar_na_lgpd`, `ler_artigo`, `responder_sobre_lgpd`. Recurso: `lgpd://artigo/{numero}`.

## O que foi verificado
| Parte | Situação |
|---|---|
| Extração da lei (65 artigos, 96 trechos), busca, expansão e métricas | testes automatizados (CI) |
| Fine-tuning e avaliação antes/depois | executados; números acima |
| Verificação de citações, fallback e proteção do contexto | testados com um LLM simulado |
| Servidor MCP | testado com o cliente oficial do SDK, em memória e via stdio |
| Chamadas reais ao Ollama e à API da Anthropic | código pronto, **não executado** (exige Ollama instalado ou chave de API) |

## Fonte e limites
- **Texto usado:** publicação **original** da Lei 13.709/2018, obtida no portal da Câmara dos Deputados (`dados/bruto/`). Alterações posteriores, como a Lei 13.853/2019 que criou a ANPD (arts. 55-A a 55-L), **não estão incluídas**.
- **Uso:** orientação informativa, não aconselhamento jurídico.

## Tecnologias
Python · sentence-transformers · PyTorch · rank-bm25 · MCP Python SDK v2 · Ollama · API Anthropic · pytest · GitHub Actions
