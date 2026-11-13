"""CLI:  python perguntar.py "preciso avisar alguém se vazarem dados?"
Defina LLM_PROVEDOR=ollama (local, grátis) ou LLM_PROVEDOR=anthropic + ANTHROPIC_API_KEY. Sem isso: resposta extrativa."""
import sys

from lgpd_rag.assistente import Assistente
from lgpd_rag.busca import Buscador
from lgpd_rag.llm import do_ambiente

pergunta = " ".join(sys.argv[1:]) or input("Pergunta: ")
r = Assistente(Buscador(), do_ambiente()).responder(pergunta)
print(r.texto)
print(f"\nFontes: {', '.join('Art. ' + a for a in r.fontes)} | modo: {r.modo}")
for a in r.avisos:
    print(f"Aviso: {a}")
