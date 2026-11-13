"""Fine-tuning do modelo de embeddings com pares (pergunta -> trecho da lei) usando MultipleNegativesRankingLoss:
cada pergunta deve ficar mais perto do seu trecho do que dos trechos das outras perguntas do lote (negativos "de graça").
Usa SOMENTE avaliacao/perguntas_treino.jsonl; as perguntas de avaliação nunca entram no treino.
Uso: python scripts/ajustar_embeddings.py [--epocas 4]"""
import argparse
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from lgpd_rag.avaliacao import carregar_perguntas
from lgpd_rag.busca import MODELO_AJUSTADO, MODELO_BASE
from lgpd_rag.corpus import carregar

ap = argparse.ArgumentParser()
ap.add_argument("--epocas", type=int, default=4)
ap.add_argument("--lote", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
a = ap.parse_args()
random.seed(42)

modelo = SentenceTransformer(MODELO_BASE, device="cpu")
modelo.max_seq_length = 192
# A camada de vocabulário multilíngue (250 mil tokens) concentra ~80% dos parâmetros. Congelá-la reduz a memória do
# treino de ~4 GB para menos de 1 GB e preserva o conhecimento multilíngue; só as camadas transformer são ajustadas.
for p in modelo[0].auto_model.embeddings.parameters():
    p.requires_grad = False
treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
total = sum(p.numel() for p in modelo.parameters())
print(f"parâmetros treináveis: {treinaveis/1e6:.1f}M de {total/1e6:.1f}M")
trechos = carregar()
por_artigo: dict[str, list] = {}
for t in trechos:
    por_artigo.setdefault(t.artigo, []).append(t)

pares = []
for q in carregar_perguntas("perguntas_treino.jsonl"):
    for art in q["artigos"]:
        cands = por_artigo[art]
        # artigo longo tem vários trechos: o positivo é o trecho mais parecido com a pergunta segundo o modelo base
        emb = modelo.encode([q["pergunta"]] + [c.texto for c in cands], normalize_embeddings=True)
        melhor = cands[int((emb[1:] @ emb[0]).argmax())]
        pares.append({"anchor": q["pergunta"], "positive": melhor.texto})
random.shuffle(pares)
print(f"{len(pares)} pares de treino")

args = SentenceTransformerTrainingArguments(
    output_dir=str(RAIZ / "modelos" / "_checkpoints"), num_train_epochs=a.epocas, per_device_train_batch_size=a.lote,
    learning_rate=a.lr, warmup_steps=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
    logging_steps=5, report_to=[], seed=42, use_cpu=True)
trainer = SentenceTransformerTrainer(model=modelo, args=args, train_dataset=Dataset.from_list(pares),
                                     loss=MultipleNegativesRankingLoss(modelo))
trainer.train()
modelo.save(str(MODELO_AJUSTADO))
(MODELO_AJUSTADO / "treino_lgpd.json").write_text(json.dumps({"base": MODELO_BASE, "pares": len(pares), **vars(a)}, indent=2))
print(f"modelo salvo em {MODELO_AJUSTADO}")
