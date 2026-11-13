import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LGPD_SO_BM25", "1")  # testes rápidos não baixam o modelo de embeddings
