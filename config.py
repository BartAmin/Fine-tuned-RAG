import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")
LLM_MODEL         = "claude-haiku-4-5"
MAX_TOKENS        = 500
MAX_TOKENS_JUDGE  = 1000 
RETRY_WAIT        = 60
MAX_RETRIES       = 3

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "microsoft/harrier-oss-v1-0.6b"
DIM         = 1024  

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS   = 20
N_TRIALS = 500
PATIENCE = 5

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 20

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_CORPUS    = "data/cache_corpus.pkl"
CACHE_QUESTIONS = "data/cache_questions.pkl"
CACHE_QA        = "data/cache_qa.pkl"
SYNTH_QUESTIONS = "data/synthetic_questions.xlsx"
BEST_MODEL      = "models/best_model.pt"
RESULTS_CSV     = "results/evaluation_results.csv"
RESULTS_PLOT    = "results/retrieval_comparison.png"