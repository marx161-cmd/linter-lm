"""contextstore -- silent, embedding-based context retrieval for the
LinteR-LM proxy. No MCP, no tool-call, no keyword/cooldown/sticky activation
rules. Ingest arbitrary files (code, lyrics, articles, transcripts);
retrieve by similarity against the live conversation; the proxy splices
hits into the outbound request before the model ever sees them.
"""
from .contextstore import ContextStore
from .models import ContextFile, RetrievalHit, RetrievalResult

__all__ = ["ContextStore", "ContextFile", "RetrievalHit", "RetrievalResult"]
