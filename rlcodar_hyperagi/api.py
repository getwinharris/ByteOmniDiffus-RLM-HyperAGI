"""
ByteOmniDiffus-RLM-HyperAGI local server and CLI adapter.

The HTTP routes keep a `/v1/...` compatibility shape for local clients. That
compatibility surface is not the project identity and does not imply any hosted
model dependency.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcodar_hyperagi.diffusion import (
    CoDARDiffusion,
    ByteIndex,
    CosineNoiseSchedule,
    ByteGroupTokenizer,
)

try:
    from fastapi import FastAPI, HTTPException, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

SERVER_TOKEN = os.getenv("BYTEOMNIDIFFUS_TOKEN") or os.getenv("RLCODAR_API_KEY")

# Global runtime instance. The implementation class name is still legacy until a
# deliberate public-symbol rename lands.
_codar: Optional[CoDARDiffusion] = None
_index: Optional[ByteIndex] = None


def init_codar(repo_root: str = None):
    """
    Initialize the ByteOmniDiffus runtime by indexing repo files.

    This is the current local loading step: the byte index is the working memory.
    """
    global _codar, _index

    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"📄 Indexing repo: {repo_root}")

    _index = ByteIndex()
    _index.add_directory(repo_root)

    schedule = CosineNoiseSchedule(T=100)
    tokenizer = ByteGroupTokenizer()
    _codar = CoDARDiffusion(_index, schedule, tokenizer)

    print(f"✅ ByteOmniDiffus initialized: {_index.stats['total_groups']} groups from {_index.stats['total_sources']} sources ({_index.stats['total_bytes']} bytes)")

    return _codar


# ============================================================================
# CLI Mode (no FastAPI needed)
# ============================================================================

def cli_completion(prompt: str) -> str:
    """Run a completion in CLI mode."""
    global _codar
    if _codar is None:
        init_codar()
    return _codar.completion(prompt)


def cli_repl():
    """Interactive REPL mode."""
    global _codar
    if _codar is None:
        init_codar()

    print("\n🤖 ByteOmniDiffus REPL (type 'quit' to exit)\n")
    while True:
        try:
            prompt = input(">>> ")
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
            response = _codar.completion(prompt)
            print(f"\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nBye!")


# ============================================================================
# FastAPI Server Mode
# ============================================================================

if HAS_FASTAPI:
    app = FastAPI(
        title="ByteOmniDiffus-RLM-HyperAGI API",
        description="Local byte-level diffusion runtime with compatibility routes",
        version="2.0.0"
    )

    security = HTTPBearer(auto_error=False)

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "byteomnidiffus"
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 16384
        stream: bool = False

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, int]

    async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
        """Verify local server token."""
        if SERVER_TOKEN is None:
            raise HTTPException(status_code=500, detail="Set BYTEOMNIDIFFUS_TOKEN before serving HTTP")
        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing token")
        if credentials.credentials != SERVER_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        return credentials.credentials

    @app.on_event("startup")
    async def startup_event():
        """Index repo on startup."""
        init_codar()

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "byteomnidiffus-rlm-hyperagi",
            "model": "byteomnidiffus",
            "indexed": _index.stats if _index else {},
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "byteomnidiffus",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "byteomnidiffus-rlm-hyperagi"
            }]
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        api_key: str = Security(verify_api_key)
    ):
        """ByteOmniDiffus local completion."""
        if _codar is None:
            raise HTTPException(status_code=503, detail="ByteOmniDiffus not initialized")

        prompt = request.messages[-1].content

        try:
            response_text = _codar.completion(prompt)

            input_tokens = len(prompt.encode('utf-8'))
            output_tokens = len(response_text.encode('utf-8'))

            return ChatCompletionResponse(
                id=f"byteomnidiffus-{int(time.time() * 1000)}",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/rlm/status")
    async def rlm_status():
        """Runtime status."""
        if _index is None:
            return {"status": "not_initialized"}
        return {
            "status": "ready",
            "model": "byteomnidiffus",
            "backend": "pure-python-byte-diffusion",
            "index": _index.stats,
            "sources": list(_index.sources.keys())[:20],
        }

    @app.post("/v1/rlm/load_context")
    async def load_context(
        files: List[str],
        api_key: str = Security(verify_api_key)
    ):
        """Add files to the byte index."""
        if _index is None:
            raise HTTPException(status_code=503, detail="ByteOmniDiffus not initialized")

        loaded = []
        for file_path in files:
            try:
                count = _index.add_file(file_path)
                loaded.append({
                    "path": file_path,
                    "groups": count,
                    "status": "indexed"
                })
            except Exception as e:
                loaded.append({
                    "path": file_path,
                    "status": "error",
                    "error": str(e)
                })

        return {"loaded": loaded}


def main():
    """Run API server or CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="ByteOmniDiffus-RLM-HyperAGI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repl", action="store_true", help="Run in REPL mode")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--repo", type=str, help="Repo root to index")

    args = parser.parse_args()

    if args.query:
        if args.repo:
            init_codar(args.repo)
        else:
            init_codar()
        print(cli_completion(args.query))
        return

    if args.repl:
        if args.repo:
            init_codar(args.repo)
        else:
            init_codar()
        cli_repl()
        return

    if not HAS_FASTAPI:
        print("❌ FastAPI not installed. Use --repl or --query mode, or: pip install fastapi uvicorn")
        return

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🚀 ByteOmniDiffus-RLM-HyperAGI API Server v2.0        ║
║   Local byte-level diffusion runtime                     ║
║   No hosted model dependency                             ║
║                                                          ║
║   Host: {args.host:<42} ║
║   Port: {args.port:<42} ║
║                                                          ║
║   Endpoints:                                             ║
║   POST /v1/chat/completions  (compatibility route)       ║
║   GET  /v1/models                                        ║
║   GET  /health                                           ║
║   GET  /v1/rlm/status                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
