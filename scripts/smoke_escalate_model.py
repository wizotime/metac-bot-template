"""Tiny live check that the escalation model answers via litellm/Gemini (costs < $0.01).

Usage: GEMINI_API_KEY=... ESCALATE_MODEL_NAME=gemini/gemini-4-hydrogen python scripts/smoke_escalate_model.py
"""
import os
import sys
import time

import litellm

model = os.getenv("ESCALATE_MODEL_NAME", "gemini/gemini-4-hydrogen")
t0 = time.time()
try:
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Answer with one word: what is the capital of Switzerland?"}],
        max_tokens=512,
        timeout=120,
    )
except Exception as e:
    print(f"SMOKE TEST FAILED for {model}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)
text = (resp.choices[0].message.content or "").strip().replace("\n", " ")
u = resp.usage
print(
    f"SMOKE TEST OK: model={model} (served as {getattr(resp, 'model', '?')}) "
    f"answer={text[:80]!r} prompt_tokens={u.prompt_tokens} completion_tokens={u.completion_tokens} "
    f"seconds={time.time() - t0:.1f}"
)
