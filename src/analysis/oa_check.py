"""Minimal OpenAI key/model check. Reads OPENAI_API_KEY from env (never hard-coded)."""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
for model in ["gpt-4o", "gpt-4.1", "gpt-4o-mini"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            max_tokens=5,
        )
        print(f"OK  {model}: {r.choices[0].message.content.strip()!r}")
    except Exception as e:
        print(f"ERR {model}: {type(e).__name__}: {str(e)[:120]}")
