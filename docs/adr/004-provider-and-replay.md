# ADR-004: One provider interface with record and replay

Status: accepted, 2026-10-25

Context: CI must run without a key, and published numbers must be reproducible after model or price changes. A live run
costs money, so a crash or a second run must not pay twice.

Decision: every LLM call goes through `LLMClient.complete(LLMRequest)`. Backends: Anthropic (official SDK; Message Batches
API for the judge) and Ollama (plain HTTP, with `num_ctx`, temperature 0 and a fixed seed in the request). A replay layer
keys each request by SHA-256 of its canonical JSON, which includes the prompt hash, the stage and the provider options;
modes are record, replay and live. Record mode reads the cassette first and calls the provider only for missing entries
(`--refresh` forces calls); every entry stores the request, the response, a timestamp, the SDK version, the Batch id and
the cost at record time, and replay reports that recorded cost next to the cost recomputed from the current price table.
Failed Batch items are never stored. Cassettes are separate files per profile: CI forces replay on a small hand-written
cassette marked synthetic, and other profiles refuse synthetic entries, so a live run can never overwrite or be served CI
fixtures. Sampling parameters are never sent to Anthropic (they return HTTP 400 on current models), so run-to-run variance
is measured with repeats, which are separate entries through the stage in the key.

Consequences: a prompt change invalidates its recordings by construction. The synthetic cassette tests the plumbing only;
the report labels any number derived from it as synthetic. The judge Batch id is written to the run directory and an
interrupted wait is resumed by running the same command again.
