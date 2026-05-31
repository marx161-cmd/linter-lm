# Notice

LinteR-LM is an experimental prototype published for public use, copying,
forking, and adaptation. The author does not promise maintenance, support,
compatibility fixes, issue response, or a stable roadmap.

## Concept Credits

- DavidAU's SillyTavern patch inspired the broader idea of watching model
  output while it streams, detecting degraded visible output, and nudging
  sampler parameters instead of treating every failure as a full retry problem.
- SillyTavern provided useful reference context for character/chat frontend
  workflows and local LLM experimentation.
- Ollama, FastAPI, httpx, and Uvicorn provide the local model and proxy stack
  this prototype is built around.

This repository does not vendor or copy DavidAU's modified SillyTavern files.
The implementation here is a small OpenAI-compatible proxy that applies a
separate, lighter V1 behavior: think-span bypass, visible-output monitoring,
simple JSON repair for tool-call-shaped text, and one-shot sampler patches.
