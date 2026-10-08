# DeepSeek-V4-Flash

Run DeepSeek-V4-Flash-0731 with llama.cpp on a GPU that has 32 GiB of VRAM. The full model does not fit in that memory. Attention, shared weights, and the KV cache stay on the GPU. Routed experts stay in the host mmap (`--cpu-moe`).

`make run` installs first when the runtime or the GGUF is missing.

## Usage

```bash
make install
make run
make run ARGS="--listen 0.0.0.0"
make clean
```

`make run` generates once and exits. `ARGS="--listen HOST"` starts an OpenAI-compatible server. The default listen address is `127.0.0.1:8188`.

```bash
DEEPSEEK_PROMPT='Hi' DEEPSEEK_N=4 DEEPSEEK_CTX=512 make run
DEEPSEEK_MANAGER=1 make run ARGS="--listen 127.0.0.1"
DEEPSEEK_CLEAN_MODELS=1 make clean
```

`DEEPSEEK_MANAGER=1` starts the server in the background and writes `var/server.pid`. `make clean` removes the build and `var/`. It keeps the model unless `DEEPSEEK_CLEAN_MODELS=1`.

## Options

| Variable | Default | Meaning |
|---|---|---|
| `DEEPSEEK_DEVICE` | auto | `cpu`, `cuda`, `rocm`, `rocm-gfx120`, `rocm-gfx1151`, `rocm-gfx110`, or `mps` |
| `DEEPSEEK_QUANT` | from RAM + 32 GiB | Unsloth quant name, such as `UD-IQ1_S` |
| `DEEPSEEK_PROMPT` | one Japanese sentence | One-shot prompt |
| `DEEPSEEK_N` | `32` | Tokens to generate |
| `DEEPSEEK_CTX` | `4096` | Context length |
| `DEEPSEEK_THREADS` | `nproc` | Threads for CPU experts |
| `DEEPSEEK_MOE_CACHE_MIB` | `8192` if RAM ≥ 64 GiB, else `0` | GPU cache for CPU experts |
| `DEEPSEEK_REASONING` | `off` for one-shot, `auto` for the server | `off`, `on`, or `auto` |
| `DEEPSEEK_LISTEN` | `127.0.0.1` | Server host |
| `DEEPSEEK_PORT` | `8188` | Server port. `DEEPSEEKK_PORT` is the same setting |

The quant is chosen from host RAM plus an assumed 32 GiB of VRAM. Below 100 GiB total, the choice is `UD-IQ1_S`. Set `DEEPSEEK_QUANT` to override it. If the experts do not fit in RAM, each token reads them from disk again.
