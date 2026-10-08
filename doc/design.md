# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog.md` を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

本リポジトリは DeepSeek-V4-Flash-0731 を、VRAM 32GiB の GPU 上で llama.cpp から実行する。モデル全体は 32GiB に入らない。attention、共有重み、埋め込み、KV キャッシュを GPU に載せ、ルーティングされた expert はホスト側に mmap した GGUF から読む。

推論エンジンは upstream の llama.cpp（`deepseek4` 対応済みの main）を、検出したデバイス向けにビルドする。重みは Unsloth の GGUF `unsloth/DeepSeek-V4-Flash-0731-GGUF` を使う。

## ディレクトリ

| パス | 内容 |
|---|---|
| `Makefile` | `install` / `run` / `clean` の入口 |
| `scripts/common.sh` | パス、デバイス検出、量子化の選択 |
| `scripts/install.sh` | 依存パッケージ、llama.cpp のビルド、モデル取得 |
| `scripts/run.sh` | 一回生成、または OpenAI 互換サーバ |
| `scripts/clean.sh` | ビルド成果物の削除 |
| `third_party/llama.cpp` | llama.cpp のソースとビルド |
| `models/DeepSeek-V4-Flash-0731-GGUF/` | 取得した GGUF シャード |
| `.install/` | ビルドとモデルのスタンプ |
| `.venv/` | `huggingface_hub` 用の仮想環境 |
| `var/` | 実行ログ、サーバの PID |

`third_party/`、`models/`、`.install/`、`.venv/`、`var/`、`*.gguf` は git の対象外である。

## コマンド

既定の目標は `run` である。`run` は `install` に依存する。

| コマンド | 動作 |
|---|---|
| `make install` | 実行環境と、選択した量子化の GGUF を用意する |
| `make run` | 一回だけ生成して終了する |
| `make run ARGS="--listen 0.0.0.0"` | `llama-server` を起動する。`--listen` の値は待受ホストになる |
| `make clean` | llama.cpp のビルド、`.install/`、`var/` を消す。動いているサーバがあれば止める |

`make clean` はモデルと仮想環境を残す。これらも消すときは `DEEPSEEK_CLEAN_MODELS=1` を付ける。

## オフロード

DeepSeek-V4-Flash-0731 は 284B 総パラメータ、13B アクティブの MoE（expert 256、層数約 43）である。Unsloth の最小量子化 UD-IQ1_S でも約 83GB あり、VRAM 32GiB には載らない。expert が重みの大半を占める。

GPU 上に置くのは attention、共有 expert、埋め込み、KV である。ルーティングされた expert は `--cpu-moe` によりホストに残す。llama.cpp には次を渡す。

- `-ngl all`
- `--cpu-moe`
- `--fit off`
- `--moe-cache-mib`

VRAM 量はドライバの報告値ではなく 32GiB として扱う（`DEEPSEEK_VRAM_GIB`、既定 32）。Radeon AI PRO R9700 の表示は約 32624MiB だが、同じ前提で計算する。

`--moe-cache-mib` は、直前に使った expert を GPU に置く量である。未指定でホスト RAM が 64GiB 以上のときは 8192MiB、それ未満のときは 0 である。この ROCm では大きな HSA 確保がランタイム内で長時間戻らないことがあり、RAM が小さいホストではキャッシュを既定で切る。`DEEPSEEK_MOE_CACHE_MIB` で上書きできる。

ROCm 実行時は `GGML_CUDA_DISABLE_GRAPHS=1` を既定にする。HIP graph が HSA ランタイム内で進まなくなるのを避けるためである。

## 量子化の選択

`DEEPSEEK_QUANT` が無ければ、ホスト RAM と 32GiB の合計から次の量子化を選ぶ。expert は mmap なので、これは VRAM の上限ではなく、ページキャッシュに載る快適さの目安である。

| RAM + VRAM | 量子化 |
|---|---|
| 175GiB 以上 | UD-Q8_K_XL |
| 165GiB 以上 | UD-Q4_K_XL |
| 120GiB 以上 | UD-IQ3_XXS |
| 100GiB 以上 | UD-IQ2_XXS |
| それ未満 | UD-IQ1_S |

取得対象は `models/DeepSeek-V4-Flash-0731-GGUF/<量子化>/` 以下の全シャードである。先頭シャードはメタデータだけのことがある。2 枚目以降が 1GB を超えていることを、取得済みの判定にする。llama.cpp には先頭シャードのパスを渡す。

サンプリングは公式の一回生成に合わせ、temperature 1.0、top-p 1.0、min-p 0.01 とする。チャットテンプレートは `--jinja` で有効にする。

## デバイス検出とビルド

`DEEPSEEK_DEVICE` が無ければ次の順で決める。

1. macOS なら `mps`
2. `nvidia-smi` が使えるなら `cuda`
3. `rocminfo` の gfx 名が `gfx120*` なら `rocm-gfx120`、`gfx1151` / `gfx1152` / `gfx1153` なら `rocm-gfx1151`、`gfx110*` なら `rocm-gfx110`、それ以外の gfx なら `rocm`
4. どれでもなければ `cpu`

ビルド時の GPU ターゲットは、検出した gfx 名を使う。`rocm-gfx120` で gfx120 系が見つからなければ `gfx1201`、`rocm-gfx110` で見つからなければ `gfx1100`、`rocm-gfx1151` は `gfx1151` である。gfx1152 と gfx1153 は rocBLAS の Tensile ライブラリが無いため、実行時に `HSA_OVERRIDE_GFX_VERSION=11.5.1` を設定し、gfx1151 のイメージとして動かす。

HIP ビルドは `hipconfig` の clang を `HIPCXX` にし、`GGML_HIP=ON` と `GPU_TARGETS` を渡す。CUDA は `GGML_CUDA=ON`、Metal は `GGML_METAL=ON`、CPU は GPU バックエンドを切る。成果物は `llama-cli` と `llama-server` である。コンパイル並列数は 6 を上限にする。

ROCm 実行時は `hipconfig -R` の `lib` と `/opt/rocm/lib` を `LD_LIBRARY_PATH` に加える。`libhipblas.so.3` などを解決するためである。

モデルのダウンロードと llama.cpp のビルドは並行する。ビルドスタンプは `デバイス|GPUターゲット|gitのリビジョン`、モデルスタンプは量子化名である。両方が一致し、バイナリとシャードが揃っていれば `make install` は再取得も再ビルドもしない。

Linux で `cmake`、`ninja`、`g++` が無いときは apt で入れる。モデル取得には `.venv` の `huggingface_hub` を使う。

## 推論

`make run` に `--listen` が無いときは `llama-cli` の一回生成である。

| 項目 | 既定 |
|---|---|
| プロンプト | `こんにちは。自己紹介を一文でしてください。`（`DEEPSEEK_PROMPT`） |
| 生成トークン数 | 32（`DEEPSEEK_N`） |
| コンテキスト長 | 4096（`DEEPSEEK_CTX`） |
| スレッド数 | `nproc`（`DEEPSEEK_THREADS`） |
| reasoning | `off`（`DEEPSEEK_REASONING`） |

`--single-turn` を付け、標準出力は `var/run.log` にも残す。起動ログに Flash Attention 失敗、または HIP / GPU のメモリ不足が出たときは、`--flash-attn off` と `--moe-cache-mib 0` で一度だけ再試行する。

`--listen` または `--listen=HOST` があるときは `llama-server` である。待受ホストの既定は `127.0.0.1`（`DEEPSEEK_LISTEN`）、ポートの既定は 8188（`DEEPSEEK_PORT`。`DEEPSEEKK_PORT` も同じポートとして読む）。reasoning の既定は `auto` である。`DEEPSEEK_MANAGER=1` のときはバックグラウンドで起動し、PID を `var/server.pid`、ログを `var/server.log` に書く。既にその PID が生きていれば起動しない。

`--listen` 以外の引数は、そのまま llama.cpp に渡す。

## 環境変数

| 変数 | 意味 |
|---|---|
| `DEEPSEEK_DEVICE` | `cpu` / `cuda` / `rocm` / `rocm-gfx120` / `rocm-gfx1151` / `rocm-gfx110` / `mps` |
| `DEEPSEEK_VRAM_GIB` | 量子化選択に使う VRAM。既定 32 |
| `DEEPSEEK_QUANT` | 量子化名。未指定なら上の表 |
| `DEEPSEEK_PROMPT` | 一回生成のプロンプト |
| `DEEPSEEK_N` | 生成トークン数 |
| `DEEPSEEK_CTX` | コンテキスト長 |
| `DEEPSEEK_THREADS` | expert 計算のスレッド数 |
| `DEEPSEEK_MOE_CACHE_MIB` | GPU 上の expert キャッシュ |
| `DEEPSEEK_REASONING` | `off` / `on` / `auto` |
| `DEEPSEEK_LISTEN` | サーバの待受ホスト |
| `DEEPSEEK_PORT` | サーバのポート。`DEEPSEEKK_PORT` も可 |
| `DEEPSEEK_MANAGER` | `1` でサーバをバックグラウンド起動 |
| `DEEPSEEK_CLEAN_MODELS` | `1` で `make clean` がモデルと仮想環境も消す |
| `GGML_CUDA_DISABLE_GRAPHS` | ROCm では既定 1。明示した値を優先する |

## 制約

ホスト RAM が量子化より小さいと、expert のページがキャッシュから落ち、トークンごとにディスクを読み直す。そのとき生成速度はディスクのランダム読み出しで決まる。VRAM 32GiB は非 expert 重みと KV の置き場であり、expert 全体を載せる容量ではない。

UD-IQ1_S は、RAM と VRAM の合計が 100GiB 未満のときの選択である。より高い量子化は `DEEPSEEK_QUANT` で指定でき、その分の RAM かディスク帯域が要る。
