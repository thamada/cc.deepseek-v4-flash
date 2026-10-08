# 変更履歴

> **注意**:
>   本ドキュメントは変更履歴です。日付はdateコマンドで確認して2026-01-23 12:34:55のように年-月-日 時:分:秒のようにします。
>   最も最新のものから順に並べて記入します。

## 2026-10-08 09:24:19

DeepSeek-V4-Flash-0731 を、VRAM 32GiB を前提に `make install` / `make run` で実行できるようにした。

- `scripts/install.sh` が llama.cpp を検出デバイス向けにビルドし、Unsloth の GGUF を取得する。ダウンロードとビルドは並行し、`.install/` のスタンプが一致すればやり直さない。
- `scripts/run.sh` が一回生成、または `ARGS="--listen HOST"` で OpenAI 互換サーバを起動する。attention と共有重みは GPU、ルーティングされた expert は `--cpu-moe` でホストに置く。
- 量子化は RAM と 32GiB の合計から選ぶ。合計が 100GiB 未満のときは UD-IQ1_S である。`DEEPSEEK_QUANT` で上書きできる。
- ROCm では `LD_LIBRARY_PATH` に HIP のライブラリを加え、HIP graph は既定で無効にする。expert の GPU キャッシュは、ホスト RAM が 64GiB 未満なら既定 0 である。
- `scripts/clean.sh` はビルドと `var/` を消す。モデルは `DEEPSEEK_CLEAN_MODELS=1` のときだけ消す。
- 設計は `doc/design.md` に書いた。

このホスト（Radeon AI PRO R9700、gfx1201、RAM 約 15GiB）では UD-IQ1_S を使い、プロンプト `Hi` から `Hello! How can` が生成され、終了コード 0 で完了した。expert が HDD 上にありページキャッシュに載り切らないため、モデル読み込みを含めて約 95 分かかった。
