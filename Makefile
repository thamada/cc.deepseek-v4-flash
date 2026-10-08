.PHONY: install run clean

.DEFAULT_GOAL := run

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# make run
# make run ARGS="--listen 0.0.0.0"
# DEEPSEEK_DEVICE=cpu|cuda|rocm|rocm-gfx120|rocm-gfx1151|rocm-gfx110|mps
# DEEPSEEK_LISTEN=127.0.0.1  DEEPSEEKK_PORT=8188  DEEPSEEK_MANAGER=0

install:
	bash $(ROOT)scripts/install.sh

run: install
	bash $(ROOT)scripts/run.sh $(ARGS)

clean:
	bash $(ROOT)scripts/clean.sh
