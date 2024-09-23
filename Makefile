APPLICATION_URL ?= ghcr.io/alignmentresearch/robust-llm
RELEASE_PREFIX ?= latest
COMMIT_HASH ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)
CPU ?= 4
MEMORY ?= 60G
SHM_SIZE ?= 4Gi
GPU ?= 1
DEVBOX_NAME ?= rllm-devbox

.PHONY: devbox devbox/% devbox/large

devbox/%:
	git push
	python -c "print(open('k8s/auto-devbox.yaml').read().format(NAME='${DEVBOX_NAME}', IMAGE='${APPLICATION_URL}:${RELEASE_PREFIX}', COMMIT_HASH='${COMMIT_HASH}', CPU='${CPU}', MEMORY='${MEMORY}', SHM_SIZE='${SHM_SIZE}', GPU='${GPU}'))" | kubectl create -f -

devbox/large:
	$(MAKE) devbox DEVBOX_NAME=rllm-devbox-large CPU=8 MEMORY=100G GPU=2

devbox/cpu:
	$(MAKE) devbox DEVBOX_NAME=rllm-devbox-cpu CPU=4 MEMORY=48G GPU=0

devbox: devbox/main
