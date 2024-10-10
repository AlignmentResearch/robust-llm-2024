APPLICATION_URL ?= ghcr.io/alignmentresearch/robust-llm
RELEASE_PREFIX ?= latest
COMMIT_HASH ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)
CPU ?= 4
MEMORY ?= 60G
SHM_SIZE ?= 4Gi
GPU ?= 1
SLEEP_TIME ?= 2d
DEVBOX_NAME ?= rllm-devbox
YAML_FILE ?= k8s/auto-devbox.yaml

.PHONY: devbox devbox/% large cpu ian tom

devbox/%:
	git push
	python -c "print(open('${YAML_FILE}').read().format(NAME='${DEVBOX_NAME}', IMAGE='${APPLICATION_URL}:${RELEASE_PREFIX}', COMMIT_HASH='${COMMIT_HASH}', CPU='${CPU}', MEMORY='${MEMORY}', SHM_SIZE='${SHM_SIZE}', GPU='${GPU}', SLEEP_TIME='${SLEEP_TIME}'))" | kubectl create -f -

large:
	$(eval CPU := 8)
	$(eval MEMORY := 100G)
	$(eval GPU := 2)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-large)

cpu:
	$(eval CPU := 4)
	$(eval MEMORY := 48G)
	$(eval GPU := 0)
	$(eval DEVBOX_NAME := $(DEVBOX_NAME)-cpu)

ian:
	$(eval YAML_FILE := k8s/auto-devbox-ian.yaml)

tom:
	$(eval CPU := 1)
	$(eval YAML_FILE := k8s/auto-devbox-tom.yaml)

devbox: devbox/main
