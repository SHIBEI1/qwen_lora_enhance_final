.PHONY: validate prepare cache train preflight infer verify

validate:
	bash scripts/prepare_dataset.sh

prepare:
	bash scripts/prepare_dataset.sh

cache:
	bash scripts/cache_features.sh

train:
	bash scripts/train_lora.sh

preflight:
	bash scripts/run_preflight.sh

infer:
	bash scripts/infer.sh --input "$(INPUT)" --output "$(OUTPUT)"

verify:
	bash scripts/verify_release.sh
