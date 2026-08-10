# Convenience targets for the common registry workflows.
# Run `make help` to see them all.

.PHONY: help generate lint validate verify clean release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

generate: ## Regenerate models.json from scratch (no URL validation)
	.venv/bin/python generate.py --force --no-validate

lint: ## Check models.json is in canonical format
	.venv/bin/python scripts/lint_json.py

lint-fix: ## Reformat models.json into canonical form in place
	.venv/bin/python scripts/lint_json.py --rewrite

validate: ## Validate models.json against schema.json + sanity checks
	.venv/bin/python validate.py

validate-online: ## Validate + HEAD-check a sample of URLs
	.venv/bin/python validate.py --online

all: generate lint validate ## Generate, lint, and validate in one shot

verify: ## Synthesise a clip for the smallest model of each type
	@echo "Verifying one model per type. This downloads ~400 MB."
	@for id in $$(.venv/bin/python scripts/pick_smallest_per_type.py | .venv/bin/python -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)]"); do \
		echo "--- $$id ---"; \
		.venv/bin/python scripts/verify_model.py $$id --cache-dir .verify-cache || exit 1; \
	done

clean: ## Remove caches and downloaded verification models
	rm -rf .verify-cache __pycache__ scripts/__pycache__

release: ## Print the tag-push command for a new release
	@echo "To publish a release:"
	@echo "  git tag v$$(date -u +%Y-%m-%d) && git push origin v$$(date -u +%Y-%m-%d)"
	@echo "The release.yml workflow rebuilds, checksums, and uploads the assets."
