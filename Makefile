run-docker-compose:
	uv sync
	docker compose up --build

clear-notebook-outputs:
	uv run --with jupyter jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb