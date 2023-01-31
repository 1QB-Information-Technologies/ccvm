.PHONY: up
up:
	docker compose -f docker/docker-compose.yaml up -d

.PHONY: exec
exec:
	docker exec -it ccvm bash

.PHONY: build
build:
	docker compose -f docker/docker-compose.yaml build

.PHONY: ps
ps:
	docker compose -f docker/docker-compose.yaml ps

.PHONY: restart
restart:
	docker compose -f docker/docker-compose.yaml down --remove-orphans && docker compose -f docker/docker-compose.yaml up -d

.PHONY: down
down:
	docker compose -f docker/docker-compose.yaml down --remove-orphans