.PHONY: up
up:
	docker compose -f docker/docker-compose.yaml up -d --no-recreate

.PHONY: build
build:
	docker compose -f docker/docker-compose.yaml build

.PHONY: ps
ps:
	docker compose -f docker/docker-compose.yaml ps

.PHONY: restart
restart:
	docker compose -f docker/docker-compose.yaml down && docker compose -f docker/docker-compose.yaml up -d

.PHONY: stop
stop:
	docker compose -f docker/docker-compose.yaml down --remove-orphans
