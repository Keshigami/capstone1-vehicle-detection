# Capstone 1: vehicle detection (port 8101). `just setup && just train && just test`, then `just up`.
set shell := ["bash", "-cu"]

port := "8101"
data_root := env_var_or_default("DATA_ROOT", env_var("HOME") + "/Caltech CTME/PGC AIML - Capstone/Datasets")
data_dir := justfile_directory() + "/../data/capstone1"
pidfile := justfile_directory() + "/.server.pid"

# uv sync, copy the shared stylesheet, extract Images.zip if missing
setup:
    uv sync
    mkdir -p static models
    cp common/ui.css static/ui.css
    if [ ! -d "{{data_dir}}/Images" ]; then \
      echo "extracting Images.zip into {{data_dir}}"; mkdir -p "{{data_dir}}"; \
      unzip -q -o "{{data_root}}/Capstone 1/Part 1/Images.zip" -d "{{data_dir}}"; \
    fi
    cp -n "{{data_root}}/Capstone 1/Part 1/labels.csv" "{{data_dir}}/labels.csv" 2>/dev/null || true
    ls "{{data_dir}}/Images" | wc -l | xargs -I{} echo "{} images ready"

# quick training: ~1,500 images, 5 epochs, a few minutes
train:
    uv run python -m src.train --quick

# full training: all 5,626 images, 20 epochs
train-full:
    uv run python -m src.train --full

# train only if there is no model yet (so `just test` works right after `just setup`)
ensure-model:
    [ -f models/detector.keras ] || just train

# foreground server
run:
    uv run uvicorn app:app --host 127.0.0.1 --port {{port}}

# background server with a pid file
up: ensure-model
    just down >/dev/null 2>&1 || true
    nohup uv run uvicorn app:app --host 127.0.0.1 --port {{port}} > .server.log 2>&1 & echo $! > "{{pidfile}}"
    for i in $(seq 1 60); do curl -sf localhost:{{port}}/health >/dev/null && break; sleep 1; done
    curl -s localhost:{{port}}/health; echo

down:
    if [ -f "{{pidfile}}" ]; then kill $(cat "{{pidfile}}") 2>/dev/null || true; rm -f "{{pidfile}}"; fi
    lsof -ti tcp:{{port}} | xargs kill 2>/dev/null || true

test: ensure-model
    uv run pytest -q tests

lint:
    uv run ruff check src app.py tests

# curl a dataset sample through /predict (needs `just up`)
demo:
    curl -s -F "file=@{{data_dir}}/Images/00000001.jpg" localhost:{{port}}/predict | uv run python -m json.tool

# same, without the server
predict image="":
    uv run python -m src.predict {{image}}
