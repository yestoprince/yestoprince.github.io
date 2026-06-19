#!/usr/bin/env python3
"""
Deploy HuggingFace embedding models to Elasticsearch via eland.
Usage:
    python deploy_model.py          # interactive menu
    python deploy_model.py jina     # deploy jina-embeddings-v3
    python deploy_model.py bge      # deploy BAAI/bge-m3
    python deploy_model.py e5large  # deploy multilingual-e5-large
    python deploy_model.py list     # list deployed models in Elastic
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

ES_HOST    = os.getenv("ES_HOST", "").rstrip("/")
ES_API_KEY = os.getenv("ES_API_KEY", "")

if not ES_HOST or not ES_API_KEY:
    print("ERROR: ES_HOST and ES_API_KEY must be set in .env")
    sys.exit(1)

MODELS = {
    "e5large": {
        "hub_id":  "intfloat/multilingual-e5-large",
        "es_id":   "intfloat__multilingual-e5-large",
        "desc":    "E5-Large · 512 tok · multilingual · ~560M params · RECOMMENDED",
        "task":    "text_embedding",
        "size_gb": 2.1,
    },
    "e5base": {
        "hub_id":  "intfloat/multilingual-e5-base",
        "es_id":   "intfloat__multilingual-e5-base",
        "desc":    "E5-Base · 512 tok · multilingual · ~270M params · fast",
        "task":    "text_embedding",
        "size_gb": 1.0,
    },
    "minilm": {
        "hub_id":  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "es_id":   "sentence-transformers__paraphrase-multilingual-minilm-l12-v2",
        "desc":    "MiniLM-L12 · 128 tok · multilingual · ~120M params · very fast",
        "task":    "text_embedding",
        "size_gb": 0.5,
    },
    # NOTE: jina-embeddings-v3 and BAAI/bge-m3 require trust_remote_code=True
    # and are NOT compatible with eland import.
}


def es_request(path, method="GET"):
    url = f"{ES_HOST}/{path.lstrip('/')}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"ApiKey {ES_API_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:
        return None, str(e)


def list_deployed_models():
    data, err = es_request("_ml/trained_models?size=100")
    if err:
        print(f"ERROR listing models: {err}")
        return
    configs = data.get("trained_model_configs", [])
    print(f"\n{'Model ID':<55} {'Type':<20} {'State'}")
    print("-" * 90)
    for m in configs:
        mid   = m.get("model_id", "")
        mtype = m.get("model_type", "")
        state = m.get("fully_defined", "")
        print(f"{mid:<55} {mtype:<20} {state}")
    print(f"\nTotal: {len(configs)} models")


def check_model_exists(es_id):
    _, err = es_request(f"_ml/trained_models/{es_id}")
    return err is None


def check_eland():
    result = subprocess.run(
        ["eland_import_hub_model", "--help"],
        capture_output=True, text=True
    )
    if result.returncode != 0 and "usage" not in result.stdout.lower():
        print("ERROR: eland not found. Install with: pip install eland[pytorch]")
        return False
    return True


def deploy_model(key):
    model = MODELS[key]
    print(f"\n{'='*60}")
    print(f"Model  : {model['hub_id']}")
    print(f"ES ID  : {model['es_id']}")
    print(f"Task   : {model['task']}")
    print(f"Size   : ~{model['size_gb']} GB download")
    print(f"{'='*60}")

    if check_model_exists(model["es_id"]):
        print(f"\nModel '{model['es_id']}' already exists in Elastic.")
        ans = input("Reimport? [y/N]: ").strip().lower()
        if ans != "y":
            print("Skipped.")
            return

    if not check_eland():
        sys.exit(1)

    print("\nStarting import — this will take several minutes...\n")

    cmd = [
        "eland_import_hub_model",
        "--url",          ES_HOST,
        "--es-api-key",   ES_API_KEY,
        "--hub-model-id", model["hub_id"],
        "--task-type",    model["task"],
        "--es-model-id",  model["es_id"],
        "--start",
    ]

    # Print command without API key for visibility
    safe_cmd = cmd[:4] + ["<api-key-hidden>"] + cmd[6:]
    print("Command:", " ".join(safe_cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ SUCCESS: '{model['es_id']}' deployed to Elastic")
        print("\nNext steps:")
        print(f"  1. Create inference endpoint: python setup_pipeline.py --inference {key}")
        print(f"  2. Reindex with new model")
    else:
        print(f"\n✗ FAILED (exit code {result.returncode})")
        print("Check output above for details.")


def interactive_menu():
    print("\n=== Elastic Model Deployer ===")
    print(f"Cluster: {ES_HOST}\n")
    print("Available models:\n")
    for key, m in MODELS.items():
        exists = check_model_exists(m["es_id"])
        status = "✓ deployed" if exists else "  not deployed"
        print(f"  [{key:<8}] {status}  {m['desc']}")
    print(f"  [list    ]            List all models in Elastic")
    print(f"  [quit    ]            Exit\n")

    choice = input("Select model to deploy: ").strip().lower()
    if choice == "quit":
        return
    if choice == "list":
        list_deployed_models()
        return
    if choice not in MODELS:
        print(f"Unknown choice: {choice}")
        return
    deploy_model(choice)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        interactive_menu()
    elif sys.argv[1] == "list":
        list_deployed_models()
    elif sys.argv[1] in MODELS:
        deploy_model(sys.argv[1])
    else:
        print(f"Unknown argument: {sys.argv[1]}")
        print(f"Valid options: {', '.join(MODELS)} | list")
        sys.exit(1)
