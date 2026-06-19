import os, requests
from dotenv import load_dotenv

load_dotenv()
ES_HOST = os.getenv("ES_HOST")
ES_API_KEY = os.getenv("ES_API_KEY")
headers = {"Authorization": f"ApiKey {ES_API_KEY}", "Content-Type": "application/json"}

# Step 1: Ingest pipeline — strip nav boilerplate using indexOf/substring (no regex)
painless = """
if (ctx.body != null) {
  String b = ctx.body;
  int navEnd = b.indexOf('Vul in wat u zoekt');
  if (navEnd > -1) {
    b = b.substring(navEnd + 18);
  }
  int footerStart = b.indexOf('Deel deze pagina');
  if (footerStart > -1) {
    b = b.substring(0, footerStart);
  }
  int cookieStart = b.indexOf('Cookie voorkeur');
  if (cookieStart > -1) {
    b = b.substring(0, cookieStart);
  }
  b = b.trim();
  ctx.body_clean = b;
  ctx.body_semantic = b;
}
"""

r = requests.put(
    f"{ES_HOST}/_ingest/pipeline/rijksoverheid-clean",
    headers=headers,
    json={
        "description": "Strip rijksoverheid.nl nav boilerplate before embedding",
        "processors": [{"script": {"lang": "painless", "source": painless}}]
    }
)
print("Pipeline:", r.status_code, r.json())

# Step 2: Create rijksoverheid-qa-v3 index (delete first if exists from bad reindex)
requests.delete(f"{ES_HOST}/rijksoverheid-qa-v3", headers=headers)
r = requests.put(
    f"{ES_HOST}/rijksoverheid-qa-v3",
    headers=headers,
    json={
        "mappings": {
            "properties": {
                "title":            {"type": "text", "analyzer": "dutch"},
                "body":             {"type": "text", "analyzer": "dutch"},
                "body_clean":       {"type": "text", "analyzer": "dutch"},
                "body_semantic":    {"type": "semantic_text", "inference_id": "rijksoverheid-embeddings-v2"},
                "url":              {"type": "keyword"},
                "url_host":         {"type": "keyword"},
                "url_path":         {"type": "keyword"},
                "meta_description": {"type": "text", "analyzer": "dutch"},
                "headings":         {"type": "text", "analyzer": "dutch"},
                "last_crawled_at":  {"type": "date"}
            }
        }
    }
)
print("Index:", r.status_code, r.json())

# Step 3: Reindex qa → qa-v3 via clean pipeline
r = requests.post(
    f"{ES_HOST}/_reindex?wait_for_completion=false",
    headers=headers,
    json={
        "source": {"index": "rijksoverheid-qa"},
        "dest":   {"index": "rijksoverheid-qa-v3", "pipeline": "rijksoverheid-clean"}
    }
)
print("Reindex task:", r.status_code, r.json())
