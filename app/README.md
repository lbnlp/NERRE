<!-- coderabbit-ai-generated:v1 -->

<!-- coderabbit-ai-generated-section:v1:6997e3ac9e8ecbbb60c0cd918083bd6bfdda5d805efe25cf2e4ffe239dfac03f -->

# Document extraction app

A local Streamlit UI and FastAPI service for sentence-level materials doping extraction. FLAN-T5-small is fine-tuned on the repository annotations and returns `basemats`, `dopants`, and `dopants2basemats`. This model is specialized for scientific doping text. Review predictions before using them.

## Project layout

- `data/raw/`: original raw documents.
- `data/labeled/`: training JSONL and held-out annotations.
- `training/finetune.py` and `training/eval.py`: training and evaluation.
- `app/backend/`: API, extraction model, and response schemas.
- `app/frontend/streamlit_app.py`: upload interface.
- `tests/`: API and frontend tests.

The original research data and scripts remain in `doping/`.

## Setup and training

Use Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m training.finetune --epochs 5 --batch-size 4
```

Training downloads `google/flan-t5-small` and saves the checkpoint to `models/doping-flan-t5/` (ignored by Git). CPU training can take a while. Use `--help` to choose another output path, base model, or hyperparameters. To retrain, choose a new output directory or remove the old checkpoint first. The training file is `data/labeled/training_json.jsonl`; `data/labeled/test.json` is held out. Targets are checked for tokenizer round trips and token limits before training. Training loss alone does not establish extraction quality.

## Run

Start the backend from the repository root:

```bash
. .venv/bin/activate
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```bash
. .venv/bin/activate
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`. The frontend uses `http://127.0.0.1:8000` by default; set `NERRE_API_URL` before starting Streamlit to use another backend. Set `NERRE_MODEL_PATH` before starting the backend to load a checkpoint from a different directory. The status indicator reports checkpoint presence; the model loads on the first extraction request. Weights are not included in Git. The app has no authentication, OCR, or persistent storage; keep it on a trusted local network.

Upload UTF-8 `.txt` or text-based `.pdf` files (maximum 5 MB, 100 sentences, and 30 PDF pages). Scanned PDFs require OCR first. A sentence over 512 input tokens, a character unsupported by the tokenizer, or invalid model output is reported as a sentence-level error. The response and UI mark results as partial when a sentence fails. Output is limited to 256 tokens per sentence.

The API accepts multipart uploads:

```bash
curl -F 'file=@document.txt' http://127.0.0.1:8000/api/extract
```

The response has a filename, schema name, overall status, and sentences. Successful sentences contain the source text and graph; failed sentences contain their source text and error. Entity IDs are local to each sentence. `GET /api/status` reports checkpoint configuration presence.

## Evaluate

Run `python -m training.eval` to score relevant sentences from `data/labeled/test.json`. The report includes valid JSON, exact graph matches, and micro precision, recall, and F1 for entity strings and linked pairs. Invalid outputs and unsupported input characters count as empty predictions for recall. Use `--limit 20` for a reproducible quick sample. These exact-string metrics are not the paper's similarity-based scores.

## Tests

```bash
pip install httpx==0.28.1
python -m unittest discover -s tests -v
```

The API and frontend tests use fake responses or a fake extractor and do not measure model accuracy.

<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:f1fff41a8ad07d330936c44e3e5a97effeba9dccd2360b9c710c858e64339a2a -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:80484d130087e316db94c981761afa4e73882b7d6d206c2db04d8e021319d777 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:0d9f405226b3b9022a92ae4fd691a545680f249ada504420a6316fce4ae22875 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:d7537a21a45edf5836652b8c120113112138213ee86b91a9812852ecdee4ee32 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:d5acc5929b8c5e85405392c67a948e8cad498b28b41d4b599a60f68047fd0a03 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:c1b725fd2f5251c9932514c84ad0a1ca5f6b896b3eb764315e2bd795dc0df5ec -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:23cb9e816922bf8a7fa98ea6459064d46331907524ab1a22c17bbe3a7e2baddd -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:39eb409c65da2e9558312f2b0cea66c1a2ac225aa8009780226303fc76782c49 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:54672b3aad2107891aabf69e75d115f50ea96957e1bdd49a7bd8bd20535302e6 -->
<!-- coderabbit-ai-generated:v1 -->
<!-- coderabbit-ai-generated-section:v1:ea2c668c9890b1b2813fba3a8eceea2cfec60f3eb768edbb64a7f6939d902bc2 -->
<!-- /coderabbit-ai-generated:v1 -->
<!-- /coderabbit-ai-generated:v1 -->


<!-- /coderabbit-ai-generated:v1 -->