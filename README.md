# ARTEMIS — Inference Server

Server inferensi untuk sistem deteksi kebakaran berbasis YOLOv8. Menerima frame dari edge node (Raspberry Pi) dan mengembalikan hasil deteksi api/asap.

## Persyaratan

- Python 3.10+
- GPU (opsional, CPU didukung)

## Instalasi

```bash
git clone https://github.com/inahrafm/artemis-server.git
cd artemis-server
pip install -r requirements.txt
```

## Download Model

```bash
mkdir -p model
curl https://model.weartemis.me/best.pt -o model/best.pt
```

## Menjalankan Server

```bash
python3 scripts/run_server.py \
    --model model/best.pt \
    --port 8000 \
    --device cpu
```

Gunakan `--device cuda` jika tersedia GPU.

## Verifikasi

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok", "device": "cpu", ...}
```

## Endpoints

| Endpoint             | Method | Deskripsi                  |
| -------------------- | ------ | -------------------------- |
| `/health`            | GET    | Status server              |
| `/status`            | GET    | Statistik request per node |
| `/nodes`             | GET    | Daftar edge node aktif     |
| `/infer`             | POST   | Inferensi gambar           |
| `/thresholds`        | GET    | Threshold deteksi aktif    |
| `/reload_thresholds` | POST   | Reload threshold dari file |
| `/reload_model`      | POST   | Reload model tanpa restart |

## Inferensi

```bash
curl -X POST http://localhost:8000/infer \
    -F "file=@gambar.jpg" | python3 -m json.tool
```

Response:

```json
{
  "confmax_fire": 0.0,
  "confavg_fire": 0.0,
  "count_fire": 0,
  "confmax_smoke": 0.85,
  "confavg_smoke": 0.85,
  "count_smoke": 1,
  "decision": "FIRE",
  "server_inference_ms": 352.19,
  "server_total_ms": 388.64
}
```
