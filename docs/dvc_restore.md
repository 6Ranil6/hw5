# Восстановление из DVC remote

Временная копия отслеживаемых Git файлов с пустым DVC cache. `dvc pull` выполнен из локального remote; ниже фактически проверены SHA-256 файлов. Базовая модель не входит в DVC, она загружается отдельно.

- `data/tokenized/train.pt`: `675b7fc5bfa3bfbebea7592a0aacb4be32296a2c6e7a3aa3ca106ec0ca470ea5` — совпадает.
- `data/tokenized/val.pt`: `c529ed7c27f82b640b1e7c328479270c9a2f85c8a6daa12d8b8f6d16b9cb5f64` — совпадает.
- `models/adapter_all_layers/adapter_model.safetensors`: `4753554728896676e92658b38c302c6a487fea6b269a637c0e644a1120dd2bda` — совпадает.
- `models/adapter_freeze14/adapter_model.safetensors`: `c6e7a548420f15bd7ec1c08f940bc72cfc6fdfc1e4ea14fddf78883ad70f99ea` — совпадает.
- `metrics/train_all_layers.json`: `3f23f18d95af822e1b478af2a8a30d8f3afc222e9d539ae15008d242bc779f55` — совпадает.
- `metrics/train_freeze14.json`: `32027158e76dbec61d465c8cd048c3535de42bbb154531b3682aed6385f4a937` — совпадает.
- `metrics/compare_all_layers.json`: `e3455005147a8eee419d965b08f7ba07bcd8a2b3791a62dd92b2d8eedaf2b21b` — совпадает.
