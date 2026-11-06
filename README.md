# VisionQC Panetones

Detección de defectos en panetones (**ok · moho · deforme**) con visión por computadora.
Este repo es el **MVP**: datos, entrenamiento y, en próximos pasos, la API serverless en AWS.
Ver la arquitectura en `docs/` del proyecto (`arquitectura-mvp-visionqc.drawio`).

## Estructura

```
configs/
  train.yaml               clases, split, modelo y etapas de entrenamiento
  external_datasets.yaml   datasets públicos de Roboflow (solo para entrenamiento)
data/                      NO se sube a git (solo la estructura de carpetas)
  raw/propia/<clase>/<ID_panetón>/*.jpg
  external/<fuente>/<clase>/*.jpg
  catalog.csv              generado por `visionqc catalogo`
src/visionqc/
  catalog.py     escanea las fotos y arma el catálogo (fuente, licencia, fecha)
  split.py       split train/val/test por panetón
  roboflow.py    descarga y adapta datasets de Roboflow
  preprocess.py  preprocesamiento único para evaluación e inferencia
  train.py       entrenamiento en dos etapas + export ONNX
  cli.py         comandos `visionqc ...`
tests/           pytest (no requiere torch)
.github/workflows/ci.yml   lint + tests en cada push
```

Próximos módulos (pasos 3 a 7 del plan): `api/` (FastAPI + ONNX en Lambda), `infra/` (Terraform),
`edge/` (captura con OpenCV + dashboard Streamlit).

## Instalación

Probado en local con **Python 3.14 + PyTorch 2.15 (CPU)**; el CI corre en Python 3.12.

> Conviene tener el repo **fuera de OneDrive** (p. ej. `C:\dev\visionqc-py`): la sincronización
> choca con `.git`, el entorno virtual y miles de imágenes.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # sin GPU
pip install -e ".[dev,train,roboflow]"
pytest -q
```

Si tienes GPU NVIDIA, instala PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/
antes del segundo `pip install`.

## Flujo de trabajo

### 1. Fotos propias

Guarda cada foto en `data/raw/propia/<clase>/<ID_panetón>/`:

```
data/raw/propia/ok/P01/IMG_0001.jpg
data/raw/propia/ok/P07/IMG_0102.jpg      ← P07 todavía sano (día 1)
data/raw/propia/moho/P07/IMG_0240.jpg    ← P07 ya con moho (día 6)
data/raw/propia/deforme/P12/IMG_0300.jpg
```

- El **ID del panetón** es lo que define el split: todas sus fotos van juntas a train, val o test.
- Un panetón puede estar en varias clases (moho que aparece con los días).
- La fecha de captura se toma del EXIF de la foto.

**Protocolo de fotos**
- Celular fijo (trípode), fondo uniforme, luz constante; luego variaciones de luz y ángulo.
- ~30 fotos por panetón: 4–6 caras/ángulos × 2–3 luces, con y sin empaque si aplica.
- Moho: humedece un poco, bolsa cerrada, lugar tibio; **foto diaria** para capturar el moho
  incipiente. No abras las bolsas para olerlas y descártalas cerradas.
- Mínimo **3 panetones por clase** para poder validar y probar; ideal **5 o más** de cada defecto.

### 2. Datasets públicos (opcional)

```powershell
$env:ROBOFLOW_API_KEY = "tu_api_key"     # https://app.roboflow.com → Settings → API Keys
visionqc externos
```

Tras la primera descarga revisa los avisos de **clases sin mapear** y ajusta `class_map` en
`configs/external_datasets.yaml`; luego `visionqc externos --solo-adaptar`.
Revisa la licencia de cada dataset: **CC BY 4.0 exige dar crédito**.

### 3. Catálogo y split

```powershell
visionqc catalogo
```

Genera `data/catalog.csv` (una fila por imagen con `panetone_id`, `label`, `source`, `license`,
`captured_at`, `split`) y muestra cuántas fotos hay por clase y split.

Reglas que aplica (y verifica):
- **Externos solo en train.** Val y test son únicamente fotos propias de panetón.
- **Split por panetón**, estratificado por tipo de defecto.

### 4. Entrenamiento

```powershell
visionqc entrenar                        # completo
visionqc entrenar --epocas 1 --sin-pesos # prueba rápida del pipeline
```

Dos etapas: primero pan con moho público (aprende la textura del moho), luego tus panetones
(aprende tu producto, cámara y luz). Si no hay externos, la primera etapa se salta.
Resultado en `runs/<fecha-hora>/`: `model.onnx`, `labels.json`, `metrics.json`.

**¿Sirven los datos públicos?** Entrena con y sin ellos (quita la etapa `externo` de
`train.yaml`) y compara `metrics.json → test` sobre el **mismo** test. Quédate con los externos
solo si mejoran el **recall de moho**: dejar pasar un panetón con moho es el error caro.

## Cómo contribuir

Trabajamos con **trunk-based development** (ramas cortas + PR + squash merge a `main`) y
**Conventional Commits** en el título de cada PR. Detalles en [CONTRIBUTING.md](CONTRIBUTING.md).
