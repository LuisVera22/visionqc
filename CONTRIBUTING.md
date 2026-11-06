# Cómo contribuir

## Flujo de trabajo: trunk-based

- **`main` siempre funciona y es desplegable.** No se hace push directo: todo entra por Pull Request
  con el CI en verde.
- Cada cambio va en una **rama corta** (horas, máximo 1–2 días) creada desde `main`.
- Los PR se integran con **squash merge**: cada PR queda como un único commit en `main`, cuyo
  mensaje es el **título del PR**.
- Las versiones se marcan con **tags semánticos** sobre `main` (`v0.1.0`, `v0.2.0`…).
- Los **experimentos de ML no van en ramas largas**: se definen en `configs/` y quedan registrados
  en `runs/<fecha-hora>/` con su `metrics.json`. Git es para el código.

```powershell
git switch main
git pull
git switch -c feat/api-inferencia
# ... cambios, commits ...
git push -u origin feat/api-inferencia
gh pr create --fill        # o desde la web de GitHub
```

Después del merge, borra la rama y vuelve a `main`:

```powershell
git switch main
git pull
git branch -d feat/api-inferencia
```

## Nombres de rama

`<tipo>/<descripcion-corta>` en minúsculas y con guiones, usando los mismos tipos de los commits:

```
feat/api-inferencia
fix/split-panetones
docs/protocolo-fotos
ci/deploy-lambda
```

## Commits: Conventional Commits

Formato: `tipo(alcance): descripción` — la descripción en español, en imperativo o presente,
sin punto final.

| Tipo | Cuándo | Efecto en la versión |
|---|---|---|
| `feat` | nueva funcionalidad | sube la versión menor (`0.1.0` → `0.2.0`) |
| `fix` | corrección de un bug | sube el parche (`0.1.0` → `0.1.1`) |
| `perf` | mejora de rendimiento | parche |
| `refactor` | cambio interno sin cambiar comportamiento | — |
| `test` | tests | — |
| `docs` | documentación | — |
| `ci` | workflows de GitHub Actions | — |
| `build` | dependencias, empaquetado, Docker | — |
| `chore` | mantenimiento que no encaja en lo anterior | — |

Un cambio incompatible se marca con `!` (`feat(api)!: ...`) y sube la versión mayor.

**Alcances** (opcionales) según el módulo del monorepo:

| Alcance | Qué cubre |
|---|---|
| `data` | catálogo, split, datasets externos |
| `train` | entrenamiento y export del modelo |
| `api` | API de inferencia (FastAPI / Lambda) |
| `infra` | Terraform |
| `edge` | agente de captura y dashboard |
| `ci` | workflows |
| `deps` | dependencias |

Ejemplos:

```
feat(api): endpoint POST /inspecciones con ONNX Runtime
fix(train): momentum de BatchNorm para datasets chicos
docs(data): protocolo para la sesión de fotos
ci: actualizar checkout y setup-python a v7
```

### ¿Qué se valida?

Como los PR se integran con squash merge, lo único que llega a `main` es el **título del PR**.
El workflow `.github/workflows/pr-title.yml` revisa que ese título siga Conventional Commits; los
commits intermedios dentro de la rama pueden ser más informales.

## Antes de abrir un PR

```powershell
ruff check .
pytest -q
```
