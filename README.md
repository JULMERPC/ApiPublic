# API de Predicción de Iris

Un servicio FastAPI mínimo que entrena un clasificador Iris con scikit-learn y expone un endpoint de predicción más un CRUD pequeño sobre las predicciones almacenadas. Todo se ejecuta localmente: sin base de datos ni servicios externos. La documentación interactiva de Swagger se sirve en `/docs`.

## Ruta rápida

1. Cree un entorno virtual e instale las dependencias.
2. Inicie la API.
3. Abra <http://127.0.0.1:8000/docs> y pruebe `POST /api/v1/predict`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Si PowerShell bloquea el script de activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

El primer arranque entrena el modelo y lo guarda en `models/iris_model.joblib`. Los arranques posteriores cargan ese archivo en lugar de reentrenar.

## Endpoints

| Método | Ruta | Descripción | Éxito | Errores |
|---|---|---|---|---|
| GET | `/health` | Comprobación de vida (liveness) | 200 | - |
| POST | `/api/v1/predict` | Predice la especie y persiste el registro | 201 | 422 |
| GET | `/api/v1/predictions` | Lista registros, los más recientes primero (`skip`, `limit`) | 200 | 422 |
| GET | `/api/v1/predictions/{id}` | Lee un registro | 200 | 404 |
| PUT | `/api/v1/predictions/{id}` | Reemplaza las medidas, vuelve a predecir y actualiza el registro | 200 | 404, 422 |
| DELETE | `/api/v1/predictions/{id}` | Elimina un registro | 204 | 404 |

Swagger UI: `/docs` · ReDoc: `/redoc` · Esquema OpenAPI: `/openapi.json`

### Probar con PowerShell

```powershell
$body = @{
    sepal_length = 5.1
    sepal_width  = 3.5
    petal_length = 1.4
    petal_width  = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri http://127.0.0.1:8000/api/v1/predict `
    -ContentType "application/json" `
    -Body $body
```

### Probar con curl (bash)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Respuesta de ejemplo:

```json
{
  "id": 1,
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2,
  "predicted_class": 0,
  "predicted_label": "setosa",
  "confidence": 0.981,
  "created_at": "2026-01-01T12:00:00+00:00",
  "updated_at": "2026-01-01T12:00:00+00:00"
}
```

## Cómo funciona

| Pieza | Decisión |
|---|---|
| Modelo | `StandardScaler` + `LogisticRegression(max_iter=200)` sobre el dataset Iris. División entrenamiento/test 80/20 con `random_state=42`. La precisión de test se registra en el log al iniciar. |
| Inferencia | `predict_proba` devuelve la clase ganadora y su probabilidad como `confidence`. |
| Persistencia | `data/predictions.json` con `{"next_id": N, "items": [...]}`. Un `threading.Lock` global del proceso serializa los ciclos de lectura-modificación-escritura y las escrituras son atómicas (archivo temporal + `os.replace`). |
| Composición de la app | `create_app()` es una fábrica; el modelo y el repositorio viven en `app.state` y se inyectan mediante dependencias de FastAPI. Los tests inyectan rutas temporales. |
| Contrato de la API | Los modelos Pydantic en `app/schemas.py` incluyen descripciones de campos, rangos y ejemplos para Swagger. |
| Errores | Los registros inexistentes devuelven `404` con `{"detail": "Predicción <id> no encontrada"}`. La entrada inválida devuelve el cuerpo `422` estándar de FastAPI. |

### Registro almacenado

```json
{
  "next_id": 2,
  "items": [
    {
      "id": 1,
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2,
      "predicted_class": 0,
      "predicted_label": "setosa",
      "confidence": 0.981,
      "created_at": "2026-01-01T12:00:00+00:00",
      "updated_at": "2026-01-01T12:00:00+00:00"
    }
  ]
}
```

## Estructura del proyecto

```
APIPUBLIC/
├── app/
│   ├── main.py            # create_app(), lifespan, metadatos de Swagger, /health
│   ├── config.py          # rutas por defecto de datos/modelo y variables de entorno
│   ├── schemas.py         # modelos Pydantic + ejemplos para Swagger
│   ├── model.py           # IrisModel: entrenar, cargar, predecir
│   ├── repository.py      # CRUD en JSON con lock + escrituras atómicas
│   ├── dependencies.py    # get_model / get_repository
│   └── routers/
│       └── predictions.py # rutas de predicción + CRUD
├── tests/test_api.py
├── data/                  # predictions.json versionado con DVC (.dvc en git)
├── models/                # iris_model.joblib versionado con DVC (.dvc en git)
├── requirements.txt       # dependencias pineadas con ==
├── requirements-dev.txt   # pytest, httpx y DVC pineados
├── pytest.ini
└── README.md
```

## Reproducibilidad y versionado de datos

### Entorno reproducible

| Pieza | Detalle |
|---|---|
| Python | 3.14.5 (probado en 3.14; requiere 3.10+) |
| Dependencias | `requirements.txt` pineado con `==` a las versiones exactas instaladas: fastapi 0.162.0, uvicorn 0.54.0, scikit-learn 1.9.1, joblib 1.6.0 |
| Dependencias dev | `requirements-dev.txt` agrega pytest 8.4.2, httpx 0.28.1 y DVC 3.67.1, también pineados |
| Semilla | `train_test_split(..., random_state=42)` y `LogisticRegression` determinístico: mismo entrenamiento ⇒ mismo modelo |
| Artefactos ignorados en git | `data/predictions.json` y `models/iris_model.joblib` se versionan con DVC, no con git |

Para recrear el entorno exacto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Versionado de datos con DVC

El repositorio usa [DVC](https://dvc.org) para versionar los archivos grandes junto a git. Git guarda solo los punteros (`*.dvc`); el contenido vive en el remoto DVC configurado.

| Archivo | Versionado |
|---|---|
| `data/predictions.json` | `data/predictions.json.dvc` |
| `models/iris_model.joblib` | `models/iris_model.joblib.dvc` |

El remoto DVC por defecto (`storage`) apunta a `../dvc-storage` (carpeta local fuera del repo). Para usar un remoto remoto real (S3, GCS, SSH, etc.):

```powershell
dvc remote add -d storage s3://mi-bucket/apipublic
dvc push
```

Flujo habitual después de clonar:

```powershell
git clone <repo>
pip install -r requirements-dev.txt
dvc pull          # descarga data/predictions.json y models/iris_model.joblib
```

Si cambiás un dato o reentrenás el modelo y querés versionar la nueva versión:

```powershell
dvc add data/predictions.json models/iris_model.joblib
git add data/predictions.json.dvc models/iris_model.joblib.dvc data/.gitignore models/.gitignore
git commit -m "chore: versionar datos y modelo"
dvc push
```

Nota: `data/predictions.json` es estado de la aplicación en tiempo de ejecución; versiona solo cuando quieras congelar un snapshot intencional. El dataset Iris proviene de `sklearn.datasets.load_iris()` y se reproduce idéntico con las versiones pineadas de scikit-learn.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

La suite cubre health, predicción, listado, get/404, actualización con re-predicción, delete/404, validación de entrada y el documento OpenAPI. Cada test usa un archivo de datos y un modelo temporales, por lo que los directorios reales `data/` y `models/` nunca se modifican.

## Configuración

| Variable | Valor por defecto | Propósito |
|---|---|---|
| `DATA_DIR` | `./data` | Directorio para `predictions.json`. |
| `MODELS_DIR` | `./models` | Directorio para `iris_model.joblib`. |

## Despliegue

Arranque mínimo estilo producción (sin recarga):

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Para Docker o un proveedor cloud, se puede agregar un `Dockerfile` que instale `requirements.txt` y ejecute el comando anterior. Eso queda intencionalmente fuera de alcance para esta configuración mínima.

## Solución de problemas

| Síntoma | Solución |
|---|---|
| `ModuleNotFoundError: fastapi` | Active `.venv` y ejecute `pip install -r requirements.txt`. |
| `AttributeError` con type hints modernos | Use Python 3.10+; este proyecto está probado con 3.14. |
| Puerto en uso | Ejecute `uvicorn app.main:app --port 8001`. |
| Activación bloqueada en PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
| Quiere un historial limpio | Detenga el servidor y elimine `data/predictions.json`. |

## Lista de verificación

- [ ] `uvicorn app.main:app --reload` arranca sin errores.
- [ ] `/docs` muestra todos los endpoints con ejemplos.
- [ ] `POST /api/v1/predict` devuelve una etiqueta y una confianza.
- [ ] Un segundo arranque carga el modelo guardado en lugar de reentrenar.
- [ ] `python -m pytest` pasa.
