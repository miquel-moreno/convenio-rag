# Detalles técnicos

## Ponerlo en marcha

```bash
cp .env.example .env   # elige proveedor de LLM y añade tu clave
docker compose up --build
```

La API queda en http://localhost:8000 (documentación interactiva en `/docs`). Al arrancar, el contenedor aplica las migraciones pendientes.

Desarrollo local:

```bash
make install                    # dependencias + hooks de pre-commit
make dev                        # API con recarga automática
make check                      # lint + tipos + tests
make eval                       # evaluación con un LLM real (solo en local)
uv run alembic upgrade head     # crea o actualiza las tablas (usa DATABASE_URL)
uv run python -m scripts.ingest # carga los convenios de data/boe/
```

## Datos: los convenios

| Convenio | BOE | Estado (28/09/2026) |
|---|---|---|
| XIX Convenio estatal de consultoría, TI y estudios de mercado | [BOE-A-2025-7766](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-7766) | Vigente hasta el 31/12/2027, prorrogable. Sus tablas salariales se actualizaron en BOE-A-2026-9024 (no incluido) |
| IV Convenio estatal del metal | [BOE-A-2022-479](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2022-479) | Vigencia inicial hasta el 31/12/2023; **sigue aplicándose hasta que lo sustituya otro (ultraactividad)**. Modificado en BOE-A-2022-7391 (no incluido) |

```bash
uv run python -m scripts.ingest              # carga data/boe/*.xml en la base de datos
uv run python -m scripts.ingest --refresh    # los vuelve a descargar del BOE antes
```

El texto oficial (XML) está versionado en `data/boe/`: la carga y la evaluación se pueden repetir sin conexión y siempre sobre el mismo texto. Resultado de la carga: 142 fragmentos (consultoría) y 430 (metal); una segunda carga no duplica nada.

## Arquitectura

```
src/convenio_rag/
├── api/        # rutas HTTP (FastAPI) y middleware
├── core/       # configuración, logging JSON, errores
├── services/   # lógica de negocio, sin red: troceado de convenios, carga
└── adapters/   # LLM, base de datos y BOE, detrás de interfaces
```

## Decisiones técnicas

| Decisión | Por qué |
|---|---|
| LLM detrás de una interfaz propia (`LLMClient`) | Cambiar de proveedor (OpenAI, Ollama local) es cambiar una variable de entorno. Los tests usan un cliente falso: la CI no gasta dinero ni necesita claves |
| Cargar desde el XML oficial del BOE, no desde el PDF | El XML marca cada encabezado (`<p class="articulo">`), capítulos y anexos: no hay que adivinar dónde empieza un artículo. Contrapartida: no trae números de página, así que las citas son convenio + artículo + enlace oficial |
| Un fragmento por artículo, disposición o anexo | Es la unidad que se cita y que entiende quien lee el convenio. Los largos se parten por párrafos en trozos de ≤ 1500 caracteres, que caben en el modelo de embeddings (~512 tokens) |
| Los artículos de reglamentos dentro de un anexo se citan como "Anexo VII · Artículo 6" | Evita que "Artículo 6" apunte a dos textos distintos: el convenio del metal incluye reglamentos con su propia numeración |
| Las leyes citadas entre comillas («Artículo 25…») se quedan dentro de su artículo | El XML las marca como encabezados; tratarlas como artículos nuevos partía el artículo que las cita |
| Las tablas (salarios) se convierten en texto `celda \| celda` | Preguntas como "¿cuánto cobra un programador?" tienen la respuesta en las tablas de los anexos |

## Evaluación

_Pendiente._ Resultados en `evals/results/`, con fecha y modelo.

## Limitaciones

- Sin números de página en las citas (el XML oficial no los trae).
- No incluye las publicaciones posteriores que modifican los convenios (tablas salariales de 2026 de consultoría, modificación de 2022 del metal).
