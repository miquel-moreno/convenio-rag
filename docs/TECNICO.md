# Detalles técnicos

## Ponerlo en marcha

```bash
cp .env.example .env   # elige proveedor de LLM y añade tu clave
docker compose up --build
```

La página de demo queda en http://localhost:8000 y la API en `/docs`. **La primera vez tarda unos 2 minutos:** el contenedor aplica las migraciones, descarga el modelo de embeddings (~250 MB, se guarda en un volumen) y carga los dos convenios (572 fragmentos). En los siguientes arranques detecta que ya están cargados y arranca en segundos.

Para las respuestas hace falta un LLM en `.env`: `LLM_PROVIDER=openai` + `OPENAI_API_KEY`, o `LLM_PROVIDER=ollama` con Ollama en la máquina anfitriona. La búsqueda (`/search`) funciona sin LLM.

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

## Búsqueda

`GET /search?q=...&mode=hybrid|vector|text&limit=5&agreement=BOE-A-...` devuelve los artículos más relevantes (uno por artículo) con su posición en cada buscador.

- **Texto completo (PostgreSQL, configuración `spanish`):** raíces de palabras ("vacación" encuentra "vacaciones") y sin palabras vacías. Se busca con OR de las palabras de la pregunta y se ordena con `ts_rank_cd`; índice GIN sobre `to_tsvector('spanish', ref || ' ' || title || ' ' || text)`.
- **Semántica (pgvector):** embeddings de 384 dimensiones con `paraphrase-multilingual-MiniLM-L12-v2` (fastembed/ONNX, en local, ~250 MB, se descarga la primera vez) y distancia coseno con índice HNSW.
- **Fusión RRF** (*Reciprocal Rank Fusion*, k = 60) de los 20 primeros de cada buscador.

Por qué dos buscadores: en una prueba con frases de ejemplo, la búsqueda semántica relacionó "¿me pagan más si trabajo de noche?" con un texto sobre el "plus de nocturnidad", con el que no comparte ninguna palabra; la de texto no puede hacerlo. A la inversa, la de texto es más fiable con términos exactos.

## Preguntar: `POST /ask`

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json"      -d '{"question": "¿Cuántos días de vacaciones tengo?", "agreement": "BOE-A-2025-7766"}'
```

Devuelve `found`, `answer` (1–3 frases), `citations` (convenio, artículo, título y enlace oficial al BOE) y `sources` (los artículos que se dieron al modelo).

1. Búsqueda híbrida → los 5 artículos más relevantes (texto completo de cada uno, hasta 4000 caracteres).
2. Se dan al LLM como fuentes numeradas `[1]…[5]` con reglas estrictas: solo esas fuentes, citar por número, decir si la respuesta no está. Salida estructurada estricta `{found, answer, citations}`, con un reintento si no es válida.
3. **Verificación:** cada cita debe ser una de las fuentes dadas; las inventadas se descartan y una respuesta sin ninguna cita válida se convierte en "no encontrado".

## Página de demo

`http://localhost:8000/` sirve una página mínima (un solo HTML sin dependencias, `src/convenio_rag/static/index.html`) con un selector de convenio, ejemplos de preguntas y la respuesta con sus citas y enlaces al BOE. Solo llama a `POST /ask`. El GIF del README se regenera con `uv run python -m scripts.record_demo --url ...` (Playwright con el Edge instalado; necesita el servicio en marcha y un LLM configurado).

## Arquitectura

```
src/convenio_rag/
├── api/        # rutas HTTP (FastAPI) y middleware
├── core/       # configuración, logging JSON, errores
├── services/   # lógica de negocio, sin red: troceado de convenios, carga
└── adapters/   # LLM, embeddings, base de datos y BOE, detrás de interfaces
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
| Búsqueda híbrida (texto + semántica) fusionada con RRF | Cada buscador falla en casos distintos: el de texto no entiende sinónimos ("noche" / "nocturnidad"); el semántico puede fallar con términos exactos ("artículo 21", "IT"). RRF solo usa posiciones, así que combina rankings con puntuaciones no comparables sin ajustar pesos |
| Embeddings en local con fastembed (ONNX) y un modelo ligero | Sin coste por consulta ni claves, y sin PyTorch (más de 1 GB). Detrás de una interfaz: cambiar de modelo es cambiar `EMBEDDING_MODEL` (y la dimensión en una migración) |
| OR de palabras en la búsqueda de texto | Con AND, una pregunta ("¿cuántos días de vacaciones tengo?") exigiría palabras que el artículo no contiene ("tengo") y no devolvería nada. El ranking premia los fragmentos con más coincidencias |
| Un resultado por artículo | Los artículos largos están partidos en varios fragmentos; sin agruparlos, el mismo artículo ocuparía varios puestos del top |
| pgvector en PostgreSQL, no una base de datos vectorial aparte | Un solo sistema que ya se usa para los datos, con transacciones, y las dos búsquedas en la misma consulta. Para decenas de miles de fragmentos es de sobra |
| Tests de búsqueda contra PostgreSQL real (servicio en la CI) | SQLite no tiene búsqueda de texto en español ni pgvector. Las pruebas `tests/pg` se ejecutan en la CI con un contenedor `pgvector/pgvector:pg16` y en local con `TEST_DATABASE_URL` |
| Índice de texto con `\|\|` y no `concat_ws` | PostgreSQL solo acepta funciones IMMUTABLE en un índice por expresión; `concat_ws` no lo es |
| El LLM cita fuentes numeradas y el código verifica las citas | El modelo solo puede citar lo que se le dio; si cita un número que no existe o no cita nada, la respuesta no se da por buena. Así "no inventa" no depende solo de las instrucciones |
| El artículo completo como contexto, no solo el fragmento encontrado | El fragmento que mejor casa puede ser la segunda parte de un artículo cuya condición está en la primera |
| "No lo regula" también es una respuesta | Si el convenio remite a otro convenio o a la ley, el modelo debe decirlo y citar ese artículo, en lugar de dar un número de memoria |

## Evaluación

```bash
make eval ARGS="--retrieval-only"                          # solo el buscador, sin LLM, gratis
make eval ARGS="--provider openai --model gpt-4.1-mini"    # buscador + respuestas
```

**Conjunto:** 30 preguntas escritas a mano (18 del convenio de consultoría y 12 del metal), cada una con el artículo esperado y el dato que debe aparecer en la respuesta, comprobados en el texto oficial; más **5 preguntas trampa** cuya respuesta no está en el convenio (coche de empresa, gimnasio, seguro de vida, guardería, bonus). Dos preguntas del metal tienen como respuesta correcta una remisión (el convenio estatal lo deja a los convenios de ámbito inferior). Todo en `evals/questions.json`.

**Recuperación** (sin LLM; ¿está el artículo esperado entre los 5 primeros?), 28/09/2026, modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2`:

| Búsqueda | hit@5 | MRR |
|---|---|---|
| Solo texto (full-text en español) | 76,7 % (23/30) | 0,623 |
| Solo semántica (pgvector) | 86,7 % (26/30) | 0,808 |
| **Híbrida (RRF)** | **96,7 % (29/30)** | **0,811** |

En una pregunta (desconexión digital, metal) ninguno de los dos buscadores ponía el artículo en su top 5, pero ambos lo tenían cerca y la fusión sí lo encuentra.

**Respuestas** (`gpt-4.1-mini`, 28/09/2026). "Correcta" = cita el artículo esperado **y** contiene el dato esperado:

| | Híbrida | Solo semántica |
|---|---|---|
| Respuestas correctas | **29 / 30 (96,7 %)** | 26 / 30 (86,7 %) |
| **Respuestas equivocadas dadas como buenas** | **0** | 2 |
| "No lo he encontrado" en preguntas con respuesta | 1 | 2 |
| Preguntas trampa rechazadas | 5 / 5 | 5 / 5 |
| Tiempo medio por pregunta | 1,3 s | 1,5 s |
| Coste (35 preguntas) | $0,044 | $0,044 |

- El único fallo de la híbrida es un "no lo he encontrado" honesto: "¿cuál es el salario mínimo en el metal?" (el art. 50 se titula "Salario del sector" y lo remite al convenio aplicable; el buscador no lo trae).
- Con solo búsqueda semántica aparecen respuestas equivocadas: cuando las fuentes recuperadas no son las correctas, el modelo responde con otra fuente real. La verificación de citas evita citas **inventadas**, no el uso de una fuente **equivocada**: contra eso, la defensa es una recuperación mejor.
- Los resultados de la híbrida fueron idénticos en dos ejecuciones; los de solo semántica variaron (25 y 26 de 30). Resultados completos, pregunta a pregunta, en `evals/results/`.

## Limitaciones

- **La recuperación depende de cómo se redacte la pregunta.** Ejemplo real: "¿Cuántas horas al año se trabajan en el metal?" trae el art. 46 (duración de la jornada) y la respuesta explica la remisión a los convenios de ámbito inferior; "¿Cuántas horas al año se trabajan?" no lo trae entre los 5 primeros (el artículo habla de "jornada" y "cómputo anual") y la respuesta es un "no lo he encontrado" correcto pero poco útil. Mejoras posibles: reescribir la pregunta con el LLM antes de buscar, o dar más artículos al modelo; habría que medirlas con la evaluación.
- **El convenio estatal del metal remite muchas condiciones a los convenios provinciales** (por ejemplo, la jornada anual, art. 46, o pluses como la nocturnidad). Para esas preguntas la respuesta correcta es "este convenio no lo fija", no un número.
- Sin números de página en las citas (el XML oficial no los trae).
- No incluye las publicaciones posteriores que modifican los convenios (tablas salariales de 2026 de consultoría, modificación de 2022 del metal).
