# Система ассетов: дизайн (старт — меши)

Рабочий документ. Цель — заменить «привязку по эфемерному имени» настоящей
идентичностью ассетов с загрузкой через граф кадра, дедупликацией, перезагрузкой
при изменении на диске и устойчивой сериализацией. Начинаем с мешей, модель
проектируем обобщаемой на текстуры/материалы.

Связано с [PREFABS.md](PREFABS.md) #4. Архитектурное правило аплоада —
см. память `gpu-upload-through-frame-graph`.

---

## 1. Текущее состояние и дыры

- Меш в ECS — `Arc<Mesh>` (GPU-хэндл) в `Primitive`. GPU-`Mesh`
  ([data.rs:61](../graphics/src/mesh/data.rs#L61)) держит только `Arc<Buffer>` +
  метаданные — **CPU-геометрии нет**. Графика гарантирует: `Arc<Mesh>` в компоненте
  **безопасно дропать в любой момент** (отложенное GPU-уничтожение).
- «Имя» = `Mesh::label()`, ключ в `MeshManager: HashMap<String, Arc<Mesh>>`.
- Загрузка GPU — **синхронная**: `create_mesh_from_cpu` + `device.write_buffer`
  ([device.rs:637](../graphics/src/device.rs#L637)), **мимо графа кадра**.

**Проблемы:** имя ≠ идентичность; двойное создание куба; round-trip на ручной
регистрации; нет связи с диском (mount/path/хэш/reload); **аплоад мимо графа**
(нарушает инвариант → синхронизационные грабли). `MeshManager` — это и есть
«тупой менеджер»; других нет.

---

## 2. Цели

- **Идентичность**, переживающая изменение содержимого файла.
- **Один источник истины** на ассет (дедуп, без двойного создания).
- **Аплоад на GPU только через `TransferPass` графа кадра.** Прямой/синхронный
  путь — **удалить** (`create_mesh_from_cpu` для ассетов).
- **Устойчивая сериализация**: пишем *источник*, не имя.
- **Reload** при изменении файла (`fs_watcher`).
- Без второго счётчика ссылок — переиспользуем существующий `Arc<Mesh>`.

---

## 3. Модель

Два понятия: **Source** (откуда; сериализуется; ключ идентичности) и **Assets**
(менеджер: владеет/кэширует `Arc<Mesh>`, стейджит, заливает через граф).
**Хэндлов и второго refcount нет** — лайфтайм держит сам `Arc<Mesh>`.

### 3.1 Source
```rust
enum MeshSource {
    /// Файл: пара (mount, path) — пути не уникальны между VFS — + индекс примитива.
    File { mount: String, path: String, primitive: u32 },
    Generated(MeshGenerator),   // Cube{half}, Sphere{r,seg,rings}, Quad{..}
    Embedded(Arc<CpuMesh>),     // геометрия запечена в префаб/сцену
}
```
`AssetId` детерминированно из `Source` (`File`→`(mount,path,primitive)`;
`Generated`→параметры; `Embedded`→контент-хэш).

> Развилка (позже): `(mount,path)` сейчас vs GUID/.meta (переживает переименование).

### 3.2 Компонент хранит Source, не Arc-напрямую
`Primitive { source: MeshSource, mesh: Option<Arc<Mesh>>, aabb }`:
- `source` — для сериализации **и** для резолва/reload (матчинг по `AssetId`);
- `mesh` — резолвится из `Assets`; `None` пока ассет не резидентен (см. §3.4).

Это не «числовой handle» (никакого ручного лайфтайма) и не «голый `Arc` без
источника» (иначе reload некуда применить). Лайфтайм — обычный `Arc<Mesh>`,
один счётчик.

### 3.3 Assets — менеджер (эволюция `MeshManager`)
```rust
struct MeshAssets {
    device: Arc<GraphicsDevice>,
    cache: HashMap<AssetId, MeshAsset>,
}
struct MeshAsset { source: MeshSource, state: MeshState, gpu: Option<Arc<Mesh>> }
enum MeshState { Reading, Staged(CpuMesh), Uploading, Resident, Failed }
impl MeshAssets {
    fn load(&mut self, source: MeshSource) -> AssetId;     // дедуп; ставит в очередь
    fn get(&self, id: AssetId) -> Option<&Arc<Mesh>>;      // Some только если Resident
    fn flush_uploads(&mut self, graph: &mut RenderGraph);  // §3.4
    fn invalidate(&mut self, id: AssetId);                 // reload (§3.5)
}
```
Менеджер держит **strong** `Arc<Mesh>` в кэше (живёт до выгрузки сцены/explicit
unload); компоненты держат тот же `Arc`. Один счётчик; дроп безопасен в любой
момент (гарантия графики). Дедуп по `AssetId`.

### 3.4 Загрузка — двухстадийная, GPU-аплоад через граф
1. **VFS read** (можно async, `background_vfs`) → парс в `CpuMesh` → `Staged`.
2. **GPU upload — ТОЛЬКО через граф.** Раз в кадр при сборке графа менеджер зовёт
   `flush_uploads(graph)`: для каждого `Staged` создаёт device-local буферы +
   staging и добавляет `TransferPass` (staging→device-local) в **тот же**
   per-frame `RenderGraph`. Существующих copy-операций `TransferPass` достаточно —
   **новый API не нужен**. После исполнения графа → `Resident`.

**Нет гарантии «тот же кадр».** Ассет, запрошенный в кадре N, становится
резидентным в N+1 (или позже). Кому нужно кадр-в-кадр — обязан предсказать/
запрефетчить или ждать ≥1 кадр. Отрисовка **пропускает** не-`Resident` примитивы
(`mesh == None`).

### 3.5 Reload
`fs_watcher` → `AssetId` по `(mount,path)` → `invalidate(id)` → перечитать →
`Staged` → следующий `flush_uploads` зальёт **новый** `Arc<Mesh>` в кэш. Система
резолва (§5) подменяет `Primitive.mesh` у компонентов с этим `source`. Старый
`Arc` дропается, как только его никто не держит (безопасно).

---

## 4. Сериализация

`Primitive` сериализует `source`:
```
{ mesh:{ kind:"file", mount:"assets", path:"props/barrel.glb", primitive:0 }, material:{..} }
{ mesh:{ kind:"generated", gen:{ cube:{ half:0.5 } } }, material:{..} }
{ mesh:{ kind:"embedded", data:<cpu mesh> }, material:{..} }
```
Десериализация: `assets.load(source)`; `Primitive.mesh = None` до резидентности.

---

## 5. Интеграция

- **Резолв-система** (exclusive, рядом с/вместо части `InitializeRenderEntities`):
  для `Primitive` с `mesh==None` или после `invalidate` — берёт `assets.get(id)`,
  ставит `Some(arc)` когда резидентно; помечает `Changed`.
- **Кадр:** `assets.flush_uploads(graph)` при сборке графа; рендер-системы рисуют
  только примитивы с `Some(mesh)`.
- **Прямой спавн:** `assets.load(source)` вместо двойного создания.
- `MeshManager` → `MeshAssets`. **Удалить** прямой `create_mesh_from_cpu`-путь для
  ассетов (оставить разве что для тестов/внутренних нужд, без графа).

---

## 6. Обобщение на текстуры/материалы

Та же пара Source/Assets с общим `flush_uploads(graph)`. `TextureManager`/
`MaterialManager` страдают тем же (имя-ключ, sync-upload). Сначала довести на
мешах, затем вынести общий каркас `Assets<T>`/`Source<T>`.

---

## 7. Решения (зафиксированы)

| Вопрос | Решение |
|---|---|
| Идентичность файлов | `(mount, path)` сейчас; GUID/.meta позже |
| Ссылка из компонента | `MeshSource` + резолвимый `Option<Arc<Mesh>>` |
| Второй refcount/handle | **Нет.** Переиспользуем `Arc<Mesh>` (один счётчик) |
| GPU upload | только `TransferPass` графа кадра; прямой путь удалить |
| VFS read | можно async (`background_vfs`), отдельно от аплоада |
| Residency | не в тот же кадр; ≥1 кадр; рендер пропускает не-resident |
| CPU-данные меша (коллизии) | отложить; меш GPU-only |
| `MeshManager` | эволюция в `MeshAssets` |

---

## 8. Фазы реализации

1. **`MeshAssets` + `MeshSource`** поверх существующих `TransferPass`-copy:
   кэш по `AssetId`, `flush_uploads(graph)`, состояния. `Primitive { source,
   mesh: Option<Arc<Mesh>> }`. Резолв-система. **Удалить** прямой аплоад-путь.
   Убрать двойное создание; отрисовка пропускает не-resident.
2. **Сериализация `MeshSource`** в `MeshRenderer` (замена имени).
3. **Reload** по `fs_watcher` (`invalidate` + резолв-подмена).
4. **glTF → спавн** через `Assets`.
5. **Обобщение** на текстуры/материалы.

---

## 9. `AssetRef` в компонентах + хот-релоад

Ссылка на ассет в данных компонента — генерик-поле `AssetRef<S>` (крейт
`redlilium-assets`): `source: S` (идентичность, сериализуется) + кэшированный
резолв `Option<Arc<S::Asset>>` (рантайм). Трейт `AssetRefSource` привязывает
source к продукту **менеджера** (`MeshSource → Mesh`,
`MaterialInstanceSource → ResolvedInstance`) — то, что видит потребитель, а не
сырой выход лоадера.

Принципы:

- **Чтение — доступ к полю.** Рендер читает `Arc` прямо из компонента: без
  локов, без обращения к мапе. Батчинг — по ptr-eq, как везде.
- **Вся мутация — через sync-систему** (`MeshLoad`): demand-driven (неразрешённый
  source запрашивается у менеджера — конструирование/десериализация компонента
  менеджеры не трогает) и через `Mut`, т.е. видима ECS: ре-резолв помечает
  компонент dirty, stateful-потребители (физика, навмеш) реагируют штатным
  change-detection.
- **Менеджеры — единственный источник истины** (resident-мапа + `failed`-set,
  чтобы битый ассет не перезапрашивался каждый кадр) и единственный requester у
  процессора.
- **Версия ассета = идентичность `Arc`.** Хот-релоад: менеджер публикует новый
  `Arc` в resident + `generation += 1`; sync-система сверяет `is_current`
  (ptr-eq) и переписывает устаревшие ref'ы. Производные кэши (пайплайны,
  коллайдеры) сверяют `Arc`-входы тем же ptr-eq — pull-валидация, без
  реверс-индекса зависимостей и без событийной шины.
- Внутренние зависимости менеджеров (instance → material → shader,
  `PipelineCache`) валидируются так же — ptr-eq входов в `drive`.

План хот-релоада поверх этого: `ChangedAssets` (правки инспектора + `FsWatcher`
→ guid по DB) → `manager.invalidate(guid)` → перезагрузка → новый `Arc` в
resident → sync-система и pull-валидация разносят обновление. Универсально для
любого типа ресурса; для пользовательских компонентов derive позже сможет
генерить обход `AssetRef`-полей (одна генерик-система на все компоненты).

---

## Статус
Ревизия 5. Фазы 1–2 и §9 (`AssetRef` + sync-система) реализованы: меши,
материал-инстансы и текстуры биндятся через `AssetRef`, менеджеры — источник
истины с `generation`. Хот-релоад работает по всей цепочке (`ChangedAssets` →
`invalidate` владеющего менеджера → pull-валидация разносит обновление).
Текстуры — ассеты: `TextureSource::File(guid) | Solid(rgba)` (Solid — дефолты
слотов схемы), `PropValue::Texture` в свойствах материала; инстанс-менеджер
резолвит текстурные свойства и биндит их в group 1 (binding 0 — упакованные
uniform-байты, дальше пары texture/sampler в порядке схемы).
