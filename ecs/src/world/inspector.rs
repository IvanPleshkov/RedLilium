use std::collections::HashMap;

use crate::entity::Entity;
use crate::sparse_set::{
    ComponentMeta, DeserializeComponentFn, InspectResult, SerializeComponentFn,
};

use super::World;

impl World {
    // ---- Component meta helpers ----

    /// Looks up component meta by name.
    fn meta_by_name(&self, name: &str) -> Option<&ComponentMeta> {
        let type_id = self.name_index.get(name)?;
        let lock = self.components.get(type_id)?;
        let storage = unsafe { &*lock.data_ptr() };
        storage.meta()
    }

    /// Iterates over all registered component metas.
    fn iter_meta(&self) -> impl Iterator<Item = &ComponentMeta> {
        self.components.values().filter_map(|lock| {
            let storage = unsafe { &*lock.data_ptr() };
            storage.meta()
        })
    }

    /// Mutates `T`'s meta in place; returns false when `T` has no meta
    /// (never registered through `register_inspector`).
    pub(crate) fn with_component_meta_mut<T: 'static>(
        &mut self,
        f: impl FnOnce(&mut ComponentMeta),
    ) -> bool {
        let Some(lock) = self.components.get_mut(&std::any::TypeId::of::<T>()) else {
            return false;
        };
        match lock.get_mut().meta.as_mut() {
            Some(meta) => {
                f(meta);
                true
            }
            None => false,
        }
    }

    /// All components that registered gizmo-anchor providers.
    pub(crate) fn gizmo_anchor_providers(
        &self,
    ) -> Vec<(&'static str, crate::sparse_set::GizmoAnchorsFn)> {
        self.iter_meta()
            .filter_map(|m| m.gizmo_anchors_fn.map(|f| (m.name, f)))
            .collect()
    }

    /// The drag provider for one component, by name.
    pub(crate) fn gizmo_drag_provider(
        &self,
        component: &str,
    ) -> Option<crate::sparse_set::GizmoDragFn> {
        let type_id = self.name_index.get(component)?;
        let lock = self.components.get(type_id)?;
        let storage = unsafe { &*lock.data_ptr() };
        storage.meta().and_then(|m| m.gizmo_drag_fn)
    }

    // ---- Asset references (generic sync / hot reload) ----

    /// Scan the asset references of every instance of every inspector-registered
    /// component type (read-only). The callback receives
    /// `(component_name, entity_index, &AssetRef<S> as &dyn Any)`; downcast to
    /// the concrete `AssetRef<S>` types you have resolvers for. Components
    /// without asset refs contribute nothing.
    ///
    /// `since` gates the walk by change detection: `Some(tick)` visits only
    /// components changed (or inserted) after that tick — the caller's cheap
    /// path when its resolvers' resident sets are unchanged. `None` visits
    /// everything.
    pub fn scan_asset_refs(
        &self,
        since: Option<u64>,
        f: &mut dyn FnMut(&'static str, u32, &dyn std::any::Any),
    ) {
        for meta in self.iter_meta() {
            let name = meta.name;
            (meta.scan_asset_refs_fn)(self, since, &mut |idx, r| f(name, idx, r));
        }
    }

    /// Re-visit one component instance's asset references mutably so the
    /// callback can apply re-resolutions (marks the component changed). The
    /// `(component, entity_index)` pair comes from
    /// [`scan_asset_refs`](Self::scan_asset_refs).
    #[cfg_attr(not(feature = "rendering"), allow(dead_code))]
    pub(crate) fn patch_asset_refs(
        &self,
        component: &str,
        entity_index: u32,
        f: &mut dyn FnMut(&mut dyn std::any::Any),
    ) {
        if let Some(meta) = self.meta_by_name(component) {
            (meta.patch_asset_refs_fn)(self, entity_index, f);
        }
    }

    // ---- Inspector ----

    /// Returns the names of all inspector-registered components that an entity has.
    ///
    /// Only includes components registered via [`register_inspector`](World::register_inspector)
    /// or [`register_inspector_default`](World::register_inspector_default).
    pub fn inspectable_components_of(&self, entity: Entity) -> Vec<&'static str> {
        let mut entries: Vec<_> = self
            .iter_meta()
            .filter(|m| (m.has_fn)(self, entity))
            .map(|m| (m.name, m.display_order))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        entries.into_iter().map(|(name, _)| name).collect()
    }

    /// Returns the type names of ALL components attached to an entity.
    ///
    /// Unlike [`inspectable_components_of`](World::inspectable_components_of), this
    /// includes components that were not registered via
    /// [`register_inspector`](World::register_inspector). The returned names are
    /// fully-qualified Rust type paths (from [`std::any::type_name`]).
    pub fn all_component_names_of(&self, entity: Entity) -> Vec<&'static str> {
        self.components
            .values()
            .filter_map(|lock| {
                let storage = lock.read();
                if storage.contains_untyped(entity.index()) {
                    Some(storage.type_name())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns component names that the entity does NOT have and that support Default insertion.
    pub fn addable_components_of(&self, entity: Entity) -> Vec<&'static str> {
        self.iter_meta()
            .filter(|m| m.insert_default_fn.is_some() && !(m.has_fn)(self, entity))
            .map(|m| m.name)
            .collect()
    }

    /// Renders the inspector UI for a component by name.
    ///
    /// Returns `Some(actions)` if the user edited any fields, or `None` if
    /// nothing changed or the entity doesn't have the component.
    pub fn inspect_by_name(&self, entity: Entity, name: &str, ui: &mut egui::Ui) -> InspectResult {
        let inspect_fn = self.meta_by_name(name).map(|m| m.inspect_fn);
        inspect_fn.and_then(|f| f(self, entity, ui))
    }

    /// Removes a component by name from an entity.
    ///
    /// Returns `true` if the component was removed.
    pub fn remove_by_name(&mut self, entity: Entity, name: &str) -> bool {
        let remove_fn = self.meta_by_name(name).map(|m| m.remove_fn);
        if let Some(f) = remove_fn {
            f(self, entity)
        } else {
            false
        }
    }

    /// Inserts a default instance of a component by name on an entity.
    ///
    /// Does nothing if the component was not registered with Default support
    /// or the name is unknown.
    pub fn insert_default_by_name(&mut self, entity: Entity, name: &str) {
        let insert_fn = self.meta_by_name(name).and_then(|m| m.insert_default_fn);
        if let Some(f) = insert_fn {
            f(self, entity);
        }
    }

    /// Extracts a type-erased clone of a component by name from an entity.
    ///
    /// Returns `None` if the component is not present or the name is unknown.
    pub(crate) fn extract_by_name(
        &self,
        entity: Entity,
        name: &str,
    ) -> Option<Box<dyn crate::prefab::ComponentBag>> {
        let extract_fn = self.meta_by_name(name).and_then(|m| m.extract_fn)?;
        extract_fn(self, entity)
    }

    /// Inserts a type-erased component from a [`ComponentBag`](crate::prefab::ComponentBag)
    /// onto an entity, consuming the bag.
    pub(crate) fn insert_bag(&mut self, entity: Entity, bag: Box<dyn crate::prefab::ComponentBag>) {
        bag.consume_into(self, entity);
    }

    /// Serializes a single component by name from an entity.
    ///
    /// Returns `Ok(None)` if the component is not present, the name is unknown,
    /// or the component doesn't support serialization.
    pub fn serialize_component_by_name(
        &self,
        entity: Entity,
        name: &str,
    ) -> Result<Option<crate::serialize::SerializedComponent>, crate::serialize::SerializeError>
    {
        let Some(meta) = self.meta_by_name(name) else {
            return Ok(None);
        };
        let mut ctx = crate::serialize::SerializeContext::new(self);
        (meta.serialize_fn)(self, entity, &mut ctx)
    }

    /// Deserializes a single component by name and inserts it onto an entity.
    ///
    /// Returns `Err` if the component type is unknown or deserialization fails.
    pub fn deserialize_component_by_name(
        &mut self,
        entity: Entity,
        serialized: &crate::serialize::SerializedComponent,
    ) -> Result<(), crate::serialize::DeserializeError> {
        let deser_fn = self
            .meta_by_name(serialized.type_name.as_str())
            .map(|m| m.deserialize_fn)
            .ok_or_else(|| crate::serialize::DeserializeError::UnknownComponent {
                type_name: serialized.type_name.clone(),
            })?;
        let mut ctx = crate::serialize::DeserializeContext::new(self);
        ctx.set_current_component(&serialized.type_name);
        deser_fn(entity, &serialized.data, &mut ctx)
    }

    /// Collects all entity references from a component by name on an entity.
    ///
    /// Appends referenced entities to `collector`. Does nothing if the
    /// component name is unknown or the entity doesn't have it.
    pub fn collect_entities_by_name(
        &self,
        entity: Entity,
        name: &str,
        collector: &mut Vec<Entity>,
    ) {
        if let Some(meta) = self.meta_by_name(name) {
            (meta.collect_entities_fn)(self, entity, collector);
        }
    }

    /// Remaps all entity references in a component by name on an entity.
    ///
    /// Does nothing if the component name is unknown or the entity doesn't have it.
    pub fn remap_entities_by_name(
        &mut self,
        entity: Entity,
        name: &str,
        map: &mut dyn FnMut(Entity) -> Entity,
    ) {
        let remap_fn = self.meta_by_name(name).map(|m| m.remap_entities_fn);
        if let Some(f) = remap_fn {
            f(self, entity, map);
        }
    }

    /// Computes the combined AABB for an entity by unioning AABBs from all its components.
    ///
    /// Iterates every registered component type and unions any
    /// AABBs returned by their [`Component::aabb`] implementations.
    /// Returns `None` if no component contributes an AABB.
    pub fn entity_aabb(&self, entity: Entity) -> Option<redlilium_core::math::Aabb> {
        let mut result: Option<redlilium_core::math::Aabb> = None;
        for meta in self.iter_meta() {
            if let Some(aabb) = (meta.aabb_fn)(self, entity) {
                result = Some(match result {
                    Some(current) => current.union(&aabb),
                    None => aabb,
                });
            }
        }
        result
    }

    /// Returns individual AABBs from each component on an entity.
    ///
    /// Unlike [`entity_aabb`](Self::entity_aabb) which unions all AABBs,
    /// this returns each component's AABB separately.
    pub fn entity_aabbs(&self, entity: Entity) -> Vec<redlilium_core::math::Aabb> {
        let mut result = Vec::new();
        for meta in self.iter_meta() {
            if let Some(aabb) = (meta.aabb_fn)(self, entity) {
                result.push(aabb);
            }
        }
        result
    }

    /// Collects all entity references from all registered components on an entity.
    ///
    /// Iterates every registered component type and appends any
    /// entity references found to `collector`.
    pub fn collect_all_entities(&self, entity: Entity, collector: &mut Vec<Entity>) {
        for meta in self.iter_meta() {
            (meta.collect_entities_fn)(self, entity, collector);
        }
    }

    /// Remaps all entity references in all registered components on an entity.
    ///
    /// Iterates every registered component type and remaps any
    /// entity references found using the provided mapping function.
    pub fn remap_all_entities(&mut self, entity: Entity, map: &mut dyn FnMut(Entity) -> Entity) {
        let fns: Vec<_> = self.iter_meta().map(|m| m.remap_entities_fn).collect();
        for f in fns {
            f(self, entity, map);
        }
    }

    /// Clones all inspector-registered components from one entity to a new entity.
    ///
    /// Spawns a new entity and copies every component registered via
    /// [`register_inspector`](World::register_inspector) or
    /// [`register_inspector_default`](World::register_inspector_default).
    ///
    /// Collects all entities in the subtree rooted at `root` via a breadth-first
    /// walk over [`Children`](crate::Children), in BFS order (index 0 = root).
    ///
    /// Deduplicates with a visited set so a malformed hierarchy (an entity
    /// appearing in multiple `Children` lists, or a cycle from manual edits)
    /// yields each entity at most once instead of being collected repeatedly or
    /// recursing without bound.
    fn collect_subtree_bfs(&self, root: Entity) -> Vec<Entity> {
        let mut order = vec![root];
        let mut visited: std::collections::HashSet<Entity> = std::collections::HashSet::new();
        visited.insert(root);
        let mut i = 0;
        while i < order.len() {
            let entity = order[i];
            if let Some(children) = self.get::<crate::Children>(entity) {
                for &child in &children.0 {
                    if visited.insert(child) {
                        order.push(child);
                    }
                }
            }
            i += 1;
        }
        order
    }

    /// Does **not** traverse the hierarchy or remap entity references.
    /// For hierarchy-aware cloning, use [`clone_entity_tree`](World::clone_entity_tree).
    ///
    /// Returns `None` if the source entity is not alive.
    pub fn clone_entity(&mut self, src: Entity) -> Option<Entity> {
        if !self.is_alive(src) {
            return None;
        }
        let dst = self.spawn();

        let clone_fns: Vec<_> = self
            .iter_meta()
            .filter_map(|m| {
                if (m.has_fn)(self, src) {
                    m.clone_fn
                } else {
                    None
                }
            })
            .collect();

        for f in clone_fns {
            f(self, src, dst);
        }

        Some(dst)
    }

    /// Clones an entity subtree, remapping all internal entity references.
    ///
    /// Performs a breadth-first walk from `root` through [`Children`](crate::Children)
    /// components, clones all clone-enabled components on every entity in the
    /// subtree, then remaps all entity references (via [`remap_all_entities`])
    /// so that internal references point to the new cloned entities.
    ///
    /// Entity references that point **outside** the subtree are left unchanged.
    ///
    /// Returns a mapping from old entity IDs to new entity IDs.
    /// The cloned root is `mapping[&root]`.
    ///
    /// Returns an empty map if `root` is not alive.
    pub fn clone_entity_tree(&mut self, root: Entity) -> HashMap<Entity, Entity> {
        if !self.is_alive(root) {
            return HashMap::new();
        }

        // 1. Collect all entities in subtree via Children (BFS)
        let old_entities = self.collect_subtree_bfs(root);

        // 2. Spawn new entities and build old->new mapping
        let mut mapping = HashMap::with_capacity(old_entities.len());
        for &old in &old_entities {
            let new = self.spawn();
            mapping.insert(old, new);
        }

        // 3. Clone all components from each old entity to corresponding new entity
        let clone_fns: Vec<_> = self
            .iter_meta()
            .filter_map(|m| m.clone_fn.map(|clone| (m.has_fn, clone)))
            .collect();

        for (&old, &new) in &mapping {
            for &(has_fn, clone_fn) in &clone_fns {
                if has_fn(self, old) {
                    clone_fn(self, old, new);
                }
            }
        }

        // 4. Remap all entity references in new entities
        let remap_fns: Vec<_> = self.iter_meta().map(|m| m.remap_entities_fn).collect();

        for &new in mapping.values() {
            for &f in &remap_fns {
                f(self, new, &mut |e| *mapping.get(&e).unwrap_or(&e));
            }
        }

        mapping
    }

    /// Extracts a prefab from an entity subtree.
    ///
    /// Performs a breadth-first walk from `root` through [`Children`](crate::Children)
    /// components and extracts all clone-enabled components into a portable
    /// [`Prefab`](crate::prefab::Prefab). The prefab is fully self-contained
    /// and can be instantiated into any world.
    ///
    /// Components without clone/extract support are silently skipped.
    ///
    /// Returns an empty prefab if `root` is not alive.
    pub fn extract_prefab(&self, root: Entity) -> crate::prefab::Prefab {
        if !self.is_alive(root) {
            return crate::prefab::Prefab::empty();
        }

        // 1. BFS walk via Children to collect all entities in subtree
        let old_entities = self.collect_subtree_bfs(root);

        // 2. Collect extract_fns
        let extract_fns: Vec<_> = self.iter_meta().filter_map(|m| m.extract_fn).collect();

        // 3. For each entity, extract all components into bags
        let entities = old_entities
            .iter()
            .map(|&entity| {
                let bags: Vec<_> = extract_fns.iter().filter_map(|f| f(self, entity)).collect();
                (entity, bags)
            })
            .collect();

        crate::prefab::Prefab::new(entities)
    }

    /// Serializes an entity subtree into a [`SerializedPrefab`](crate::serialize::SerializedPrefab)
    /// as a **same-world snapshot** (undo/redo backups).
    ///
    /// Entity references pointing outside the subtree are serialized as-is:
    /// restoring in the same world keeps still-alive targets linked, and
    /// targets that died in the meantime stay dangling. For self-contained
    /// prefab **assets** use [`serialize_prefab_asset`](World::serialize_prefab_asset).
    ///
    /// Performs a breadth-first walk from `root` through [`Children`](crate::Children)
    /// components and serializes all serializable components. Components that
    /// return [`NotSerializable`](crate::serialize::SerializeError::NotSerializable)
    /// are silently skipped.
    ///
    /// The resulting prefab can be encoded to RON or bincode via the
    /// [`format`](crate::serialize::format) module and deserialized back with
    /// [`deserialize_prefab`](World::deserialize_prefab).
    ///
    /// Returns an empty prefab if `root` is not alive.
    pub fn serialize_prefab(
        &self,
        root: Entity,
    ) -> Result<crate::serialize::SerializedPrefab, crate::serialize::SerializeError> {
        self.serialize_prefab_impl(root, false)
    }

    /// Serializes an entity subtree into a self-contained **prefab asset**.
    ///
    /// Unlike [`serialize_prefab`](World::serialize_prefab) (a same-world
    /// snapshot), a prefab asset must not depend on the source world:
    ///
    /// - a reference to a **live entity outside the subtree** fails with
    ///   [`ExternalEntityRef`](crate::serialize::SerializeError::ExternalEntityRef) —
    ///   include the target in the prefab or clear the link;
    /// - a **dangling** reference (dead target) is allowed and stays dangling
    ///   after instantiation (resolved to [`Entity::DANGLING`] by
    ///   [`deserialize_prefab_asset`](World::deserialize_prefab_asset)).
    ///
    /// The root's `Parent` is exempt: it is dropped from the output, as with
    /// [`serialize_prefab`](World::serialize_prefab).
    pub fn serialize_prefab_asset(
        &self,
        root: Entity,
    ) -> Result<crate::serialize::SerializedPrefab, crate::serialize::SerializeError> {
        self.serialize_prefab_impl(root, true)
    }

    fn serialize_prefab_impl(
        &self,
        root: Entity,
        forbid_external_refs: bool,
    ) -> Result<crate::serialize::SerializedPrefab, crate::serialize::SerializeError> {
        if !self.is_alive(root) {
            return Ok(crate::serialize::SerializedPrefab {
                entities: Vec::new(),
            });
        }

        // 1. BFS walk via Children to collect all entities in subtree
        let old_entities = self.collect_subtree_bfs(root);

        // 1b. Asset mode: every entity reference must resolve inside the
        // subtree or be dangling. The root's Parent is skipped — it is
        // dropped from the output below (external by construction).
        if forbid_external_refs {
            let scope: std::collections::HashSet<Entity> = old_entities.iter().copied().collect();
            let mut refs = Vec::new();
            for &entity in &old_entities {
                for meta in self.iter_meta() {
                    if entity == root && meta.name == "Parent" {
                        continue;
                    }
                    refs.clear();
                    (meta.collect_entities_fn)(self, entity, &mut refs);
                    for &target in &refs {
                        if !scope.contains(&target) && self.is_alive(target) {
                            return Err(crate::serialize::SerializeError::ExternalEntityRef {
                                entity,
                                component: meta.name.to_owned(),
                                target,
                            });
                        }
                    }
                }
            }
        }

        // 2. Collect serialize_fns
        let serialize_fns: Vec<SerializeComponentFn> =
            self.iter_meta().map(|m| m.serialize_fn).collect();

        // 3. Create context (Arc dedup tracked across all entities)
        let mut ctx = crate::serialize::SerializeContext::new(self);

        // 4. For each entity, serialize all components. The root's `Parent` is
        // deliberately dropped: it references an entity *outside* the subtree
        // (attachment context, not prefab content) and would fail the entity
        // remap on deserialization. Callers that care (delete-undo, spawn
        // under a node) re-attach the root themselves.
        let serialized_entities = old_entities
            .iter()
            .map(|&entity| {
                let components: Vec<_> = serialize_fns
                    .iter()
                    .filter_map(|f| f(self, entity, &mut ctx).transpose())
                    .filter(|c| {
                        !(entity == root && c.as_ref().is_ok_and(|c| c.type_name == "Parent"))
                    })
                    .collect::<Result<_, _>>()?;
                Ok(crate::serialize::SerializedEntity {
                    entity_index: entity.index(),
                    entity_spawn_tick: entity.spawn_tick(),
                    entity_flags: self.get_entity_flags(entity),
                    components,
                })
            })
            .collect::<Result<_, crate::serialize::SerializeError>>()?;

        Ok(crate::serialize::SerializedPrefab {
            entities: serialized_entities,
        })
    }

    /// Serializes the **entire world** — every alive entity and every opted-in
    /// snapshot resource — into a [`SerializedWorld`](crate::serialize::SerializedWorld).
    ///
    /// This is the single snapshot mechanism shared by Play mode, scene
    /// save/load, and warm-restart reload (see issue #48). Unlike
    /// [`serialize_prefab`](World::serialize_prefab), it enumerates all alive
    /// entities via [`iter_entities`](World::iter_entities) rather than walking
    /// a subtree, and keeps `Parent` links intact (every referenced entity is
    /// part of the snapshot). Entity references are captured as-is and remapped
    /// on restore; a reference to an entity that dies before the snapshot is
    /// taken stays dangling and resolves to
    /// [`Entity::DANGLING`](crate::Entity::DANGLING) on restore.
    ///
    /// Only resources registered via
    /// [`register_snapshot_resource`](World::register_snapshot_resource) are
    /// captured; host/GPU managers and ephemeral resources are excluded by not
    /// opting in.
    pub fn serialize_world(
        &self,
    ) -> Result<crate::serialize::SerializedWorld, crate::serialize::SerializeError> {
        // 1. Every alive entity (no BFS root, no scope restriction).
        let entities: Vec<Entity> = self.iter_entities().collect();

        // 2. Collect serialize fns and a shared context (Arc dedup spans all
        // entities, exactly as in the prefab path).
        let serialize_fns: Vec<SerializeComponentFn> =
            self.iter_meta().map(|m| m.serialize_fn).collect();
        let mut ctx = crate::serialize::SerializeContext::new(self);

        // 3. Serialize every component of every entity. Nothing is dropped:
        // a whole-world snapshot keeps `Parent` because the target is captured
        // too and remaps on restore.
        let serialized_entities = entities
            .iter()
            .map(|&entity| {
                let components: Vec<_> = serialize_fns
                    .iter()
                    .filter_map(|f| f(self, entity, &mut ctx).transpose())
                    .collect::<Result<_, _>>()?;
                Ok(crate::serialize::SerializedEntity {
                    entity_index: entity.index(),
                    entity_spawn_tick: entity.spawn_tick(),
                    entity_flags: self.get_entity_flags(entity),
                    components,
                })
            })
            .collect::<Result<_, crate::serialize::SerializeError>>()?;

        // 4. Capture opted-in snapshot resources.
        let resources = self.capture_snapshot_resources()?;

        // 5. Phase 5: Build schema metadata (component hashes for validation on restore).
        let mut component_schemas = std::collections::HashMap::new();
        for meta in self.iter_meta() {
            if !meta.schema_hash.is_empty() {
                component_schemas.insert(meta.name.to_string(), meta.schema_hash.clone());
            }
        }
        // Phase 5: TODO - track actual engine fingerprint for reload validation
        // For now, use empty string; will be populated when generation_registry API is exposed
        let engine_fingerprint = String::new();
        let metadata = crate::serialize::SnapshotMetadata {
            engine_fingerprint,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            component_schemas,
        };

        Ok(crate::serialize::SerializedWorld {
            entities: serialized_entities,
            resources,
            metadata,
        })
    }

    /// Restores a [`SerializedWorld`](crate::serialize::SerializedWorld) into
    /// this world, spawning fresh entities and restoring opted-in resources.
    ///
    /// The target world must already have the relevant component types
    /// registered (via `register_std_components`, `register_rendering_components`
    /// and any plugin) and the relevant resources registered via
    /// [`register_snapshot_resource`](World::register_snapshot_resource) — the
    /// snapshot carries data, not registration.
    ///
    /// Entity references remap to the freshly spawned entities; a reference
    /// whose target is not in the snapshot (it died before capture) resolves to
    /// [`Entity::DANGLING`](crate::Entity::DANGLING). Unknown component and
    /// resource types are skipped silently, so a snapshot can be restored into
    /// a world missing a plugin that was present at capture.
    ///
    /// Returns the newly spawned entities in snapshot order. On error, every
    /// entity spawned by this call is despawned — a failed restore leaves the
    /// pre-existing world contents untouched (resources already restored before
    /// the failure are not rolled back).
    pub fn deserialize_world_into(
        &mut self,
        snapshot: &crate::serialize::SerializedWorld,
    ) -> Result<Vec<Entity>, crate::serialize::DeserializeError> {
        // 0. Phase 6: Validate component schema hashes (detect field reordering)
        // before deserializing any components. Build a map of component name → schema_hash.
        let schema_hashes: HashMap<&str, &str> = self
            .iter_meta()
            .map(|m| (m.name, m.schema_hash.as_str()))
            .collect();

        // Check each component in the snapshot against the current schema hash.
        // If a hash is stored and doesn't match, reject the snapshot (prevent silent corruption).
        for entity_record in &snapshot.entities {
            for component_data in &entity_record.components {
                let component_name = component_data.type_name.as_str();
                if let Some(&current_hash) = schema_hashes.get(component_name)
                    && let Some(stored_hash) =
                        snapshot.metadata.component_schemas.get(component_name)
                    && current_hash != stored_hash.as_str()
                {
                    return Err(crate::serialize::DeserializeError::SchemaMismatch {
                        component: component_name.to_string(),
                        expected: current_hash.to_string(),
                        found: stored_hash.clone(),
                    });
                }
            }
        }

        // 1. Spawn one fresh entity per serialized entity.
        let new_entities: Vec<Entity> = snapshot.entities.iter().map(|_| self.spawn()).collect();

        // 2. Restore components (Dangling policy: refs to entities absent from
        // the snapshot are dead targets → Entity::DANGLING), then resources.
        let result = self
            .deserialize_prefab_body(
                &snapshot.entities,
                &new_entities,
                crate::serialize::UnmappedEntityPolicy::Dangling,
            )
            .and_then(|()| self.restore_snapshot_resources(&snapshot.resources));

        match result {
            Ok(()) => Ok(new_entities),
            Err(e) => {
                // Roll back the entities this call spawned.
                for &entity in &new_entities {
                    self.despawn(entity);
                }
                Err(e)
            }
        }
    }

    /// Deserializes a [`SerializedPrefab`](crate::serialize::SerializedPrefab) into new entities
    /// as a **same-world snapshot** restore (undo/redo).
    ///
    /// Spawns new entities, deserializes components, and remaps entity references
    /// (e.g., [`Parent`](crate::Parent) / [`Children`](crate::Children)) to the
    /// newly spawned entities. References whose target is not part of the
    /// prefab keep their original handle: a still-alive target stays linked, a
    /// dead one stays dangling. For prefab **assets** use
    /// [`deserialize_prefab_asset`](World::deserialize_prefab_asset). Arc
    /// values that were shared during serialization are shared again via the
    /// deduplication cache.
    ///
    /// Returns the list of new entities in BFS order (index 0 = root). On
    /// error, every entity spawned by this call is despawned — a failed
    /// restore leaves the world unchanged.
    ///
    /// Unknown component types are silently skipped.
    pub fn deserialize_prefab(
        &mut self,
        prefab: &crate::serialize::SerializedPrefab,
    ) -> Result<Vec<Entity>, crate::serialize::DeserializeError> {
        self.deserialize_prefab_impl(prefab, crate::serialize::UnmappedEntityPolicy::Keep)
    }

    /// Deserializes a **prefab asset** (see
    /// [`serialize_prefab_asset`](World::serialize_prefab_asset)) into new entities.
    ///
    /// Same as [`deserialize_prefab`](World::deserialize_prefab), except that a
    /// reference whose target is not part of the prefab resolves to
    /// [`Entity::DANGLING`] instead of keeping the original handle — in a
    /// foreign world the source handle could alias an unrelated live entity.
    /// A reference that was dangling when the asset was exported is thus
    /// dangling after instantiation, in any world.
    pub fn deserialize_prefab_asset(
        &mut self,
        prefab: &crate::serialize::SerializedPrefab,
    ) -> Result<Vec<Entity>, crate::serialize::DeserializeError> {
        self.deserialize_prefab_impl(prefab, crate::serialize::UnmappedEntityPolicy::Dangling)
    }

    fn deserialize_prefab_impl(
        &mut self,
        prefab: &crate::serialize::SerializedPrefab,
        unmapped_policy: crate::serialize::UnmappedEntityPolicy,
    ) -> Result<Vec<Entity>, crate::serialize::DeserializeError> {
        // 1. Spawn entities
        let new_entities: Vec<Entity> = prefab.entities.iter().map(|_| self.spawn()).collect();

        match self.deserialize_prefab_body(&prefab.entities, &new_entities, unmapped_policy) {
            Ok(()) => Ok(new_entities),
            Err(e) => {
                // Roll back: a failed restore must not leave a partially
                // spawned entity tree in the world.
                for &entity in &new_entities {
                    self.despawn(entity);
                }
                Err(e)
            }
        }
    }

    fn deserialize_prefab_body(
        &mut self,
        entities: &[crate::serialize::SerializedEntity],
        new_entities: &[Entity],
        unmapped_policy: crate::serialize::UnmappedEntityPolicy,
    ) -> Result<(), crate::serialize::DeserializeError> {
        // 2. Build entity remap: (old_index, old_spawn_tick) -> new Entity
        let entity_map: HashMap<(u32, u64), Entity> = entities
            .iter()
            .zip(new_entities.iter())
            .map(|(se, &new_e)| ((se.entity_index, se.entity_spawn_tick), new_e))
            .collect();

        // 3. Extract deserialize fns and schema versions by name (before borrowing self mutably via context)
        let deserialize_fns: HashMap<&str, DeserializeComponentFn> = self
            .iter_meta()
            .map(|m| (m.name, m.deserialize_fn))
            .collect();
        let schema_versions: HashMap<&str, u32> = self
            .iter_meta()
            .map(|m| (m.name, m.schema_version))
            .collect();

        // 4. Restore entity flags
        for (i, se) in entities.iter().enumerate() {
            if se.entity_flags != 0 {
                self.set_entity_flags(new_entities[i], se.entity_flags);
            }
        }

        // 5. Pre-scan EVERY component's data (including those whose type is
        // unknown/skipped) to record each shared Arc's raw inner. This lets an
        // `ArcRef` on a registered component resolve even when the original
        // `ArcValue` lived on a component that gets skipped below.
        {
            let mut ctx = crate::serialize::DeserializeContext::new(self);
            for se in entities {
                for comp in &se.components {
                    ctx.prescan_arc_values(&comp.data);
                }
            }
        } // ctx dropped here

        // 6. For each entity, deserialize components (with migration support for schema changes)
        for (i, se) in entities.iter().enumerate() {
            let entity = new_entities[i];
            for comp in &se.components {
                if let Some(&deser_fn) = deserialize_fns.get(comp.type_name.as_str()) {
                    // Get the current schema version for this component type
                    let current_schema = schema_versions
                        .get(comp.type_name.as_str())
                        .copied()
                        .unwrap_or(1);

                    // Check if schema version has changed (Phase 4: primary migration trigger)
                    if comp.schema_version != current_schema {
                        // Schema mismatch: attempt migration before normal deserialization
                        let migration_result = self.migration_registry.try_migrate(
                            comp.type_name.as_str(),
                            &comp.data,
                            comp.schema_version,
                            current_schema,
                        );

                        if let Some(transformed) = migration_result {
                            // Migration transformed the data; retry with transformed data
                            let mut ctx = crate::serialize::DeserializeContext::new(self);
                            ctx.set_entity_map(entity_map.clone());
                            ctx.set_unmapped_policy(unmapped_policy);
                            ctx.set_current_component(&comp.type_name);

                            match deser_fn(entity, &transformed, &mut ctx) {
                                Ok(()) => {
                                    // Success after migration
                                }
                                Err(crate::serialize::DeserializeError::NotDeserializable {
                                    ..
                                }) => {
                                    // Migrated data still can't deserialize; skip component
                                    eprintln!(
                                        "warning: Component `{}` on entity {:?} skipped: \
                                         migrated data incompatible with deserializer.",
                                        comp.type_name, entity
                                    );
                                }
                                Err(e) => {
                                    // Migration applied but deserialization still failed
                                    return Err(e);
                                }
                            }
                        } else {
                            // No migration available for version mismatch; skip component
                            eprintln!(
                                "warning: Component `{}` on entity {:?} skipped: \
                                 schema version changed ({} → {}) but no migration registered.",
                                comp.type_name, entity, comp.schema_version, current_schema
                            );
                        }
                    } else {
                        // Schema version matches; attempt normal deserialization
                        let mut ctx = crate::serialize::DeserializeContext::new(self);
                        ctx.set_entity_map(entity_map.clone());
                        ctx.set_unmapped_policy(unmapped_policy);
                        ctx.set_current_component(&comp.type_name);

                        match deser_fn(entity, &comp.data, &mut ctx) {
                            Ok(()) => {
                                // Success on first try
                            }
                            Err(crate::serialize::DeserializeError::NotDeserializable {
                                ..
                            }) => {
                                // Component type doesn't support deserialization
                            }
                            Err(deser_err) => {
                                // Deserialization failed (data format mismatch, not schema change)
                                return Err(deser_err);
                            }
                        }
                    }
                }
                // Skip unknown component types silently
            }
        }

        Ok(())
    }
}
