/// A 64-bit entity handle packed as `id:24 + spawn_tick:40`.
///
/// - **id** (bits 0–23): slot index in the entity store (max ~16M entities)
/// - **spawn_tick** (bits 24–63): truncated world tick when this entity was
///   spawned — used for ABA detection when a slot is recycled (~145 years
///   at 240 FPS before overflow)
///
/// Per-entity mutable state (flags, full spawn tick) lives in [`Entities`],
/// not in the handle itself, because `Entity` is `Copy` and would go stale.
///
/// # Identity
///
/// Two entities are equal if their packed bits are identical (same id and
/// spawn_tick).
///
/// # Example
///
/// ```
/// use redlilium_ecs::Entity;
///
/// // Entities are compared by id and spawn_tick
/// // (Entity creation is handled by World)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    bits: u64,
}

impl Entity {
    /// Entity is manually disabled by user/system.
    pub const DISABLED: u32 = 1 << 0;
    /// Entity is disabled because a parent was disabled (propagated).
    pub const INHERITED_DISABLED: u32 = 1 << 1;
    /// Entity is manually marked as static (rarely-changing, e.g. terrain).
    ///
    /// Static entities are excluded from both `Read<T>` and `Write<T>` queries.
    /// Use `ReadAll<T>` to include them in read-only queries, or access them
    /// directly via exclusive systems (`&mut World`).
    pub const STATIC: u32 = 1 << 2;
    /// Entity is static because a parent was marked static (propagated).
    pub const INHERITED_STATIC: u32 = 1 << 3;
    /// Entity is an editor-only entity (camera, grid, gizmos, etc.).
    ///
    /// Editor entities are excluded from `Read<T>` and `Write<T>` queries.
    /// Use `ReadAll<T>` / `WriteAll<T>` to include them, or access them
    /// directly via exclusive systems (`&mut World`).
    pub const EDITOR: u32 = 1 << 4;
    /// Entity is an editor entity because a parent was marked as editor (propagated).
    pub const INHERITED_EDITOR: u32 = 1 << 5;

    const ID_BITS: u32 = 24;
    const ID_MASK: u64 = (1u64 << Self::ID_BITS) - 1; // 0x00FF_FFFF
    const TICK_SHIFT: u32 = Self::ID_BITS;
    const TICK_MASK: u64 = !Self::ID_MASK; // upper 40 bits

    /// Maximum entity index (2^24 - 1).
    pub const MAX_INDEX: u32 = (1u32 << Self::ID_BITS) - 1;

    /// Creates a new entity handle from a slot index and a spawn tick.
    ///
    /// Only the lower 40 bits of `spawn_tick` are stored in the handle.
    pub(crate) fn new(index: u32, spawn_tick: u64) -> Self {
        debug_assert!(
            index <= Self::MAX_INDEX,
            "Entity index {index} exceeds maximum {}",
            Self::MAX_INDEX
        );
        Self {
            bits: (index as u64 & Self::ID_MASK)
                | ((spawn_tick << Self::TICK_SHIFT) & Self::TICK_MASK),
        }
    }

    /// Returns the slot index of this entity.
    pub fn index(&self) -> u32 {
        (self.bits & Self::ID_MASK) as u32
    }

    /// Returns the truncated spawn tick stored in this handle (40-bit).
    pub fn spawn_tick(&self) -> u64 {
        (self.bits & Self::TICK_MASK) >> Self::TICK_SHIFT
    }
}

impl std::fmt::Debug for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}@{})", self.index(), self.spawn_tick())
    }
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}@{})", self.index(), self.spawn_tick())
    }
}

/// Owns and manages all entity slots with per-slot generation tracking.
///
/// When an entity is despawned, its slot's generation is bumped and the slot is
/// added to a free list. The next spawn reuses the slot with the new
/// generation, invalidating any old `Entity` handles.
///
/// The generation is **independent of the world tick** so that a despawn and a
/// respawn within the same frame still produce distinct handles (ABA-safe). The
/// generation occupies the same 40-bit field that the `Entity` handle exposes
/// via [`Entity::spawn_tick`].
///
/// A slot is considered dead when its generation entry equals [`DEAD_TICK`].
pub struct Entities {
    /// Per-slot generation stamped into live handles (40-bit). Dead slots store
    /// [`Entities::DEAD_TICK`].
    generations: Vec<u64>,
    /// Generation to stamp the next time each slot is recycled (40-bit, never
    /// equal to [`Entities::DEAD_TICK`]).
    next_generation: Vec<u64>,
    /// Per-slot flag bits (disabled, inherited-disabled, etc.).
    flags: Vec<u32>,
    /// Free list of recyclable indices (LIFO stack).
    free_list: Vec<u32>,
    /// Total number of currently alive entities.
    count: u32,
}

impl Entities {
    /// Generation value written into dead slots (all 40 bits set = `2^40 - 1`).
    const DEAD_TICK: u64 = (1u64 << 40) - 1;

    /// Creates a new empty entity store.
    pub(crate) fn new() -> Self {
        Self {
            generations: Vec::new(),
            next_generation: Vec::new(),
            flags: Vec::new(),
            free_list: Vec::new(),
            count: 0,
        }
    }

    /// Returns the generation following `g`, wrapped to 40 bits and skipping the
    /// dead sentinel.
    fn next_after(g: u64) -> u64 {
        let n = (g + 1) & ((1u64 << 40) - 1);
        if n == Self::DEAD_TICK { 0 } else { n }
    }

    /// Allocates a new entity, reusing a recycled slot if available.
    ///
    /// The handle's generation is independent of the world tick, so a slot
    /// recycled within the same frame yields a fresh, non-aliasing handle.
    pub(crate) fn allocate(&mut self) -> Entity {
        self.count += 1;

        if let Some(index) = self.free_list.pop() {
            let idx = index as usize;
            let generation = self.next_generation[idx];
            self.generations[idx] = generation;
            self.flags[idx] = 0;
            Entity::new(index, generation)
        } else {
            let index = self.generations.len() as u32;
            assert!(
                index <= Entity::MAX_INDEX,
                "Entity limit exceeded (max {})",
                Entity::MAX_INDEX + 1
            );
            self.generations.push(0);
            self.next_generation.push(Self::next_after(0));
            self.flags.push(0);
            Entity::new(index, 0)
        }
    }

    /// Deallocates an entity. Returns false if already dead or generation mismatch.
    pub(crate) fn deallocate(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx >= self.generations.len() || self.generations[idx] != entity.spawn_tick() {
            return false;
        }
        if self.generations[idx] == Self::DEAD_TICK {
            return false;
        }

        // Bump the generation so the next occupant of this slot gets a handle
        // distinct from the one being freed (ABA detection).
        self.next_generation[idx] = Self::next_after(self.generations[idx]);
        self.generations[idx] = Self::DEAD_TICK;
        self.flags[idx] = 0;
        self.free_list.push(entity.index());
        self.count -= 1;
        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.generations.len()
            && self.generations[idx] != Self::DEAD_TICK
            && self.generations[idx] == entity.spawn_tick()
    }

    /// Returns the number of alive entities.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the total number of slots (alive + dead).
    pub fn slots_len(&self) -> usize {
        self.generations.len()
    }

    /// Sets flag bits on an entity slot (OR operation).
    pub(crate) fn set_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] |= bits;
    }

    /// Clears flag bits on an entity slot (AND-NOT operation).
    pub(crate) fn clear_flags(&mut self, index: u32, bits: u32) {
        self.flags[index as usize] &= !bits;
    }

    /// Returns the flags for a given entity slot.
    pub fn get_flags(&self, index: u32) -> u32 {
        self.flags[index as usize]
    }

    /// Returns the current generation (spawn_tick) for the given slot.
    pub fn get_spawn_tick(&self, index: u32) -> u64 {
        self.generations[index as usize]
    }

    /// Allocates `count` entities at once, reusing recycled slots first.
    ///
    /// More efficient than calling [`allocate`](Self::allocate) in a loop
    /// because internal vectors are grown in bulk.
    pub(crate) fn allocate_many(&mut self, count: u32) -> Vec<Entity> {
        let mut entities = Vec::with_capacity(count as usize);

        // Reuse from free list first
        let reuse = count.min(self.free_list.len() as u32);
        for _ in 0..reuse {
            let index = self.free_list.pop().unwrap();
            let idx = index as usize;
            let generation = self.next_generation[idx];
            self.generations[idx] = generation;
            self.flags[idx] = 0;
            entities.push(Entity::new(index, generation));
        }

        // Allocate fresh slots for remainder
        let fresh = count - reuse;
        if fresh > 0 {
            let start = self.generations.len() as u32;
            assert!(
                start + fresh - 1 <= Entity::MAX_INDEX,
                "Entity limit exceeded (max {})",
                Entity::MAX_INDEX + 1
            );
            for i in 0..fresh {
                self.generations.push(0);
                self.next_generation.push(Self::next_after(0));
                self.flags.push(0);
                entities.push(Entity::new(start + i, 0));
            }
        }

        self.count += count;
        entities
    }

    /// Returns the alive entity at the given index, or `None` if the slot is
    /// empty or has been recycled.
    pub fn entity_at_index(&self, index: u32) -> Option<Entity> {
        let idx = index as usize;
        if idx < self.generations.len() && self.generations[idx] != Self::DEAD_TICK {
            Some(Entity::new(index, self.generations[idx]))
        } else {
            None
        }
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.generations
            .iter()
            .enumerate()
            .filter(|(_, generation)| **generation != Self::DEAD_TICK)
            .map(|(idx, generation)| Entity::new(idx as u32, *generation))
    }
}

impl Default for Entities {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_size() {
        assert_eq!(std::mem::size_of::<Entity>(), 8);
    }

    #[test]
    fn pack_unpack() {
        let e = Entity::new(42, 12345);
        assert_eq!(e.index(), 42);
        assert_eq!(e.spawn_tick(), 12345);
    }

    #[test]
    fn max_index() {
        let e = Entity::new(Entity::MAX_INDEX, 0);
        assert_eq!(e.index(), Entity::MAX_INDEX);
    }

    #[test]
    fn allocate_sequential() {
        let mut alloc = Entities::new();
        let e0 = alloc.allocate();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();

        assert_eq!(e0.index(), 0);
        assert_eq!(e1.index(), 1);
        assert_eq!(e2.index(), 2);
        // Fresh slots start at generation 0.
        assert_eq!(e0.spawn_tick(), 0);
        assert_eq!(e1.spawn_tick(), 0);
        assert_eq!(e2.spawn_tick(), 0);
    }

    #[test]
    fn is_alive_after_allocate() {
        let mut alloc = Entities::new();
        let entity = alloc.allocate();
        assert!(alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_makes_dead() {
        let mut alloc = Entities::new();
        let entity = alloc.allocate();
        assert!(alloc.deallocate(entity));
        assert!(!alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_stale_entity() {
        let mut alloc = Entities::new();
        let entity = alloc.allocate();
        assert!(alloc.deallocate(entity));
        // Deallocating again returns false
        assert!(!alloc.deallocate(entity));
    }

    #[test]
    fn recycled_slot_bumps_generation() {
        let mut alloc = Entities::new();
        let e0 = alloc.allocate();
        alloc.deallocate(e0);
        let e1 = alloc.allocate();

        assert_eq!(e1.index(), 0); // Same slot
        assert_eq!(e0.spawn_tick(), 0);
        assert_eq!(e1.spawn_tick(), 1); // Generation bumped on recycle
        assert_ne!(e0, e1);
    }

    #[test]
    fn aba_same_slot_recycled_without_tick_change() {
        // Regression for the Entity ABA bug: despawn + respawn of the same slot
        // must yield a distinct handle even with no intervening world-tick
        // change. The generation alone guarantees this.
        let mut alloc = Entities::new();
        let old = alloc.allocate();
        alloc.deallocate(old);
        let new = alloc.allocate();

        assert_eq!(old.index(), new.index());
        assert_ne!(old, new);
        assert!(!alloc.is_alive(old));
        assert!(alloc.is_alive(new));
    }

    #[test]
    fn generation_increments_across_many_recycles() {
        let mut alloc = Entities::new();
        let mut prev = alloc.allocate();
        for expected_gen in 1..5u64 {
            alloc.deallocate(prev);
            let next = alloc.allocate();
            assert_eq!(next.index(), 0);
            assert_eq!(next.spawn_tick(), expected_gen);
            assert!(!alloc.is_alive(prev));
            prev = next;
        }
    }

    #[test]
    fn stale_entity_not_alive() {
        let mut alloc = Entities::new();
        let old = alloc.allocate();
        alloc.deallocate(old);
        let _new = alloc.allocate();

        // Old handle (generation 0) is not alive even though slot 0 is alive.
        assert!(!alloc.is_alive(old));
    }

    #[test]
    fn count_tracks_alive() {
        let mut alloc = Entities::new();
        assert_eq!(alloc.count(), 0);

        let e0 = alloc.allocate();
        let _e1 = alloc.allocate();
        assert_eq!(alloc.count(), 2);

        alloc.deallocate(e0);
        assert_eq!(alloc.count(), 1);
    }

    #[test]
    fn iter_alive_correctness() {
        let mut alloc = Entities::new();
        let entities: Vec<_> = (0..5).map(|_| alloc.allocate()).collect();

        alloc.deallocate(entities[1]);
        alloc.deallocate(entities[3]);

        let alive: Vec<_> = alloc.iter_alive().collect();
        assert_eq!(alive.len(), 3);
        assert!(alive.contains(&entities[0]));
        assert!(alive.contains(&entities[2]));
        assert!(alive.contains(&entities[4]));
    }

    #[test]
    fn debug_format() {
        let entity = Entity::new(42, 100);
        assert_eq!(format!("{:?}", entity), "Entity(42@100)");
        assert_eq!(format!("{}", entity), "Entity(42@100)");
    }

    #[test]
    fn allocate_many_fresh() {
        let mut alloc = Entities::new();
        let entities = alloc.allocate_many(5);

        assert_eq!(entities.len(), 5);
        assert_eq!(alloc.count(), 5);
        for (i, e) in entities.iter().enumerate() {
            assert_eq!(e.index(), i as u32);
            assert_eq!(e.spawn_tick(), 0); // Fresh slots start at generation 0.
            assert!(alloc.is_alive(*e));
        }
    }

    #[test]
    fn allocate_many_reuses_free_list() {
        let mut alloc = Entities::new();
        let originals: Vec<_> = (0..5).map(|_| alloc.allocate()).collect();

        // Despawn some
        alloc.deallocate(originals[1]);
        alloc.deallocate(originals[3]);

        let batch = alloc.allocate_many(4);
        assert_eq!(batch.len(), 4);
        assert_eq!(alloc.count(), 7); // 3 original alive + 4 new

        // First 2 should reuse recycled slots (indices 3 and 1, LIFO) with a
        // bumped generation (1).
        assert_eq!(batch[0].index(), 3);
        assert_eq!(batch[0].spawn_tick(), 1);
        assert_eq!(batch[1].index(), 1);
        assert_eq!(batch[1].spawn_tick(), 1);

        // Next 2 should be fresh (generation 0).
        assert_eq!(batch[2].index(), 5);
        assert_eq!(batch[2].spawn_tick(), 0);
        assert_eq!(batch[3].index(), 6);
        assert_eq!(batch[3].spawn_tick(), 0);

        for e in &batch {
            assert!(alloc.is_alive(*e));
        }
    }

    #[test]
    fn allocate_many_zero() {
        let mut alloc = Entities::new();
        let entities = alloc.allocate_many(0);
        assert!(entities.is_empty());
        assert_eq!(alloc.count(), 0);
    }

    #[test]
    fn equality_and_hash() {
        let e1 = Entity::new(42, 100);
        let e2 = Entity::new(42, 100);
        let e3 = Entity::new(42, 101);
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash = |e: Entity| {
            let mut h = DefaultHasher::new();
            e.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(e1), hash(e2));
    }

    #[test]
    fn flag_operations() {
        let mut alloc = Entities::new();
        let e = alloc.allocate();
        assert_eq!(alloc.get_flags(e.index()), 0);

        alloc.set_flags(e.index(), Entity::DISABLED);
        assert_eq!(
            alloc.get_flags(e.index()) & Entity::DISABLED,
            Entity::DISABLED
        );

        alloc.set_flags(e.index(), Entity::INHERITED_DISABLED);
        assert_eq!(
            alloc.get_flags(e.index()),
            Entity::DISABLED | Entity::INHERITED_DISABLED
        );

        alloc.clear_flags(e.index(), Entity::DISABLED);
        assert_eq!(alloc.get_flags(e.index()), Entity::INHERITED_DISABLED);

        alloc.clear_flags(e.index(), Entity::INHERITED_DISABLED);
        assert_eq!(alloc.get_flags(e.index()), 0);
    }

    #[test]
    fn deallocate_clears_flags() {
        let mut alloc = Entities::new();
        let e = alloc.allocate();
        alloc.set_flags(e.index(), Entity::DISABLED);
        alloc.deallocate(e);

        let e2 = alloc.allocate();
        assert_eq!(e2.index(), 0); // Same slot
        assert_eq!(alloc.get_flags(e2.index()), 0); // Flags reset
    }

    #[test]
    fn generation_wraps_and_skips_dead_sentinel() {
        // The generation that immediately precedes the dead sentinel must wrap
        // to 0 (never to DEAD_TICK) so a recycled slot is never mistaken for
        // dead.
        assert_eq!(Entities::next_after(Entities::DEAD_TICK - 1), 0);
        assert_eq!(Entities::next_after((1u64 << 40) - 1), 0);
        assert_eq!(Entities::next_after(41), 42);
    }
}
