/// A lightweight entity identifier with generational index.
///
/// Entities are represented as a 32-bit index + 32-bit generation,
/// packed into a single `u64`. The generation prevents ABA problems
/// when entity slots are recycled.
///
/// # Example
///
/// ```
/// use redlilium_ecs::Entity;
///
/// // Entities are compared by both index and generation
/// // (Entity creation is handled by World)
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    id: u64,
}

impl Entity {
    /// Creates a new entity from an index and generation.
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self {
            id: (generation as u64) << 32 | index as u64,
        }
    }

    /// Returns the index portion of the entity ID.
    pub fn index(&self) -> u32 {
        self.id as u32
    }

    /// Returns the generation portion of the entity ID.
    pub fn generation(&self) -> u32 {
        (self.id >> 32) as u32
    }
}

impl std::fmt::Debug for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}:{})", self.index(), self.generation())
    }
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({}:{})", self.index(), self.generation())
    }
}

/// Allocates and recycles entity IDs with generational tracking.
///
/// When an entity is despawned, its slot is added to a free list.
/// The next spawn reuses the slot with an incremented generation.
pub(crate) struct EntityAllocator {
    /// Generation for each slot. Index = entity index.
    generations: Vec<u32>,
    /// Alive flag per slot.
    alive: Vec<bool>,
    /// Free list of recyclable indices (LIFO stack).
    free_list: Vec<u32>,
    /// Total number of currently alive entities.
    count: u32,
}

impl EntityAllocator {
    /// Creates a new empty allocator.
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
            count: 0,
        }
    }

    /// Allocates a new entity, reusing a recycled slot if available.
    pub fn allocate(&mut self) -> Entity {
        self.count += 1;

        if let Some(index) = self.free_list.pop() {
            self.alive[index as usize] = true;
            Entity::new(index, self.generations[index as usize])
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            Entity::new(index, 0)
        }
    }

    /// Deallocates an entity. Returns false if already dead or generation mismatch.
    pub fn deallocate(&mut self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        if idx >= self.alive.len()
            || !self.alive[idx]
            || self.generations[idx] != entity.generation()
        {
            return false;
        }

        self.alive[idx] = false;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.free_list.push(entity.index());
        self.count -= 1;
        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        idx < self.alive.len() && self.alive[idx] && self.generations[idx] == entity.generation()
    }

    /// Returns the number of alive entities.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter(|(_, alive)| **alive)
            .map(|(idx, _)| Entity::new(idx as u32, self.generations[idx]))
    }
}

impl Default for EntityAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_sequential() {
        let mut alloc = EntityAllocator::new();
        let e0 = alloc.allocate();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();

        assert_eq!(e0.index(), 0);
        assert_eq!(e1.index(), 1);
        assert_eq!(e2.index(), 2);
        assert_eq!(e0.generation(), 0);
        assert_eq!(e1.generation(), 0);
        assert_eq!(e2.generation(), 0);
    }

    #[test]
    fn is_alive_after_allocate() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate();
        assert!(alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_makes_dead() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate();
        assert!(alloc.deallocate(entity));
        assert!(!alloc.is_alive(entity));
    }

    #[test]
    fn deallocate_stale_entity() {
        let mut alloc = EntityAllocator::new();
        let entity = alloc.allocate();
        assert!(alloc.deallocate(entity));
        // Deallocating again returns false
        assert!(!alloc.deallocate(entity));
    }

    #[test]
    fn recycled_slot_new_generation() {
        let mut alloc = EntityAllocator::new();
        let e0 = alloc.allocate();
        alloc.deallocate(e0);
        let e1 = alloc.allocate();

        assert_eq!(e1.index(), 0); // Same slot
        assert_eq!(e1.generation(), 1); // New generation
    }

    #[test]
    fn stale_entity_not_alive() {
        let mut alloc = EntityAllocator::new();
        let old = alloc.allocate();
        alloc.deallocate(old);
        let _new = alloc.allocate();

        // Old entity (gen 0) is not alive even though slot 0 is alive (gen 1)
        assert!(!alloc.is_alive(old));
    }

    #[test]
    fn count_tracks_alive() {
        let mut alloc = EntityAllocator::new();
        assert_eq!(alloc.count(), 0);

        let e0 = alloc.allocate();
        let _e1 = alloc.allocate();
        assert_eq!(alloc.count(), 2);

        alloc.deallocate(e0);
        assert_eq!(alloc.count(), 1);
    }

    #[test]
    fn iter_alive_correctness() {
        let mut alloc = EntityAllocator::new();
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
        let entity = Entity::new(42, 3);
        assert_eq!(format!("{:?}", entity), "Entity(42:3)");
        assert_eq!(format!("{}", entity), "Entity(42:3)");
    }
}
