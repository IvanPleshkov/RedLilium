//! Render graph definition and compilation

use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use std::collections::{HashMap, HashSet};

/// The main render graph structure
pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
    pass_nodes: Vec<PassNode>,
    resources: Vec<VirtualResource>,
    next_pass_id: u32,
    next_resource_id: u32,

    /// External resources (like swapchain)
    external_resources: HashMap<String, ResourceId>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            pass_nodes: Vec::new(),
            resources: Vec::new(),
            next_pass_id: 0,
            next_resource_id: 0,
            external_resources: HashMap::new(),
        }
    }

    /// Register an external resource (like swapchain image)
    pub fn register_external(&mut self, name: &str) -> ResourceId {
        let id = ResourceId(self.next_resource_id);
        self.next_resource_id += 1;
        self.resources.push(VirtualResource::External(id));
        self.external_resources.insert(name.to_string(), id);
        id
    }

    /// Get external resource by name
    pub fn get_external(&self, name: &str) -> Option<ResourceId> {
        self.external_resources.get(name).copied()
    }

    /// Add a render pass to the graph
    pub fn add_pass<P: RenderPass + 'static>(
        &mut self,
        pass: P,
        pass_type: PassType,
        screen_width: u32,
        screen_height: u32,
    ) -> PassId {
        let id = PassId(self.next_pass_id);
        self.next_pass_id += 1;

        let name = pass.name().to_string();
        let mut boxed_pass = Box::new(pass);

        // Setup the pass
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        {
            let mut ctx = PassSetupContext {
                resources: &mut self.resources,
                inputs: &mut inputs,
                outputs: &mut outputs,
                next_resource_id: &mut self.next_resource_id,
                screen_width,
                screen_height,
            };
            boxed_pass.setup(&mut ctx);
        }

        self.passes.push(boxed_pass);
        self.pass_nodes.push(PassNode {
            id,
            name,
            pass_type,
            inputs,
            outputs,
        });

        id
    }

    /// Compile the graph - topological sort and resource allocation planning
    pub fn compile(&self) -> CompiledGraph {
        // Build dependency graph
        let mut dependencies: HashMap<PassId, HashSet<PassId>> = HashMap::new();

        for node in &self.pass_nodes {
            dependencies.insert(node.id, HashSet::new());
        }

        // A pass depends on another if it reads a resource that the other writes
        for reader in &self.pass_nodes {
            for writer in &self.pass_nodes {
                if reader.id == writer.id {
                    continue;
                }

                // Check if reader reads any resource that writer writes
                for input in &reader.inputs {
                    if writer.writes_resource(input.resource) {
                        dependencies.get_mut(&reader.id).unwrap().insert(writer.id);
                    }
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut in_degree: HashMap<PassId, usize> = HashMap::new();
        for node in &self.pass_nodes {
            in_degree.insert(node.id, dependencies[&node.id].len());
        }

        let mut queue: Vec<PassId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted_passes = Vec::new();

        while let Some(pass_id) = queue.pop() {
            sorted_passes.push(pass_id);

            // Find passes that depend on this one
            for node in &self.pass_nodes {
                if dependencies[&node.id].contains(&pass_id) {
                    let degree = in_degree.get_mut(&node.id).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(node.id);
                    }
                }
            }
        }

        // Determine resource lifetimes
        let mut resource_lifetimes: HashMap<ResourceId, ResourceLifetime> = HashMap::new();

        for (order, &pass_id) in sorted_passes.iter().enumerate() {
            let node = self.pass_nodes.iter().find(|n| n.id == pass_id).unwrap();

            for access in node.inputs.iter().chain(node.outputs.iter()) {
                let lifetime = resource_lifetimes
                    .entry(access.resource)
                    .or_insert(ResourceLifetime {
                        first_use: order,
                        last_use: order,
                    });
                lifetime.last_use = order;
            }
        }

        CompiledGraph {
            pass_order: sorted_passes,
            resource_lifetimes,
        }
    }

    /// Get all passes
    pub fn passes(&self) -> &[Box<dyn RenderPass>] {
        &self.passes
    }

    /// Get mutable passes
    pub fn passes_mut(&mut self) -> &mut [Box<dyn RenderPass>] {
        &mut self.passes
    }

    /// Get pass nodes (metadata)
    pub fn pass_nodes(&self) -> &[PassNode] {
        &self.pass_nodes
    }

    /// Get all resources
    pub fn resources(&self) -> &[VirtualResource] {
        &self.resources
    }

    /// Get pass by ID
    pub fn get_pass(&self, id: PassId) -> Option<&dyn RenderPass> {
        let index = self.pass_nodes.iter().position(|n| n.id == id)?;
        Some(self.passes[index].as_ref())
    }

    /// Get pass node by ID
    pub fn get_pass_node(&self, id: PassId) -> Option<&PassNode> {
        self.pass_nodes.iter().find(|n| n.id == id)
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource lifetime in terms of pass execution order
#[derive(Debug, Clone, Copy)]
pub struct ResourceLifetime {
    pub first_use: usize,
    pub last_use: usize,
}

/// Compiled render graph with execution order and resource lifetimes
#[derive(Debug)]
pub struct CompiledGraph {
    pub pass_order: Vec<PassId>,
    pub resource_lifetimes: HashMap<ResourceId, ResourceLifetime>,
}

impl CompiledGraph {
    /// Check if a resource is alive at a given execution step
    pub fn is_resource_alive(&self, resource: ResourceId, step: usize) -> bool {
        if let Some(lifetime) = self.resource_lifetimes.get(&resource) {
            step >= lifetime.first_use && step <= lifetime.last_use
        } else {
            false
        }
    }
}
