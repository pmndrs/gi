// Engine-free BVH builders (docs/design/bvh-framework.md). A builder produces a
// MeshTreeTopology; the tracer in @pmndrs/ddgi-core turns it into its GPU layout.
export * from './topology.ts';
export * from './tinybvh.ts';
