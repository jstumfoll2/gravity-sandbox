// Drops newly spawned particles into the free (dead) slots at the end of the live state.
// The last sort left the live/dead boundary in count[t - 1]; when the buffer is full the slots
// wrap round to the start and recycle live particles. Slots follow the batch index, so the
// result is deterministic.

@group(0) @binding(0) var<uniform> u: SimU;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst_aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read> spawn_in: array<vec4f>;

@compute @workgroup_size(256)
fn spawn(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= u.spawn_count) { return; }
    let slot = (counts[u.t - 1u] + k) % u.n;
    dst[slot] = spawn_in[k];
    dst_aux[slot] = 0u;
}

@compute @workgroup_size(1)
fn bump() {
    counts[u.t - 1u] = min(u.n, counts[u.t - 1u] + u.spawn_count);
}
