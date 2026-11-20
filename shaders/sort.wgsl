// Counting sort of the particles by hashed grid cell, every step.
//
//   hash:    key[i] = bucket of i's cell (dead -> bucket t), count[key]++
//   (scan.wgsl turns counts into exclusive starts, in place)
//   scatter: perm[start[key]++] = i              (count[] now holds bucket ends)
//   fix:     sort each bucket's perm by index   (atomics hand out slots in any order;
//                                                this makes the result deterministic)
//   gather:  dst[s] = src[perm[s]]
//
// After the sort, a cell's particles are contiguous, so contacts only read the 3x3 cells around
// a particle, and memory access everywhere follows space.

@group(0) @binding(0) var<uniform> u: SimU;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read> src_aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst_aux: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> perm: array<u32>;
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>>;

// Buckets larger than this keep atomic order (only in pathological pile-ups).
const MAX_FIX: u32 = 512u;

@compute @workgroup_size(256)
fn hash(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = flat_group(wid, nwg) * WG + lid;
    if (i >= u.n) { return; }
    let p = src[i].xy;
    var key = u.t;
    if (is_alive(p)) { key = bucket_of(cell_of(p)); }
    keys[i] = key;
    atomicAdd(&counts[key], 1u);
}

@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = flat_group(wid, nwg) * WG + lid;
    if (i >= u.n) { return; }
    let slot = atomicAdd(&counts[keys[i]], 1u);
    perm[slot] = i;
}

@compute @workgroup_size(256)
fn fix(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid, nwg) * WG + lid;
    if (b >= u.t) { return; }
    var start = 0u;
    if (b > 0u) { start = atomicLoad(&counts[b - 1u]); }
    let end = atomicLoad(&counts[b]);
    if (end - start < 2u || end - start > MAX_FIX) { return; }
    for (var a = start + 1u; a < end; a++) {
        let v = perm[a];
        var j = a;
        while (j > start && perm[j - 1u] > v) {
            perm[j] = perm[j - 1u];
            j--;
        }
        perm[j] = v;
    }
}

@compute @workgroup_size(256)
fn gather(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let s = flat_group(wid, nwg) * WG + lid;
    if (s >= u.n) { return; }
    let live = atomicLoad(&counts[u.t - 1u]);   // end of the last live bucket
    if (s < live) {
        let j = perm[s];
        dst[s] = src[j];
        dst_aux[s] = src_aux[j];
    } else {
        dst[s] = DEAD;
        dst_aux[s] = 0u;
    }
}
