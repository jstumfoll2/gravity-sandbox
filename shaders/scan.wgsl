// Exclusive prefix sum, 1024 elements per workgroup. sim.rs runs scan_block on every level
// (each level holds the block totals of the one below) and then scan_add back down.

struct ScanU { n: u32, _p0: u32, _p1: u32, _p2: u32 }

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;
@group(0) @binding(2) var<uniform> su: ScanU;

var<workgroup> s: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_block(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let blk = wid.y * nwg.x + wid.x;
    let base = blk * 1024u + lid * 4u;
    var v: array<u32, 4>;
    var total = 0u;
    for (var k = 0u; k < 4u; k++) {
        var x = 0u;
        if (base + k < su.n) { x = data[base + k]; }
        v[k] = total;
        total += x;
    }
    s[lid] = total;
    workgroupBarrier();
    // Hillis-Steele inclusive scan of the 256 thread totals
    for (var off = 1u; off < 256u; off <<= 1u) {
        var add = 0u;
        if (lid >= off) { add = s[lid - off]; }
        workgroupBarrier();
        s[lid] += add;
        workgroupBarrier();
    }
    let before = s[lid] - total;
    for (var k = 0u; k < 4u; k++) {
        if (base + k < su.n) { data[base + k] = v[k] + before; }
    }
    if (lid == 255u) { sums[blk] = s[255]; }
}

@compute @workgroup_size(256)
fn scan_add(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let blk = wid.y * nwg.x + wid.x;
    if (blk == 0u) { return; }
    let add = sums[blk];
    let base = blk * 1024u + lid * 4u;
    for (var k = 0u; k < 4u; k++) {
        if (base + k < su.n) { data[base + k] += add; }
    }
}
