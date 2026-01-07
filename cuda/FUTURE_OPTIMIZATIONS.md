# Future Optimizations

Low-priority optimizations that have negligible impact on current performance but may be worth revisiting if bottlenecks shift.

---

## lgamma Overhead

**File**: `cuda/include/log_arithmetic.cuh` lines 242-255

**Current**: `lgamma()` called once per m in `compute_log_prefact_ymm()`.

```cuda
C lgamma_2m2 = lgamma(C(2) * m_c + C(2));  // lgamma(2m+2)
C lgamma_m1 = lgamma(m_c + C(1));           // lgamma(m+1)
```

**Why it doesn't matter**:
- Called once per m, not per (l,m) pair
- Total calls = l_max+1 per transform
- At ~50 cycles each: `3001 × 50 / 1.5e9 ≈ 0.1ms` for l_max=3000
- Transform takes ~100-200ms, so overhead is <0.1%

**Potential fixes if ever needed**:
1. Precompute lgamma table on host, upload once (~48KB for l_max=3000)
2. Use incremental recurrence: `lgamma(n+1) = lgamma(n) + log(n)`
3. Store in constant memory

**Why recurrence doesn't work well**:
- Kernel parallelizes by m (each block handles one m)
- Incremental recurrence creates serial dependency between blocks
- Would require either precomputation or two-pass approach

---

## Sign Tracking Memory Overhead

**Files**: `cuda/src/map2alm_v6.cu`, `cuda/include/log_arithmetic.cuh`

**Current**: Storing `int8_t sign` arrays alongside `log_Ylm` adds ~12% overhead to Phase 2.

```cuda
// Current: separate sign tracking per ring
R log_Ylm_prev1[RINGS_PER_LANE];     // 8 bytes × 64 = 512 bytes (f64)
int8_t sign_prev1[RINGS_PER_LANE];   // 1 byte × 64 = 64 bytes
// Overhead: 64 / 512 = 12.5%
```

**Why signs are necessary**:
- Ylm values can be negative (depends on θ)
- Sign pattern is NOT simply `(-1)^(l+m)` - it depends on the recurrence at each θ
- Each ring (different θ) has independent sign evolution
- Cannot be computed on-the-fly without the full recurrence

**Why it doesn't matter much**:
- 12% overhead, not "doubles bandwidth"
- Sign arrays are 64 bytes vs 512 bytes for log values (f64, 64 rings)
- Actual memory pressure is from log_Ylm arrays, not signs

**Potential optimizations (if ever needed)**:

### Option 1: Bit Packing

Pack 8 signs into one `uint8_t` (or 64 into one `uint64_t`):

```cuda
// Packing: store signs for rings 0-7 in one byte
uint8_t packed_signs = 0;
for (int i = 0; i < 8; i++) {
    // Store 1 for negative, 0 for positive
    packed_signs |= ((sign[i] < 0) ? 1 : 0) << i;
}

// Unpacking: extract sign for ring i
int get_sign(uint8_t packed, int i) {
    return ((packed >> i) & 1) ? -1 : 1;
}
```

**Trade-offs**:
- Memory: 64 bytes → 8 bytes (87.5% reduction for signs)
- Compute: Adds bit shift/mask operations per access
- Overall impact: Saves ~64 bytes but adds ~2-4 ops per Ylm access
- Net effect: Probably worse due to bit manipulation overhead

### Option 2: Pack Sign into Float

Use the sign bit of the float to encode Ylm sign:

```cuda
// Store: log(|Ylm|) with sign encoded in float's sign bit
R signed_log_ylm = (sign < 0) ? -log_ylm : log_ylm;

// Extract:
int8_t sign = (signed_log_ylm < 0) ? -1 : 1;
R log_ylm = fabs(signed_log_ylm);
```

**Trade-offs**:
- Zero extra storage
- Adds `fabs()` call per access (~1 cycle)
- Risk: Confusing semantics (log value encodes sign of original value)

### Option 3: Store as Float (Current Best Alternative)

Replace `int8_t sign` with `float sign` (values: -1.0f or +1.0f):

```cuda
R sign_prev1[RINGS_PER_LANE];  // Now matches log_Ylm type
```

**Trade-offs**:
- Memory: 64 bytes → 256 bytes (f32) or 512 bytes (f64)
- Compute: Eliminates int↔float conversions in accumulation
- Better vectorization (same type throughout)

**Recommendation**: Keep current `int8_t` approach. The 12% overhead is acceptable, and alternatives either add compute or increase memory.

---
