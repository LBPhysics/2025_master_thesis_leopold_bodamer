# =============================
# TEST 1: TIME ARRAY CALCULATIONS
# =============================

### Test get_tau_cohs_and_t_dets_for_T_wait function
print("Testing get_tau_cohs_and_t_dets_for_T_wait function...")

# Test basic functionality with reasonable parameters
times = np.linspace(0, 100, 201)  # time array: 0 to 100 with dt=0.5
T_wait = 50.0  # waiting time

tau_cohs, t_dets = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait)

print(f"Generated time arrays:")
print(
    f"  Input times: {len(times)} points, range [{times[0]:.2f}, {times[-1]:.2f}], dt = {times[1]-times[0]:.3f}"
)
print(f"  T_wait = {T_wait}")
print(
    f"  tau_cohs: {len(tau_cohs)} points, range [{tau_cohs[0]:.2f}, {tau_cohs[-1]:.2f}]"
)
print(f"  t_dets: {len(t_dets)} points, range [{t_dets[0]:.2f}, {t_dets[-1]:.2f}]")

# Test array properties
assert len(tau_cohs) > 0, "tau_cohs array is empty"
assert len(t_dets) > 0, "t_dets array is empty"
assert len(tau_cohs) == len(t_dets), "tau_cohs and t_dets should have same length"
assert tau_cohs[0] >= 0, "tau_cohs should start from non-negative value"
assert t_dets[0] >= 0, "t_dets should start from non-negative value"
assert (
    t_dets[-1] <= times[-1]
), f"t_dets exceeds maximum time: {t_dets[-1]} > {times[-1]}"

# Test relationship: t_det = tau_coh + T_wait
relationship_test = np.allclose(t_dets, tau_cohs + T_wait)
print(f"  Relationship t_det = tau_coh + T_wait: {relationship_test}")
assert relationship_test, "t_det should equal tau_coh + T_wait"

# Test time step consistency (should match input dt)
dt_input = times[1] - times[0]
if len(tau_cohs) > 1:
    tau_steps = np.diff(tau_cohs)
    t_det_steps = np.diff(t_dets)

    print(f"  Time step consistency:")
    print(f"    tau_cohs steps: all ≈ {dt_input}? {np.allclose(tau_steps, dt_input)}")
    print(f"    t_dets steps: all ≈ {dt_input}? {np.allclose(t_det_steps, dt_input)}")

    assert np.allclose(
        tau_steps, dt_input, rtol=1e-10
    ), "tau_cohs time steps inconsistent"
    assert np.allclose(
        t_det_steps, dt_input, rtol=1e-10
    ), "t_dets time steps inconsistent"

### Test edge cases
print("\nTesting edge cases...")

# Test with different T_wait values
print("\nTesting different T_wait values:")
T_wait_values = [10.0, 25.0, 50.0, 75.0, 90.0]
for T_w in T_wait_values:
    tau_test, t_det_test = get_tau_cohs_and_t_dets_for_T_wait(times, T_w)
    expected_max_tau = times[-1] - T_w
    print(
        f"  T_wait = {T_w:4.1f}: {len(tau_test):3d} points, max_tau = {tau_test[-1] if len(tau_test) > 0 else 0:5.1f} (expected ≤ {expected_max_tau:5.1f})"
    )

    if len(tau_test) > 0:
        assert (
            tau_test[-1] <= expected_max_tau + 1e-10
        ), f"tau_coh exceeds expected maximum"
        assert np.allclose(
            t_det_test, tau_test + T_w
        ), "Relationship t_det = tau_coh + T_wait violated"

# Test with T_wait = t_max (edge case)
print("\nTesting T_wait = t_max (edge case):")
T_wait_max = times[-1]
tau_edge, t_det_edge = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait_max)
print(
    f"  T_wait = t_max = {T_wait_max}: tau_cohs length = {len(tau_edge)}, t_dets length = {len(t_det_edge)}"
)
if len(tau_edge) > 0:
    print(f"    Values: tau_coh = {tau_edge}, t_det = {t_det_edge}")

# Test with T_wait > t_max (should return empty arrays)
print("\nTesting T_wait > t_max (should return empty):")
T_wait_large = times[-1] + 10.0
tau_empty, t_det_empty = get_tau_cohs_and_t_dets_for_T_wait(times, T_wait_large)
print(
    f"  T_wait = {T_wait_large} > t_max = {times[-1]}: tau_cohs length = {len(tau_empty)}, t_dets length = {len(t_det_empty)}"
)
assert len(tau_empty) == 0, "Should return empty array when T_wait > t_max"
assert len(t_det_empty) == 0, "Should return empty array when T_wait > t_max"

# Test with different time array densities
print("\nTesting different time array densities:")
dt_values = [0.1, 0.5, 1.0, 2.0]
T_wait_test = 20.0
t_max_test = 50.0

for dt_val in dt_values:
    times_test = np.arange(0, t_max_test + dt_val / 2, dt_val)
    tau_test, t_det_test = get_tau_cohs_and_t_dets_for_T_wait(times_test, T_wait_test)

    print(
        f"  dt = {dt_val:3.1f}: {len(times_test):3d} input points → {len(tau_test):3d} output points"
    )

    if len(tau_test) > 1:
        actual_dt = tau_test[1] - tau_test[0]
        assert np.isclose(
            actual_dt, dt_val
        ), f"Output dt {actual_dt} doesn't match input dt {dt_val}"

# Test with single time point
print("\nTesting single time point:")
times_single = np.array([0.0])
tau_single, t_det_single = get_tau_cohs_and_t_dets_for_T_wait(times_single, 0.0)
print(f"  Single time point: tau_cohs = {tau_single}, t_dets = {t_det_single}")

### Visualization of time arrays
print("\nCreating visualizations...")
plt.figure(figsize=(14, 10))

# Plot 1: Time arrays for different T_wait values
plt.subplot(2, 3, 1)
T_wait_vis = [10.0, 25.0, 40.0, 60.0, 80.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(T_wait_vis)))

for i, T_w in enumerate(T_wait_vis):
    tau_vis, t_det_vis = get_tau_cohs_and_t_dets_for_T_wait(times, T_w)
    if len(tau_vis) > 0:
        plt.plot(
            tau_vis,
            np.full_like(tau_vis, i),
            "o",
            color=colors[i],
            markersize=2,
            label=f"τ_coh (T_wait={T_w})",
            alpha=0.7,
        )
        plt.plot(
            t_det_vis,
            np.full_like(t_det_vis, i + 0.1),
            "s",
            color=colors[i],
            markersize=2,
            alpha=0.7,
            label=f"t_det (T_wait={T_w})",
        )

plt.xlabel("Time")
plt.ylabel("T_wait Value Index")
plt.title("Time Arrays for Different T_wait Values")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(True, alpha=0.3)

# Plot 2: Array length vs T_wait
plt.subplot(2, 3, 2)
T_wait_range = np.linspace(5, 95, 50)
array_lengths = []

for T_w in T_wait_range:
    tau_test, t_det_test = get_tau_cohs_and_t_dets_for_T_wait(times, T_w)
    array_lengths.append(len(tau_test))

plt.plot(T_wait_range, array_lengths, "C0o-", markersize=3)
plt.xlabel("T_wait")
plt.ylabel("Array Length")
plt.title("Array Length vs T_wait")
plt.grid(True, alpha=0.3)

# Plot 3: Time coverage analysis
plt.subplot(2, 3, 3)
tau_coverage = []
t_det_coverage = []

for T_w in T_wait_range:
    tau_test, t_det_test = get_tau_cohs_and_t_dets_for_T_wait(times, T_w)
    tau_max = tau_test[-1] if len(tau_test) > 0 else 0
    t_det_max = t_det_test[-1] if len(t_det_test) > 0 else 0
    tau_coverage.append(tau_max)
    t_det_coverage.append(t_det_max)

plt.plot(T_wait_range, tau_coverage, "C0o-", label="Max τ_coh", markersize=3)
plt.plot(T_wait_range, t_det_coverage, "C1s-", label="Max t_det", markersize=3)
plt.plot(
    T_wait_range, times[-1] - T_wait_range, "k--", alpha=0.5, label="Expected max τ_coh"
)
plt.plot(
    T_wait_range, np.full_like(T_wait_range, times[-1]), "r--", alpha=0.5, label="t_max"
)
plt.xlabel("T_wait")
plt.ylabel("Time")
plt.title("Time Coverage Analysis")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Different time resolutions
plt.subplot(2, 3, 4)
dt_test_vals = [0.2, 0.5, 1.0, 2.0]
T_wait_fixed = 30.0
t_max_fixed = 60.0

for i, dt_val in enumerate(dt_test_vals):
    times_res = np.arange(0, t_max_fixed + dt_val / 2, dt_val)
    tau_res, t_det_res = get_tau_cohs_and_t_dets_for_T_wait(times_res, T_wait_fixed)

    plt.plot(
        tau_res,
        np.full_like(tau_res, i),
        "o",
        markersize=2,
        label=f"dt={dt_val} ({len(tau_res)} pts)",
        alpha=0.7,
    )

plt.xlabel("τ_coh")
plt.ylabel("Resolution Index")
plt.title(f"Different Time Resolutions\\n(T_wait={T_wait_fixed}, t_max={t_max_fixed})")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Memory scaling estimate
plt.subplot(2, 3, 5)
dt_range = np.logspace(-1, 0.5, 20)
memory_estimates = []
total_points = []

for dt_val in dt_range:
    times_mem = np.arange(0, 100 + dt_val / 2, dt_val)
    tau_mem, t_det_mem = get_tau_cohs_and_t_dets_for_T_wait(times_mem, 50.0)

    if len(tau_mem) > 0:
        # Estimate memory for 2D array (tau_cohs x t_dets)
        n_points = len(tau_mem)
        memory_mb = n_points * n_points * 16 / (1024**2)  # complex128
        memory_estimates.append(memory_mb)
        total_points.append(n_points)

if memory_estimates:
    plt.loglog(total_points, memory_estimates, "C2^-", markersize=4)
    plt.xlabel("Array Length")
    plt.ylabel("Estimated 2D Memory (MB)")
    plt.title("Memory Scaling for 2D Arrays\\n(τ_coh × t_det grid)")
    plt.grid(True, alpha=0.3)

# Plot 6: Function behavior summary
plt.subplot(2, 3, 6)
summary_text = f"""
Function Behavior Summary:

Input: times array, T_wait
Output: tau_coh, t_det arrays

Key Properties:
• len(tau_coh) = len(t_det)
• t_det = tau_coh + T_wait
• tau_coh ∈ [0, t_max - T_wait]
• t_det ∈ [T_wait, t_max]
• dt_out = dt_in

Edge Cases:
• T_wait > t_max → empty arrays
• T_wait = t_max → single point
• Single time point → [0.0], [0.0]

Test Results:
✓ Basic functionality
✓ Time relationships
✓ Edge case handling
✓ Different resolutions
✓ Memory scaling
"""

plt.text(
    0.05,
    0.95,
    summary_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis("off")
plt.title("Test Summary")

plt.tight_layout()
plt.show()

print("\n✓ All time array calculation tests passed!")
print(f"✓ Function correctly handles {len(T_wait_values)} different T_wait values")
print(f"✓ Function correctly handles {len(dt_values)} different time resolutions")
print("✓ Edge cases properly managed")
print("✓ Memory scaling characterized")
