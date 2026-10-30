#!/usr/bin/env bash
# Run TestIDocBuild A/B profiles for Priority 1 regression investigation.
set -euo pipefail

REPEATS="${REPEATS:-3}"
BM_ROOT="${BM_ROOT:-/tmp/nornir-test-output}"
BM_REPO="/workspace/nornir-buildmanager"
IMG_REPO="/workspace/nornir-imageregistration"
LDC="${IMG_REPO}/nornir_imageregistration/local_distortion_correction.py"
LDC_BACKUP="${LDC}.ab_investigation.bak"
COMPARE="${IMG_REPO}/scripts/compare_idoc_profile_runs.py"

run_case() {
  local label="$1"
  shift
  local i
  for ((i = 1; i <= REPEATS; i++)); do
    local run_label="${label}_run${i}"
    echo "===== ${run_label} ====="
    rm -rf "${BM_ROOT}/TestIDocBuild"
    (
      cd "${BM_REPO}"
      NORNIR_HEADLESS=1 PROFILE=1 NORNIR_SKIP_MOSAIC_PLOTS=1 "$@" \
        pytest tests/pipeline/test_idoc.py::TestIDocBuild::test_i_doc_build_test -q --tb=line
    )
    python3 "${COMPARE}" --archive "${BM_ROOT}/TestIDocBuild" --label "${run_label}" --archive-root "${BM_ROOT}"
  done
}

restore_ldc() {
  if [[ -f "${LDC_BACKUP}" ]]; then
    mv -f "${LDC_BACKUP}" "${LDC}"
  fi
}
trap restore_ldc EXIT

echo "Case 1: fixed Priority 1 (cache + memory fixes)"
run_case "p1_fixed" env NORNIR_LOG_GPU_MEM=1

echo "Case 2: prewarp cache disabled (1B retained)"
run_case "p1_no_cache" env NORNIR_LOG_GPU_MEM=1 NORNIR_DISABLE_PREWARP_CACHE=1

echo "Case 3: full Priority 1 revert (committed HEAD)"
cp -f "${LDC}" "${LDC_BACKUP}"
git -C "${IMG_REPO}" show HEAD:nornir_imageregistration/local_distortion_correction.py > "${LDC}"
run_case "p1_head"

echo "Restoring ${LDC}"
restore_ldc
trap - EXIT

python3 "${COMPARE}" --summarize "${BM_ROOT}/TestIDocBuild-after-transfer-opt" 2>/dev/null || true
for label in p1_fixed_run1 p1_fixed_run2 p1_fixed_run3 p1_no_cache_run1 p1_no_cache_run2 p1_no_cache_run3 p1_head_run1 p1_head_run2 p1_head_run3; do
  if [[ -d "${BM_ROOT}/TestIDocBuild-${label}" ]]; then
    python3 "${COMPARE}" --summarize "${BM_ROOT}/TestIDocBuild-${label}"
  fi
done
