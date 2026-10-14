#!/usr/bin/env bash
# quickstart.sh - one command that sets up and verifies this package, run from the package root:
#     bash quickstart.sh [--no-sim] [--cuda] [--skip-install] [--help]
# It finds a Python 3.10, creates ./.venv, clones the upstream LeWM code (./le-wm) at the pinned commit,
# installs torch (CPU wheel unless --cuda) + requirements.txt + env/requirements-env.txt, verifies SHA256SUMS,
# runs demo.py (must print PASS), probes a MuJoCo rendering backend and runs the simulator demo
# (must print DEMO OK; --no-sim skips the simulator part). Every step prints a heading, then OK, or FAIL
# with what is wrong and what to do. Re-running continues where it stopped: the venv and the clone are
# reused, installs are skipped when already satisfied, the demos run again (new output directory each time).
# Tested on Linux and WSL (Ubuntu 24.04). Windows: run it inside WSL (Ubuntu), from a clone made inside WSL.
# macOS is untested: Apple silicon should work (bash 3.2 is enough; without GNU timeout - brew install coreutils -
# a step that hangs is not detected); Intel Macs cannot (torch 2.14.0 has no x86_64 macOS wheels).
# First run ~3 min (2.5 GB of packages; ~1.5 min with a warm pip cache), later runs ~40 s.
# Needs: bash 3.2+, git, network, sha256sum or shasum; timeout (coreutils) recommended.
IFS= read -r _l < "$0"; case "$_l" in *$'\r'*) echo "FAIL quickstart.sh has CRLF line endings (checked out with Git for Windows / core.autocrlf=true?)"; echo "  -> clone inside WSL or Linux, or run: git config core.autocrlf false && git checkout -- .   then re-run"; exit 1;; esac #
set -u

UPSTREAM_URL=https://github.com/lucas-maes/le-wm.git
UPSTREAM_COMMIT=8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
TORCH_PINS="torch==2.14.0 torchvision==0.29.0"
NO_SIM=0; CUDA=0; SKIP_INSTALL=0; SUMMARY=(); T0=$SECONDS; OUT=""; GL=""; NO_TIMEOUT=0
for arg in "$@"; do
    case "$arg" in
        --no-sim) NO_SIM=1 ;;  --cuda) CUDA=1 ;;  --skip-install) SKIP_INSTALL=1 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "FAIL unknown option '$arg' (see: bash quickstart.sh --help)"; exit 2 ;;
    esac
done

step() { STEP_NAME="$2"; STEP_T=$SECONDS; echo; echo "== [$1] $2${3:+: $3}"; }
ok()   { echo "OK ${1:-}"; SUMMARY+=("$(printf '  %-28s OK  %4d s  %s' "$STEP_NAME" $((SECONDS - STEP_T)) "${1:-}")"); }
fail() { echo "FAIL $1"; echo "  -> $2"; exit 1; }
run_logged() { local log=$1; shift; "$@" 2>&1 | tee "$log"; return "${PIPESTATUS[0]}"; }   # show + keep a copy
if ! command -v timeout >/dev/null 2>&1; then
    if command -v gtimeout >/dev/null 2>&1; then timeout() { gtimeout "$@"; }                 # macOS + coreutils
    else timeout() { shift; "$@"; }; NO_TIMEOUT=1; fi                                         # no time limit at all
fi
TMPLOG=$(mktemp -t quickstart.XXXXXX) || { echo "FAIL cannot create a temp file"; exit 1; }
trap 'rm -f "$TMPLOG"' EXIT

# ---- 0. location -----------------------------------------------------------------------------------------------
step 0 "location check"
[ -f model_state.pt ] && [ -f requirements.txt ] && [ -f demo.py ] ||
    fail "model_state.pt / requirements.txt / demo.py are not in the current directory ($PWD)" \
         "cd into the package root (the directory that contains model_state.pt), then: bash quickstart.sh"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) fail "native Windows shell detected" "run this script inside WSL (Ubuntu)";; esac
command -v git >/dev/null 2>&1 || fail "git is not installed" "Debian/Ubuntu: sudo apt install git; macOS: xcode-select --install; then re-run"
[ $NO_TIMEOUT = 0 ] || echo "   note: no 'timeout' command, so a step that hangs cannot be detected (macOS: brew install coreutils)"
ok "$PWD"

# ---- 1. Python 3.10 --------------------------------------------------------------------------------------------
step 1 "Python 3.10"
pyver() { "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null; }
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python; HOW="existing .venv"
elif [ -n "${PYTHON310:-}" ]; then
    PY=$PYTHON310; HOW="\$PYTHON310"
elif command -v python3.10 >/dev/null 2>&1; then
    PY=$(command -v python3.10); HOW="python3.10 on PATH"
else
    UV=$(command -v uv 2>/dev/null || true); [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV=$HOME/.local/bin/uv
    [ -n "$UV" ] || fail "no python3.10 on PATH and no uv (Python 3.12 etc. are untested, 3.10 is required)" \
        "either  curl -LsSf https://astral.sh/uv/install.sh | sh  and re-run (the script then gets 3.10 through uv), or  pyenv install 3.10  and  export PYTHON310=/path/to/python3.10"
    PY=$("$UV" python find 3.10 2>/dev/null) || { echo "   uv: installing CPython 3.10 ..."; "$UV" python install 3.10 ||
        fail "uv python install 3.10 failed" "check the network, or export PYTHON310=/path/to/python3.10 and re-run"; PY=$("$UV" python find 3.10) ||
        fail "uv cannot find the 3.10 it just installed" "run  uv python find 3.10  and export PYTHON310=<that path>"; }
    HOW="uv"
fi
VER=$(pyver "$PY") || fail "$PY does not run ($HOW)" "export PYTHON310=/path/to/a/working/python3.10 (or rm -rf .venv) and re-run"
case "$VER" in 3.10.*) ;; *) fail "$PY is Python $VER, not 3.10 ($HOW)" \
    "export PYTHON310=/path/to/python3.10 (or rm -rf .venv if it was made with another version) and re-run";; esac
ok "$PY  (CPython $VER, $HOW)"

# ---- 2. virtual environment ------------------------------------------------------------------------------------
step 2 "virtual environment ./.venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv ||
    fail "$PY -m venv .venv failed" "Debian/Ubuntu: sudo apt install python3.10-venv; then rm -rf .venv and re-run"
VPY=.venv/bin/python; PIP=(.venv/bin/python -m pip)
"${PIP[@]}" --version >/dev/null 2>&1 || "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 ||
    fail ".venv has no pip" "rm -rf .venv and re-run (Debian/Ubuntu: sudo apt install python3.10-venv first)"
ok "$VPY (CPython $(pyver "$VPY")); the script calls .venv/bin/python directly, no activation needed"

# ---- 3. upstream LeWM code -------------------------------------------------------------------------------------
step 3 "upstream LeWM code ./le-wm" "commit $UPSTREAM_COMMIT"
[ -e le-wm ] || git clone --quiet "$UPSTREAM_URL" le-wm ||
    fail "git clone $UPSTREAM_URL failed" "check the network / proxy (git must reach github.com), rm -rf le-wm, re-run"
HEAD=$(git -C le-wm rev-parse HEAD 2>/dev/null) ||
    fail "./le-wm exists but is not a git checkout" "rm -rf le-wm and re-run (the script clones $UPSTREAM_URL itself)"
if [ "$HEAD" != "$UPSTREAM_COMMIT" ]; then
    echo "   le-wm is at $HEAD, checking out the pinned commit ..."
    { git -C le-wm fetch --quiet && git -C le-wm checkout --quiet "$UPSTREAM_COMMIT"; } ||
        fail "could not check out $UPSTREAM_COMMIT in ./le-wm" "rm -rf le-wm and re-run (it must be a clone of $UPSTREAM_URL)"
    HEAD=$(git -C le-wm rev-parse HEAD); [ "$HEAD" = "$UPSTREAM_COMMIT" ] || fail "le-wm HEAD is still $HEAD" "rm -rf le-wm and re-run"
fi
for f in jepa.py module.py config/train/model/lewm.yaml; do
    [ -f "le-wm/$f" ] || fail "./le-wm is incomplete: $f is missing" "rm -rf le-wm and re-run"
done
git -C le-wm diff --quiet -- jepa.py module.py config/train/model/lewm.yaml ||
    fail "./le-wm has local changes in the files the loader fingerprints" "git -C le-wm checkout -- .  (or rm -rf le-wm) and re-run"
ok "commit $HEAD, fingerprinted files present and unmodified"

# ---- 4. packages -----------------------------------------------------------------------------------------------
step 4 "packages" "torch $([ $CUDA = 1 ] && echo CUDA || echo CPU) + requirements.txt$([ $NO_SIM = 1 ] || echo ' + env/requirements-env.txt')"
REQS=(-r requirements.txt); [ $NO_SIM = 1 ] || REQS+=(-r env/requirements-env.txt)
pins_satisfied() {  # exit 0 when every  name==version  /  name @ url  line of the given files is installed
    "$VPY" - "$@" <<'PYEOF'
import importlib.metadata as m, re, sys
bad = []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.split("#")[0].split(";")[0].strip()
        if not line or line.startswith("-"): continue
        name, want = (line.split("@")[0], None) if "@" in line else (line.split("==")[0], line.split("==")[1].strip())
        name = re.sub(r"\[.*\]", "", name).strip()
        try: have = m.version(name)
        except m.PackageNotFoundError: bad.append(name + " missing"); continue
        if want and have.split("+")[0] != want: bad.append(f"{name} {have}!={want}")
print("   not yet satisfied: " + ", ".join(bad) if bad else "   all pins already installed"); sys.exit(1 if bad else 0)
PYEOF
}
if [ $SKIP_INSTALL = 1 ]; then
    echo "   --skip-install: pip not run"
else
    "${PIP[@]}" install --quiet --upgrade pip || fail "pip self-upgrade failed" "check the network / proxy and re-run (or --skip-install if everything is installed)"
    if "$VPY" -c 'import torch, torchvision' >/dev/null 2>&1 && pins_satisfied <(printf '%s\n' $TORCH_PINS) >/dev/null; then
        echo "   torch $("$VPY" -c 'import torch; print(torch.__version__)') already installed, keeping it"
    elif [ $CUDA = 1 ]; then
        "${PIP[@]}" install --quiet $TORCH_PINS || fail "pip install $TORCH_PINS (CUDA wheels from PyPI) failed" "check the network and re-run (or without --cuda)"
    else
        "${PIP[@]}" install --quiet $TORCH_PINS --index-url https://download.pytorch.org/whl/cpu ||
            fail "pip install of the torch CPU wheels failed" "check that download.pytorch.org is reachable and re-run ('no matching distribution' on macOS: only Apple silicon has torch 2.14.0 wheels)"
    fi
    pins_satisfied requirements.txt $([ $NO_SIM = 1 ] || echo env/requirements-env.txt) ||
        { echo "   pip install ${REQS[*]} ..."; "${PIP[@]}" install --quiet "${REQS[@]}" ||
          fail "pip install ${REQS[*]} failed" "read pip's message above (network? git for stable-worldmodel? disk space?), fix it and re-run"; }
    "${PIP[@]}" check || fail "pip check found broken requirements" "rm -rf .venv and re-run (the pins are mutually consistent in a clean venv)"
fi
"$VPY" -c 'import torch, transformers, stable_pretraining, hydra, einops, PIL' 2>"$TMPLOG" ||
    fail "the model stack does not import: $(tail -1 "$TMPLOG")" "re-run without --skip-install (or rm -rf .venv and re-run)"
[ $NO_SIM = 1 ] || MUJOCO_GL=disable "$VPY" -c 'import mujoco, ogbench, stable_worldmodel, dm_control' 2>"$TMPLOG" ||   # GL is step 7's job
    fail "the simulator stack does not import: $(tail -1 "$TMPLOG")" "re-run without --skip-install (or rm -rf .venv and re-run)"
ok "torch $("$VPY" -c 'import torch; print(torch.__version__)'), $("${PIP[@]}" list 2>/dev/null | tail -n +3 | wc -l | tr -d ' ') packages"

# ---- 5. checksums ----------------------------------------------------------------------------------------------
step 5 "checksums" "SHA256SUMS"
if command -v sha256sum >/dev/null 2>&1; then SUMCMD=(sha256sum -c SHA256SUMS)
elif command -v shasum >/dev/null 2>&1; then SUMCMD=(shasum -a 256 -c SHA256SUMS)
else fail "neither sha256sum nor shasum is available" "install coreutils (Linux) or perl (shasum; macOS has it) and re-run"; fi
"${SUMCMD[@]}" >"$TMPLOG" 2>&1 || { grep -v ': OK$' "$TMPLOG" | sed 's/^/   /';
    fail "some files are modified or missing (FAILED lines above)" "restore them from the repository (git checkout -- <file>, or re-clone) and re-run; a changed model_state.pt or heads/tilt_head.pt must not be used"; }
ok "$(grep -c ': OK$' "$TMPLOG") files verified with ${SUMCMD[0]}"

# ---- 6. model demo ---------------------------------------------------------------------------------------------
step 6 "model demo" "$VPY demo.py --upstream ./le-wm"
run_logged "$TMPLOG" timeout 900 "$VPY" demo.py --upstream ./le-wm; RC=$?
grep -q '^PASS' "$TMPLOG" || fail "demo.py did not print PASS (exit $RC)" \
    "read the FAIL line / traceback above: 'Missing upstream LeWM file' or 'differs from the code' -> rm -rf le-wm and re-run; an ImportError -> re-run without --skip-install"
ok "$(grep '^PASS' "$TMPLOG")"

if [ $NO_SIM = 0 ]; then
# ---- 7. rendering backend --------------------------------------------------------------------------------------
GLDESC="probing egl, osmesa, glfw"; [ -z "${MUJOCO_GL:-}" ] || GLDESC="exported MUJOCO_GL=$MUJOCO_GL"
step 7 "MuJoCo rendering backend" "$GLDESC"
[ $NO_TIMEOUT = 0 ] || echo "   note: no 'timeout' command; a probe that hangs (glfw without a display) must be stopped with Ctrl-C, then export MUJOCO_GL=egl or osmesa"
GLPROBE='import mujoco; m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"1 1 1\"/></worldbody></mujoco>"); r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render(); print("ok")'
for cand in ${MUJOCO_GL:-egl osmesa glfw}; do
    if MUJOCO_GL=$cand timeout 60 "$VPY" -c "$GLPROBE" >"$TMPLOG" 2>&1; then GL=$cand; break; fi
    echo "   $cand: no ($(grep -Ev '^ |^Traceback' "$TMPLOG" | tail -1 | cut -c1-150))"
done
[ -n "$GL" ] || fail "no working MuJoCo rendering backend${MUJOCO_GL:+ (your exported MUJOCO_GL=$MUJOCO_GL fails; unset it to let the script probe)}" \
    "egl needs a GPU driver that exposes EGL (Debian/Ubuntu: sudo apt install libegl1 + NVIDIA/Mesa driver); osmesa needs sudo apt install libosmesa6; glfw needs a display (DISPLAY / WSLg / macOS)"
ok "MUJOCO_GL=$GL"

# ---- 8. simulator demo -----------------------------------------------------------------------------------------
OUT=env_demo_out; n=2; while [ -e "$OUT" ]; do OUT=env_demo_out_$n; n=$((n + 1)); done
step 8 "simulator demo" "MUJOCO_GL=$GL $VPY env/env_demo.py --upstream ./le-wm --out $OUT"
MUJOCO_GL=$GL run_logged "$TMPLOG" timeout 1800 "$VPY" env/env_demo.py --upstream ./le-wm --out "$OUT"; RC=$?
if ! grep -q '^DEMO OK' "$TMPLOG"; then
    grep -q '^DEMO RAN' "$TMPLOG" && fail "the simulator demo completed but the model missed its verdict on this machine's physics (summary above; outputs in $OUT)" \
        "not an install problem but a model-quality report: compare with our run via  --compare examples/env_demo_reference.npz  (README, 'Running the model in the simulator')"
    fail "env/env_demo.py did not print DEMO OK (exit $RC)" "read the FAIL line / traceback above; a rendering error -> export MUJOCO_GL=<egl|osmesa|glfw> and re-run; a timeout -> re-run on a less loaded machine"
fi
ok "$(grep '^DEMO OK' "$TMPLOG" | cut -c1-110) ..."
fi

# ---- 9. summary ------------------------------------------------------------------------------------------------
echo; echo "== summary (total $((SECONDS - T0)) s)"
printf '%s\n' "${SUMMARY[@]}"
echo "  python: $PWD/$VPY   upstream: $PWD/le-wm"
if [ $NO_SIM = 1 ]; then
    echo "  simulator demo skipped (--no-sim); re-run without it or see README 'Running the model in the simulator'"
else
    echo "  simulator outputs: $PWD/$OUT  (summary.json, steps.csv, strip.png, tilt_chart.png, rollout.gif, record.npz)"
    if grep -qi microsoft /proc/version 2>/dev/null; then echo "  view rollout.gif (WSL): explorer.exe $OUT"
    elif command -v xdg-open >/dev/null 2>&1; then echo "  view rollout.gif: xdg-open $OUT/rollout.gif"
    elif [ "$(uname -s)" = Darwin ]; then echo "  view rollout.gif: open $OUT/rollout.gif"
    else echo "  view rollout.gif: copy $OUT/rollout.gif to a machine with an image viewer or browser"; fi
fi
echo "ALL STEPS OK"
