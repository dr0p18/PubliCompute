#!/usr/bin/env bash
# Reconstruct and verify the pinned engine sources at DEST.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:?Usage: source.sh DEST}
DEST=$(realpath -m "$DEST")
read_pin() { sed -n "s/^$1=//p" "$HERE/BASE" | head -n1; }
UPSTREAM=${PUBLICOMPUTE_UPSTREAM:-$(read_pin UPSTREAM)}
COMMIT=$(read_pin COMMIT)
TREE=$(read_pin TREE)
[ -n "$UPSTREAM" ] && [ -n "$COMMIT" ] && [ -n "$TREE" ] || {
    echo "BASE must specify UPSTREAM, COMMIT and TREE." >&2
    exit 2
}

# Reserve the destination for a single reconstruction at a time.
LOCK="$DEST.lock"
mkdir "$LOCK" 2>/dev/null || { echo "Cannot acquire $LOCK." >&2; exit 3; }
TMP=$(mktemp -d "$DEST.tmp.XXXXXX") || { rmdir "$LOCK"; exit 3; }
trap 'rm -rf -- "$TMP" "$LOCK"' EXIT

# Local Git settings and hooks must not affect the reconstructed tree.
while IFS= read -r name; do unset "$name"; done < <(compgen -v GIT_ || true)
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
git_engine() {
    git -c core.hooksPath=/dev/null -c core.autocrlf=false -c core.fsmonitor=false \
        -c commit.gpgsign=false "$@"
}

# Reuse a clean checkout so incremental builds keep their source timestamps.
if [ -d "$DEST/.git" ] && [ "$(git_engine -C "$DEST" rev-parse -q --verify 'HEAD^{tree}' 2>/dev/null)" = "$TREE" ] \
    && [ -z "$(git_engine -C "$DEST" status --porcelain --ignored 2>/dev/null)" ]; then
    echo "Reused $DEST at tree $TREE"
    exit 0
fi

SRC="$TMP/src"
mkdir "$SRC"
cd "$SRC"
git_engine init -q --template= .
git_engine fetch -q --depth 1 "$UPSTREAM" "$COMMIT"
[ "$(git_engine rev-parse FETCH_HEAD)" = "$COMMIT" ] || {
    echo "Upstream did not return $COMMIT." >&2
    exit 4
}
git_engine checkout -q FETCH_HEAD
for patch in "$HERE"/patches/*.patch; do
    git_engine apply --index "$patch" || { echo "Cannot apply $patch." >&2; exit 5; }
done

ACTUAL_TREE=$(git_engine write-tree)
if [ "$ACTUAL_TREE" != "$TREE" ]; then
    echo "Engine tree mismatch: got $ACTUAL_TREE, expected $TREE." >&2
    exit 7
fi
git_engine -c user.name=PubliCompute -c user.email=engine commit -qm 'Apply engine patches'
if [ -n "$(git_engine status --porcelain --ignored)" ]; then
    echo "Engine checkout has local changes after applying the patches." >&2
    exit 6
fi

cd "$HERE"
# Publish the candidate only after the index and working tree pass verification.
rm -rf -- "$DEST"
mv -- "$SRC" "$DEST"
echo "Built $DEST at tree $ACTUAL_TREE"
