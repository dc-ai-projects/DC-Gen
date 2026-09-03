#!/usr/bin/env bash
set -euo pipefail

worktree_root="$(git rev-parse --show-toplevel)"
git_dir="$(git rev-parse --path-format=absolute --git-dir)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"

# pre-commit exposes the arguments from Git's post-checkout hook through these
# environment variables. The initial checkout from `git worktree add` has a
# null previous ref (all zeroes) and checkout type 1. A linked worktree also has
# a per-worktree Git directory that differs from the shared common directory.
if [[ "$git_dir" == "$common_git_dir" ]] ||
    [[ "${PRE_COMMIT_CHECKOUT_TYPE:-}" != "1" ]] ||
    [[ ! "${PRE_COMMIT_FROM_REF:-}" =~ ^0+$ ]]; then
    exit 0
fi

main_root="$(dirname "$common_git_dir")"
mkdir -p "$worktree_root/assets"

ensure_link() {
    local source="$1"
    local destination="$2"

    # Preserve every existing destination, including a broken symlink.
    if [[ ! -e "$destination" && ! -L "$destination" ]]; then
        ln -s "$source" "$destination"
    fi
}

ensure_link "$main_root/assets/data" "$worktree_root/assets/data"
ensure_link "$main_root/assets/dataset" "$worktree_root/assets/dataset"
ensure_link "$main_root/assets/checkpoints" "$worktree_root/assets/checkpoints"
ensure_link "$main_root/assets/venvs" "$worktree_root/assets/venvs"

# The trailing slash limits the glob to directories (and symlinks to
# directories). nullglob makes the loop a no-op when no exp* directory exists.
shopt -s nullglob
for source in "$main_root"/exp*/; do
    source="${source%/}"
    ensure_link "$source" "$worktree_root/${source##*/}"
done
