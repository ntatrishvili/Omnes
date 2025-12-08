#!/usr/bin/env bash
# Generate docs/*.md from app/*.py, skipping __init__.py and tests/
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

find app -type f -name "*.py" ! -name "__init__.py" ! -path "*/tests/*" -print0 |
while IFS= read -r -d '' src; do
    rel=${src#app/}                       # path under app/
    modpath=${rel%.py}                    # remove .py
    module="app.${modpath//\//.}"         # add app. prefix and replace / with .
    out="docs/${modpath}.md"              # docs/<same structure>.md
    mkdir -p "$(dirname "$out")"
    if [ -e "$out" ]; then
        printf "Skipping existing: %s\n" "$out"
        continue
    fi
    cat > "$out" <<EOF
::: ${module}
    options:
        show_root_heading: true
        show_source: false
EOF
    printf "Created: %s\n" "$out"
done