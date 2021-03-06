#!/bin/bash

die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in mkdocs pdoc3; do
    command -v "$cmd" >/dev/null ||
        die "Missing $cmd; \`pip install $cmd\`"
done

PACKAGE="rlxnix"
PACKAGEROOT="$(dirname "$(realpath "$0")")"
BUILDROOT="$PACKAGEROOT/site"

echo
echo "Clean up documentation of $PACKAGE"
echo

rm -rf "$BUILDROOT" 2> /dev/null || true
mkdir -p "$BUILDROOT"

echo
echo "Building API reference docs for $PACKAGE"
echo

cd "$PACKAGEROOT"
pdoc3 --html --output-dir "$BUILDROOT/api-tmp" $PACKAGE
mv "$BUILDROOT/api-tmp/$PACKAGE" "docs/api"
rmdir "$BUILDROOT/api-tmp"
cd - > /dev/null

echo "Building general documentation for $PACKAGE"
echo

cd "$PACKAGEROOT"
mkdocs build --config-file mkdocs.yml --site-dir "$BUILDROOT" 
cd - > /dev/null

echo
echo "Done. Docs in:"
echo
echo "    file://$BUILDROOT/index.html"
echo