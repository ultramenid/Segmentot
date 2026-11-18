#!/bin/sh
# Build the release zip for plugins.qgis.org: Segmentot-<version>.zip, next to the plugin folder (or in $1).
# Only what the plugin needs at run time goes in: no tests, no docs, no scripts, no caches.
set -e
cd "$(dirname "$0")/.."
version=$(sed -n 's/^version=//p' metadata.txt)
out="$(cd "${1:-..}" && pwd)/Segmentot-$version.zip"
tmp=$(mktemp -d)
mkdir "$tmp/Segmentot"
cp __init__.py metadata.txt LICENSE README.md "$tmp/Segmentot/"
cp -R core gui server resources "$tmp/Segmentot/"
find "$tmp" -name __pycache__ -prune -exec rm -rf {} +
rm -f "$out"
(cd "$tmp" && zip -qr -X "$out" Segmentot)
rm -rf "$tmp"
echo "$out"
