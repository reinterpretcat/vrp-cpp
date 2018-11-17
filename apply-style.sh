#!/bin/bash

# Applies clang-formatter to project source directories

CLANG_FORMAT="clang-format-6.0"

for DIRECTORY in src test tools
do
    echo "Formatting code under $DIRECTORY/"
    find "$DIRECTORY" \( -name '*.h' -or -name '*.hpp'  -or -name '*.cpp' -or -name '*.cc' -or -name '*.inl' \) -print0 | xargs -0 "$CLANG_FORMAT" -i
done
