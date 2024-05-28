#!/bin/bash

# Compare two files and count the differing lines
diff_lines=$(diff --new-line-format='%L' --unchanged-line-format= --old-line-format= <(sort "$1") <(sort "$2") | wc -l)
# Calculate the total number of lines in both files
total_lines=$(wc -l < "$1")
total_lines=$((total_lines + $(wc -l < "$2")))

# Calculate the percentage of similarity
similarity_percentage=$((100 - ((diff_lines * 100) / total_lines)))
echo "Similarity Percentage: $similarity_percentage%"

