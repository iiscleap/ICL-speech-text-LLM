#!/bin/bash

# Navigate to the ICL directory
cd /data2/neeraja/neeraja/code/ICL

# Get current date for commit message
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Add all changes (including deletions)
git add -A

# Only commit if there are changes
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Auto-commit: ${DATE}"
    echo "Changes committed on ${DATE}"
else
    echo "No changes to commit on ${DATE}"
fi 