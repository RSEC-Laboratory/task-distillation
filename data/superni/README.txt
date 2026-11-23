SUPER-NATURALINSTRUCTIONS SOURCE DATA

sources.json records the upstream commit, source-file hashes, and original
instance licenses for the reserved and development tasks used in this project.
The metadata license in LICENSE.txt does not relicense the underlying instances.

From the repository root, fetch source tasks for optional revalidation/live runs:

    python scripts/fetch_data.py

The downloader verifies pinned raw bytes and restores the recorded newline
convention. Full source tasks are placed in tasks/, which is ignored by Git.
Saved-output evaluation works without this download. Source-instance validation
coverage is reported explicitly in the reserved NLP result JSON.
