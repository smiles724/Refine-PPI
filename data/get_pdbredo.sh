mkdir -p ./PDB_REDO
rsync -av \
    --include='*_final.cif' \
    --include='*/' \
    --exclude='*' \
    rsync://rsync.pdb-redo.eu/pdb-redo ./PDB_REDO