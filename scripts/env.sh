# source this: points geant4_pybind at the conda-forge Geant4 datasets (no download). bash/zsh safe.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
while read -r _name _var _dir; do
  [ -d "$_dir" ] || _dir="${_dir%/*}/${_dir##*/G4}"   # geant4-config prints G4ENSDFSTATE3.0, dir is ENSDFSTATE3.0
  export "$_var=$_dir"
done < <("$ROOT/.conda/bin/geant4-config" --datasets)
