#!/bin/bash
# install_miniforge_bluepebble.sh
# Downloads and installs Miniforge (conda + mamba) on BluePebble HPC,
# then moves the conda initialization block out of ~/.bashrc into
# ~/initMamba.sh so it can be sourced on demand or added to job scripts.
set -euo pipefail

INSTALLER="Miniforge3-$(uname)-$(uname -m).sh"
INSTALL_DIR="${WORK}/miniforge3"
BASHRC="${HOME}/.bashrc"
INIT_MAMBA="${HOME}/initMamba.sh"

echo ">>> Downloading ${INSTALLER}..."
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"

echo ">>> Making installer executable..."
chmod u+x "${INSTALLER}"

echo ">>> Installing into ${INSTALL_DIR} (batch mode, no prompts)..."
./"${INSTALLER}" -b -p "${INSTALL_DIR}"

echo ">>> Cleaning up installer script..."
rm -f "${INSTALLER}"

echo ">>> Initializing conda for bash (writes block into ~/.bashrc)..."
"${INSTALL_DIR}/bin/conda" init bash

echo ">>> Disabling base env auto-activation on login..."
"${INSTALL_DIR}/bin/conda" config --set auto_activate_base false

# --- Move the conda initialize block from ~/.bashrc into ~/initMamba.sh ---
echo ">>> Backing up ~/.bashrc..."
cp "${BASHRC}" "${BASHRC}.backup"

START_MARK=">>> conda initialize >>>"
END_MARK="<<< conda initialize <<<"

if ! grep -qF "${START_MARK}" "${BASHRC}"; then
    echo "!!! Could not find conda initialize block in ${BASHRC}. Skipping move."
else
    echo ">>> Extracting conda initialize block into ${INIT_MAMBA}..."
    sed -n "/${START_MARK}/,/${END_MARK}/p" "${BASHRC}" > "${INIT_MAMBA}"

    echo ">>> Removing conda initialize block from ${BASHRC}..."
    sed -i "/${START_MARK}/,/${END_MARK}/d" "${BASHRC}"

    echo ">>> Done."
    echo "Backup of original .bashrc saved at: ${BASHRC}.backup"
    echo "Conda init block moved to: ${INIT_MAMBA}"
    echo ""
    echo "To activate conda in a shell or submission script, run:"
    echo "    . ~/initMamba.sh"
    echo "To deactivate:"
    echo "    mamba deactivate"
fi