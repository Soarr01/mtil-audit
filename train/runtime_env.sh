# Sourced by the GPU stream scripts. The container's root overlay (/, incl. /tmp and /root) is 100% full on host 3
# (shared with other containers), so every scratch/cache path our jobs write to is redirected to RAM (/dev/shm).
# Warm copies of /root/.cache/torch (DINOv2 hub repo+weights) were made into /dev/shm/mtil_env. /dev/shm is mounted
# noexec, so Triton's compiled .so cache lives on the NAS instead (warm copy of /root/.triton/cache).
E=/dev/shm/mtil_env
mkdir -p $E/tmp $E/mpl $E/xdg $E/torch
TC=/data/nas07_new/PersonalData/phuocthien/ssm_vla/.triton_cache_host3; mkdir -p $TC
export TMPDIR=$E/tmp TEMP=$E/tmp TMP=$E/tmp MPLCONFIGDIR=$E/mpl XDG_CACHE_HOME=$E/xdg TORCH_HOME=$E/torch TRITON_CACHE_DIR=$TC
