#!/bin/bash
# Build the 25->100 (x4) C=512 deployable resize overlay for the all-PL on-board run.
set -e
cd ~/OccFPGA/fpga/hls
rm -rf build2510 && mkdir build2510
cp resize_deploy.cpp run_deploy_resize.tcl build2510/
sed -E 's/(#define RZ_HIN[[:space:]]+)100/\125/; s/(#define RZ_WIN[[:space:]]+)100/\125/; s/(#define RZ_HOUT[[:space:]]+)200/\1100/; s/(#define RZ_WOUT[[:space:]]+)200/\1100/' resize.hpp > build2510/resize.hpp
echo "=== 25->100 sizing ==="
grep -E '#define RZ_(C|HIN|WIN|HOUT|WOUT)' build2510/resize.hpp
cd build2510
source ~/occfpga_hls_env.sh
echo "=== HLS synth+export (deployable resize 25->100) ==="
vitis-run --mode hls --tcl run_deploy_resize.tcl
echo "=== HLS IP zip ==="
ls -la resize_deploy_ip/sol/impl/ip/*.zip
mkdir -p vivado && cd vivado
rm -rf ip_repo && mkdir ip_repo
unzip -o ../resize_deploy_ip/sol/impl/ip/*.zip -d ip_repo >/dev/null
cp ~/OccFPGA/fpga/vivado_resize/build.tcl .
echo "=== Vivado overlay build ==="
vivado -mode batch -source build.tcl
echo "=== artifacts ==="
find rovl -name 'design_1_wrapper.bit' 2>/dev/null
find rovl -name 'design_1.hwh' 2>/dev/null
echo "BUILD2510_DONE"
