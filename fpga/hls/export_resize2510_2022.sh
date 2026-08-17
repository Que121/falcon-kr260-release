#!/bin/bash
# Re-export the 25->100 (x4) resize IP in Vitis HLS 2022.2 with a DISTINCT VLNV (resize2510),
# so it can coexist with the 100->200 resize_bilinear in the DPU bitstream.
set -e
cd ~/OccFPGA/fpga/hls
rm -rf rz2510_22 && mkdir rz2510_22
cp resize.cpp rz2510_22/
sed -E 's/(#define RZ_HIN[[:space:]]+)100/\125/; s/(#define RZ_WIN[[:space:]]+)100/\125/; s/(#define RZ_HOUT[[:space:]]+)200/\1100/; s/(#define RZ_WOUT[[:space:]]+)200/\1100/' resize.hpp > rz2510_22/resize.hpp
cat > rz2510_22/run.tcl <<'TCL'
open_project -reset resize2510_ip22
set_top resize_bilinear
add_files resize.cpp -cflags "-I."
open_solution -reset sol
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
config_export -format ip_catalog -ipname resize2510 -vendor xilinx.com -library hls -version 1.0
csynth_design
export_design -format ip_catalog
exit
TCL
cd rz2510_22
echo "=== sizing ==="; grep -E '#define RZ_(HIN|WIN|HOUT|WOUT)' resize.hpp | grep -vE '[[:space:]](4|8)$'
source ~/occfpga_viv22_env.sh
vitis_hls -f run.tcl
echo "=== exported IP (VLNV) ==="
find resize2510_ip22 -name 'component.xml' | head -1 | xargs grep -m1 -oE '<spirit:name>resize[^<]*' 2>/dev/null
ls resize2510_ip22/sol/impl/ip/*.zip
echo "EXPORT2510_DONE"
