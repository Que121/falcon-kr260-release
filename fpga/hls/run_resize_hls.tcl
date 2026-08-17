# Vitis HLS batch C-synthesis of the bilinear-resize IP, real K26/xczu5ev part.
# Run headless:  vitis_hls -f run_resize_hls.tcl
# NOTE: resize.hpp is compiled with its default UP2 sizing (C=512, 100x100 -> 200x200, CTILE=64).
# If csynth reports BRAM > 100% (the tmp[CTILE][HOUT][WIN] array is large), reduce RZ_CTILE or
# switch to the row-fused variant (tmp shrinks to one output row) — see SYNTH-RESULTS once it runs.
open_project -reset resize_hls
set_top resize_bilinear
add_files resize.cpp -cflags "-I."

# --- 200 MHz target ---
open_solution -reset sol200_xck26
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design

# --- 300 MHz target ---
open_solution -reset sol300_xck26
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default
csynth_design

exit
