# Vitis HLS batch C-synthesis of the view-transform gather IP, real K26/xczu5ev part.
# Run headless:  vitis_hls -f run_gather_hls.tcl
# K26 SOM part (KR260) = xck26-sfvc784-2LV-c  (underlying device xczu5ev).
open_project -reset gather_hls
set_top bev_gather
add_files bev_gather.cpp -cflags "-I."

# --- 200 MHz target ---
open_solution -reset sol200_xck26
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design

# --- 300 MHz target (for the @300MHz column) ---
open_solution -reset sol300_xck26
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default
csynth_design

exit
