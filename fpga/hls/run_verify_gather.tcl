# Gather IP verification: full-size C-sim (bit-exact vs scalar golden) + small-size C/RTL cosim.
# Run: vitis-run --mode hls --tcl run_verify_gather.tcl
# --- full-size C-sim: proves the wide-word IP == scalar algorithm (GATHER_MISMATCHES must be 0) ---
open_project -reset gather_verify_full
set_top bev_gather
add_files bev_gather.cpp -cflags "-I."
add_files -tb bev_gather_tb.cpp -cflags "-I."
open_solution -reset s
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csim_design

# --- small-size C/RTL cosim: RTL == C and the EXACT measured cycle count (xsim) ---
open_project -reset gather_verify_small
set_top bev_gather
add_files bev_gather.cpp -cflags "-I. -DHLS_SMALL"
add_files -tb bev_gather_tb.cpp -cflags "-I. -DHLS_SMALL"
open_solution -reset s
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design
cosim_design
exit
