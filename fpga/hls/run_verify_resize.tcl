# Resize IP verification: full-size C-sim (bit-exact vs scalar golden) + small-size C/RTL cosim.
# Run: vitis-run --mode hls --tcl run_verify_resize.tcl
# --- full-size C-sim: proves the wide-word IP == scalar algorithm (RESIZE_MISMATCHES must be 0) ---
open_project -reset resize_verify_full
set_top resize_bilinear
add_files resize.cpp -cflags "-I."
add_files -tb resize_tb.cpp -cflags "-I."
open_solution -reset s
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csim_design

# --- small-size C/RTL cosim: RTL == C and the EXACT measured cycle count (xsim) ---
open_project -reset resize_verify_small
set_top resize_bilinear
add_files resize.cpp -cflags "-I. -DHLS_SMALL"
add_files -tb resize_tb.cpp -cflags "-I. -DHLS_SMALL"
open_solution -reset s
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design
cosim_design
exit
