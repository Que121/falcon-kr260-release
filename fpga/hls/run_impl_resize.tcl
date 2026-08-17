# Out-of-context place&route of the resize IP on the real K26 -> post-route Fmax + utilization.
# Run: vitis-run --mode hls --tcl run_impl_resize.tcl   (slow: runs Vivado synth + impl)
open_project -reset resize_impl
set_top resize_bilinear
add_files resize.cpp -cflags "-I."
open_solution -reset s
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design
export_design -flow impl -rtl verilog
exit
