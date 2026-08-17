open_project -reset gather_deploy_ip
set_top bev_gather
add_files bev_gather_deploy.cpp -cflags "-I."
open_solution -reset sol
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5 -name default
csynth_design
export_design -format ip_catalog
exit
