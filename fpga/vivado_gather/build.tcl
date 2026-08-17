# PS + deployable-gather overlay for the on-board run (Vivado 2025.2, xck26).
# Models fpga/vivado_resize/build.tcl: PS master HPM0_FPD -> s_axi_control; the IP's m_axi masters
# (6 of them: gmem0..5 = feat/depth/bev + the three streamed index-table bundles) -> SmartConnect -> HP0.
create_project -force gvl ./gvl -part xck26-sfvc784-2LV-c
catch { set_property BOARD_PART xilinx.com:kr260_som:part0:1.1 [current_project] }
set_property ip_repo_paths [list /home/ANON/OccFPGA/fpga/hls/gather_deploy_ip/sol/impl/ip] [current_project]
update_ip_catalog -rebuild
create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 0} [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP2 {0} CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128}] [get_bd_cells zynq_ultra_ps_e_0]
catch { set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} [get_bd_cells zynq_ultra_ps_e_0] }
create_bd_cell -type ip -vlnv xilinx.com:hls:bev_gather:1.0 gather_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* psr0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* smc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells smc_ctrl]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* smc_data
set_property -dict [list CONFIG.NUM_SI {6} CONFIG.NUM_MI {1}] [get_bd_cells smc_data]
set clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net $clk [get_bd_pins psr0/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins psr0/ext_reset_in]
set rstn [get_bd_pins psr0/peripheral_aresetn]
foreach p {zynq_ultra_ps_e_0/maxihpm0_fpd_aclk zynq_ultra_ps_e_0/saxihp0_fpd_aclk smc_ctrl/aclk smc_data/aclk gather_0/ap_clk} { connect_bd_net $clk [get_bd_pins $p] }
foreach p {smc_ctrl/aresetn smc_data/aresetn gather_0/ap_rst_n} { connect_bd_net $rstn [get_bd_pins $p] }
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins gather_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem0] [get_bd_intf_pins smc_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem1] [get_bd_intf_pins smc_data/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem2] [get_bd_intf_pins smc_data/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem3] [get_bd_intf_pins smc_data/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem4] [get_bd_intf_pins smc_data/S04_AXI]
connect_bd_intf_net [get_bd_intf_pins gather_0/m_axi_gmem5] [get_bd_intf_pins smc_data/S05_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_data/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
assign_bd_address
validate_bd_design
save_bd_design
puts "GVL_BD_OK"
set bdf [get_files design_1.bd]
make_wrapper -files $bdf -top -import
set_property top design_1_wrapper [current_fileset]
generate_target all $bdf
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "IMPL: [get_property STATUS [get_runs impl_1]]"
write_hw_platform -fixed -include_bit -force gather_ovl.xsa
puts "GVL_BITSTREAM_DONE"
