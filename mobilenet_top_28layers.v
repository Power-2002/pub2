`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:16:55
// Design Name: 
// Module Name: mobilenet_top_28layers
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module mobilenet_top_28layers #(
  parameter integer START_LAYER_ID = 6'd0,  
  parameter integer MAX_LAYER_ID   = 6'd28   
)(
  input  wire CLK,
  input  wire RESETn,
  input  wire start,
  output reg  done,
  output wire [2:0] fsm_state,
  output wire [5:0] current_layer,
  
  output wire        fc_out_valid,
  output wire [10:0] fc_out_class_idx,
  output wire signed [7:0] fc_out_logit
);

  localparam integer ADDR_W       = 16;

  localparam S_IDLE = 3'd0;
  localparam S_RUN  = 3'd1;
  localparam S_NEXT = 3'd2;
  localparam S_DONE = 3'd3;

  reg [2:0] state, state_n;
  reg [5:0] cur_layer;
  reg       layer_start;
  wire      layer_done;
  
  wire feat_bank_wr_sel;
wire feat_bank_rd_sel;


wire        le_fc_out_valid;
wire [10:0] le_fc_out_class_idx;
wire signed [7:0] le_fc_out_logit;


  assign fsm_state     = state;
assign current_layer = (cur_layer > MAX_LAYER_ID) ? MAX_LAYER_ID : cur_layer;


  always @(*) begin
    state_n     = state;
    layer_start = 1'b0;
    done        = 1'b0;

    case (state)
      S_IDLE: begin
        if (start) begin
          state_n     = S_RUN;
          layer_start = 1'b1;
        end
      end

      S_RUN: begin
        if (layer_done)
          state_n = S_NEXT;
      end

      S_NEXT: begin
  if (cur_layer > MAX_LAYER_ID) begin
    state_n = S_DONE;
  end else begin
    state_n     = S_RUN;
    layer_start = 1'b1;
  end
end


      S_DONE: begin
        done = 1'b1;
        if (! start)
          state_n = S_IDLE;
      end

      default: state_n = S_IDLE;
    endcase
  end

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      state     <= S_IDLE;
      cur_layer <= START_LAYER_ID[5:0];   
    end else begin
      state <= state_n;

      if (state == S_IDLE && start) begin
        cur_layer <= START_LAYER_ID[5:0];  
      end else if (state == S_RUN && layer_done) begin
        //if (cur_layer != MAX_LAYER_ID)
          cur_layer <= cur_layer + 6'd1;
      end
    end
  end

  // ===== layer_config_rom  =====
  wire [19:0] w_base_cur;
  wire [11:0] b_base_cur;
  wire [2:0]  layer_type_cur;  
  wire [10:0] cin_cur, cout_cur;
  wire [7:0]  img_w_cur, img_h_cur;
  wire [1:0]  stride_cur;

wire [5:0] rom_layer_addr = (cur_layer > MAX_LAYER_ID) ? MAX_LAYER_ID : cur_layer;



  layer_config_rom u_layer_cfg (
    .id        (rom_layer_addr),
    .w_base    (w_base_cur),
    .b_base    (b_base_cur),
    .layer_type(layer_type_cur),
    .cin       (cin_cur),
    .cout      (cout_cur),
    .img_w     (img_w_cur),
    .img_h     (img_h_cur),
    .stride    (stride_cur)
  );

  // ===== quant LUT =====
  wire signed [31:0] quant_M;
  wire        [5:0]  quant_s;
  wire signed [7:0]  quant_zp;

  quant_params_lut u_quant_lut (
    . layer_sel   (cur_layer[4:0]),
    .mult_scalar (quant_M),
    .shift_scalar(quant_s),
    .zp_out      (quant_zp)
  );

// ===== unified weights + arbiter =====

localparam integer WBUF_ADDR_W = 16;
localparam integer WBUF_DEPTH  = (1<<WBUF_ADDR_W);

wire                     bmg_ena;
wire [WBUF_ADDR_W-1:0]    bmg_addra;    
wire [127:0]              bmg_douta;

// DMA preload 输出：写 enable + 写地址 + 写数据
wire                     dma_wr_en;
wire [WBUF_ADDR_W-1:0]   dma_wr_addr;   
wire [127:0]             dma_wr_data;

blk_mem_gen_unified #(
  .BUF_ADDR_W (WBUF_ADDR_W),
  .WIDTH      (128),
  .DEPTH      (WBUF_DEPTH)
) u_unified_wmem (
  .clka        (CLK),

  .ena         (bmg_ena),
  .addra       (bmg_addra),
  .douta       (bmg_douta),

  .dma_wr_en   (dma_wr_en),
  .dma_wr_addr (dma_wr_addr),
  .dma_wr_data (dma_wr_data)
);

  wire        dw_weight_req,   dw_weight_grant,  dw_weight_valid,  dw_weight_done;
  wire [ADDR_W-1:0] dw_weight_base;
  wire [16:0]       dw_weight_count;
  wire [127:0]      dw_weight_data;

  wire        pw_weight_req,   pw_weight_grant,  pw_weight_valid,  pw_weight_done;
  wire [ADDR_W-1:0] pw_weight_base;
  wire [16:0]       pw_weight_count;
  wire [127:0]      pw_weight_data;

wire        ldr_start;
wire [ADDR_W-1:0] ldr_base;
wire [16:0] ldr_count;
wire        ldr_valid;
wire [127:0] ldr_data;
wire        ldr_done;

// ===== arbiter <-> preload ctrl =====
wire              preload_req;
wire [ADDR_W-1:0] preload_base;
wire [16:0]       preload_count;
wire              preload_done;

// ===== arbiter <-> weight buffer (read) =====
wire              bmg_en;
wire [WBUF_ADDR_W-1:0] bmg_addr;  
wire [127:0]       bmg_data;

// BRAM read port bridge: arbiter -> BRAM
assign bmg_ena   = bmg_en;
assign bmg_addra = bmg_addr;
assign bmg_data  = bmg_douta;

weight_loader_arbiter #(
  .ADDR_W(ADDR_W),
  .BUF_ADDR_W(WBUF_ADDR_W)   
) u_weight_arbiter (
  .clk   (CLK),              
  .rst_n (RESETn),            

  .dw_req   (dw_weight_req),
  .dw_base  (dw_weight_base),
  .dw_count (dw_weight_count),
  .dw_grant (dw_weight_grant),
  .dw_valid (dw_weight_valid),
  .dw_data  (dw_weight_data),
  .dw_done  (dw_weight_done),


  .pw_req   (pw_weight_req),
  .pw_base  (pw_weight_base),
  .pw_count (pw_weight_count),
  .pw_grant (pw_weight_grant),
  .pw_valid (pw_weight_valid),
  .pw_data  (pw_weight_data),
  .pw_done  (pw_weight_done),

  // preload handshake
  .preload_req   (preload_req),
  .preload_base  (preload_base),
  .preload_count (preload_count),
  .preload_done  (preload_done),

  // weight buffer read port
  .bmg_en   (bmg_en),
  .bmg_addr (bmg_addr),
  .bmg_data (bmg_data)
);

// 仿真用的 DMA preload 控制器
dma_preload_ctrl_sim #(
    .ADDR_W(ADDR_W),
    .DATA_W(128),
    .BUF_ADDR_W(WBUF_ADDR_W),
    .MEM_FILE("D:/NoC/mycode/mobilenet_acc3/data/weights/all_weights_128b.mem")  // .mem 文件路径
) u_dma_preload_ctrl_sim (
    .clk              (CLK),
    .rst_n            (RESETn),
    .preload_req      (preload_req),   // 来自 arbiter
    .preload_base     (preload_base),  // 来自 arbiter
    .preload_count    (preload_count), // 来自 arbiter
    .preload_done     (preload_done),  // 给 arbiter

    // 给 weight_loader_universal 的写口
    .dma_wr_en        (dma_wr_en),
    .dma_wr_addr      (dma_wr_addr),
    .dma_wr_data      (dma_wr_data)
);


  // ===== unified feature buffer =====
  wire        feat_wr_en;
  wire [16:0] feat_wr_addr;
  wire [127:0]feat_wr_data;
  wire        feat_rd_en;
  wire [16:0] feat_rd_addr;
  wire [127:0]feat_rd_data;
  wire        feat_rd_valid;

  unified_feature_buffer u_feature_buf (
    .clk      (CLK),
    .rst_n    (RESETn),
    // === 新增：bank 选择 ===
  .bank_wr_sel (feat_bank_wr_sel),
  .bank_rd_sel (feat_bank_rd_sel),
    .wr_en    (feat_wr_en),
    .wr_addr  (feat_wr_addr),
    .wr_data  (feat_wr_data),
    .rd_en    (feat_rd_en),
    .rd_addr  (feat_rd_addr),
    .rd_data  (feat_rd_data),
    .rd_valid (feat_rd_valid)
  );

  wire [15:0] le_wr_local_addr;
  wire [15:0] le_rd_local_addr;

// === BYPASS address_manager: unified address space ===
assign feat_wr_addr = le_wr_local_addr;
assign feat_rd_addr = le_rd_local_addr;


  // ===== bias ROM =====
  wire [511:0] bias_vec;
  wire         bias_valid;
  wire [6:0]   bias_block_idx;
  wire         bias_rd_en;

  bias_rom_unified #(
    . LANES     (16),
    .TOTAL_BIAS(11945)
  ) u_bias_rom (
    .clk         (CLK),
    .rst_n       (RESETn),
    .base_addr_in(b_base_cur),
    .block_idx   (bias_block_idx),
    .rd_en       (bias_rd_en),
    .bias_out    (bias_vec),
    .bias_valid  (bias_valid)
  );


  // ===== layer_exec  =====
  layer_exec #(
    . ADDR_W(ADDR_W),
    .FAST_SIM_EN(1),
    .FAST_PW_COUT_SUBSAMPLE(16),
    .FAST_PW_PX_SUBSAMPLE(16),
    .PW_WRITE_LIMIT_EN(1),
    .PW_WRITE_LIMIT_WORDS(1024)
  ) u_layer_exec (
    .CLK        (CLK),
    .RESETn     (RESETn),
    .start      (layer_start),
    .done       (layer_done),
    .layer_id   (cur_layer),
    .layer_type (layer_type_cur),  // 3λ
    .cin        (cin_cur),
    .cout       (cout_cur),
    .img_w      (img_w_cur),
    .img_h      (img_h_cur),
    .stride     (stride_cur),

    .w_base     (w_base_cur[ADDR_W-1:0]),
    .b_base     (b_base_cur),

    .quant_M    (quant_M),
    .quant_s    (quant_s),
    .quant_zp   (quant_zp),
    .dw_req         (dw_weight_req),
    .dw_base        (dw_weight_base),
    .dw_count       (dw_weight_count),
    .dw_grant       (dw_weight_grant),
    .dw_valid       (dw_weight_valid),
    .dw_data        (dw_weight_data),
    .dw_done        (dw_weight_done),
    .pw_req         (pw_weight_req),
    .pw_base        (pw_weight_base),
    .pw_count       (pw_weight_count),
    .pw_grant       (pw_weight_grant),
    .pw_valid       (pw_weight_valid),
    .pw_data        (pw_weight_data),
    .pw_done        (pw_weight_done),

    .bias_vec       (bias_vec),
    .bias_valid     (bias_valid),
    .bias_block_idx (bias_block_idx),
    .bias_rd_en     (bias_rd_en),

    .feat_wr_en         (feat_wr_en),
    .feat_wr_local_addr (le_wr_local_addr),
    .feat_wr_data       (feat_wr_data),
    .feat_rd_en         (feat_rd_en),
    .feat_rd_local_addr (le_rd_local_addr),
    .feat_rd_data       (feat_rd_data),
    .feat_rd_valid      (feat_rd_valid),
    
    .fc_out_valid     (le_fc_out_valid),
.fc_out_class_idx (le_fc_out_class_idx),
.fc_out_logit     (le_fc_out_logit),
    .feat_bank_wr_sel (feat_bank_wr_sel),
    .feat_bank_rd_sel (feat_bank_rd_sel)
  );

assign fc_out_valid     = le_fc_out_valid;
assign fc_out_class_idx = le_fc_out_class_idx;
assign fc_out_logit     = le_fc_out_logit;

endmodule