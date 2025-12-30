`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:29:20
// Design Name: 
// Module Name: layer_exec
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
// Description: Per-layer execution unit for MobileNet top-28layers.
module layer_exec #(
  parameter integer ADDR_W = 16,
  parameter integer FAST_SIM_EN            = 1,
  parameter integer FAST_PW_COUT_SUBSAMPLE = 16,   // compute 1/16 cout tiles
  parameter integer FAST_PW_PX_SUBSAMPLE   = 16,   // compute 1/16 pixels
 parameter integer PW_WRITE_LIMIT_EN      = 1,
  parameter integer PW_WRITE_LIMIT_WORDS   = 1024,
  parameter integer USE_PW32x16 = 1   
)(
  input  wire CLK,
  input  wire RESETn,
  input  wire start,
  output reg  done,

  // ---- Layer configuration from layer_config_rom ----
  input  wire [5:0] layer_id,
  input  wire [2:0] layer_type,      //  0=CONV, 1=DW, 2=PW, 3=AP, 4=FC
  input  wire [10:0] cin,
  input  wire [10:0] cout,
  input  wire [7:0]  img_w,
  input  wire [7:0]  img_h,
  input  wire [1:0]  stride,

  // Weight / bias base
  input  wire [ADDR_W-1:0] w_base,
  input  wire [11:0]       b_base,

  // Quantization parameters
  input  wire signed [15:0] quant_M,
  input  wire        [5:0]  quant_s,
  input  wire signed [7:0]  quant_zp,

  // ---- Weight loader arbiter interface ----
  // DW channel
  output wire               dw_req,
  output wire [ADDR_W-1:0]  dw_base,
  output wire [16:0]        dw_count,
  input  wire               dw_grant,
  input  wire               dw_valid,
  input  wire [127:0]       dw_data,
  input  wire               dw_done,

  // PW / CONV channel
  output wire               pw_req,
  output wire [ADDR_W-1:0]  pw_base,
  output wire [16:0]        pw_count,
  input  wire               pw_grant,
  input  wire               pw_valid,
  input  wire [127:0]       pw_data,
  input  wire               pw_done,

  // ---- Bias ROM interface ----
  input  wire [511:0]       bias_vec,
  input  wire               bias_valid,
  output reg  [6:0]         bias_block_idx,
  output reg                bias_rd_en,

  // ---- Feature buffer interface (local addr) ----
  // write side
  output wire               feat_wr_en,
  output wire [15:0]        feat_wr_local_addr,
  output wire [127:0]       feat_wr_data,
  // read side
  output wire               feat_rd_en,
  output wire [15:0]        feat_rd_local_addr,
  input  wire [127:0]       feat_rd_data,
  input  wire               feat_rd_valid,
  // ---- FC output (real dataflow) ----
  output wire               fc_out_valid,
  output wire [10:0]        fc_out_class_idx,
  output wire signed [7:0]  fc_out_logit,
  
  output reg feat_bank_wr_sel,
  output reg feat_bank_rd_sel

  
);

  // ============================================================
  // PE в - ??? 3216
  // ============================================================
  localparam integer PE_ROWS = 32;
  localparam integer PE_COLS = 32;

// ============================================================
  // Layer type definitions
  // ============================================================
  localparam [2:0] TYPE_CONV = 3'd0;
  localparam [2:0] TYPE_DW   = 3'd1;
  localparam [2:0] TYPE_PW   = 3'd2;
  localparam [2:0] TYPE_AP   = 3'd3;
  localparam [2:0] TYPE_FC   = 3'd4;


  // DW generic
  // ============================================================
  localparam integer DW_UNIT = 16;

  wire is_dw_layer = (layer_type == TYPE_DW);
  wire is_ap_layer = (layer_type == TYPE_AP);
  wire is_fc_layer = (layer_type == TYPE_FC);

  wire [6:0] dw_num_blocks = (cin + DW_UNIT - 1) / DW_UNIT;
  reg  [6:0] dw_block_idx;
  wire dw_last_block = (dw_block_idx == dw_num_blocks - 1);
  wire dw_stride2 = (stride == 2'd2);
  reg  dw_block_started;
  wire dw_block_start;

// ============================================================
  // 1. Local FSM:  ?? AP  FC
  // ============================================================
  localparam LE_IDLE    = 4'd0;
  localparam LE_RUN_L0  = 4'd1;   // L0: 3x3 CONV
  localparam LE_RUN_L1D = 4'd2;   // L1: 3x3 DW
  localparam LE_RUN_L2P = 4'd3;   // L2: 1x1 PW
  localparam LE_RUN_AP  = 4'd4;   // Layer 27: Global Average Pooling
  localparam LE_RUN_FC  = 4'd5;   // Layer 28: Fully Connected
  localparam LE_BYPASS  = 4'd6;

  reg [3:0] le_state, le_state_n;

  // Per-layer start/done
  reg  l0_start;  wire l0_done;
  reg  l1_start;  wire l1_done;
  reg  l2_start;  wire l2_done;
  reg  ap_start;  wire ap_done;
  reg  fc_start;  wire fc_done;


  wire [6:0] fc_bias_block_idx;


// main FSM register
always @(posedge CLK or negedge RESETn) begin
  if (!RESETn) begin
    le_state <= LE_IDLE;

    // bank init
    feat_bank_wr_sel <= 1'b0;
    feat_bank_rd_sel <= 1'b1;

  end else begin
    le_state <= le_state_n;

    // swap bank at layer boundary
    if (
      (le_state == LE_RUN_L0  && l0_done)  ||
      (le_state == LE_RUN_L1D && l1_done)  ||
      (le_state == LE_RUN_L2P && l2_done)  ||
      (le_state == LE_RUN_FC  && fc_done)
    ) begin
      feat_bank_wr_sel <= ~feat_bank_wr_sel;
      feat_bank_rd_sel <= ~feat_bank_rd_sel;
    end
  end
end

  // FSM next-state
  always @(*) begin
    le_state_n = le_state;
    done       = 1'b0;
    l0_start   = 1'b0;
    l1_start   = 1'b0;
    l2_start   = 1'b0;
    ap_start   = 1'b0;
    fc_start   = 1'b0;

    bias_rd_en     = 1'b0;
    bias_block_idx = dw_block_idx;

    case (le_state)
      LE_IDLE:  begin
        if (start) begin
          case (layer_type)
            TYPE_CONV: begin  // Layer 0
              if (layer_id == 6'd0) begin
                le_state_n = LE_RUN_L0;
                l0_start   = 1'b1;
              end else begin
                le_state_n = LE_BYPASS;
              end
            end
            TYPE_DW: begin    // DW layers
              le_state_n = LE_RUN_L1D;
            end
            TYPE_PW: begin    // PW layers
              le_state_n = LE_RUN_L2P;
              l2_start   = 1'b1;
            end
            TYPE_AP: begin    // Layer 27: Global Average Pooling
              le_state_n = LE_RUN_AP;
              ap_start   = 1'b1;
            end
            TYPE_FC: begin    // Layer 28: Fully Connected
              le_state_n = LE_RUN_FC;
              fc_start   = 1'b1;
            end
            default: begin
              le_state_n = LE_BYPASS;
            end
          endcase
        end
      end

      LE_RUN_L0: begin
        if (l0_done) begin
          le_state_n = LE_IDLE;
          done       = 1'b1;
        end
      end

      LE_RUN_L1D:  begin
        bias_rd_en     = 1'b1;
        bias_block_idx = dw_block_idx;
        if (l1_done) begin
          le_state_n = LE_IDLE;
          done       = 1'b1;
        end
      end

      LE_RUN_L2P: begin
        if (l2_done) begin
          le_state_n = LE_IDLE;
          done       = 1'b1;
        end
      end

      LE_RUN_AP:  begin
        if (ap_done) begin
          le_state_n = LE_IDLE;
          done       = 1'b1;
        end
      end

      LE_RUN_FC: begin
        // FC??bias
        bias_rd_en     = 1'b1;
        bias_block_idx = fc_bias_block_idx;
        if (fc_done) begin
          le_state_n = LE_IDLE;
          done       = 1'b1;
        end
      end

      LE_BYPASS: begin
        done       = 1'b1;
        le_state_n = LE_IDLE;
      end

      default: le_state_n = LE_IDLE;
    endcase
  end

  // ============================================================
  // 2. Shared PE array (for L0/L2, FC can also reuse)
  // ============================================================
wire [PE_ROWS*8-1:0] arr_active_left;   // 256 bits
  wire [PE_COLS*8-1:0] arr_weight_above;  // 128 bits
  wire                 arr_w_en;
  wire [PE_COLS*32-1:0] arr_out_sum;      
  // For legacy 32x16 schedulers (L0) keep a 16-col view
  wire [511:0] arr_out_sum_16 = arr_out_sum[511:0];


(* keep_hierarchy = "yes", dont_touch = "true" *)
PE_array_single_weight #(
    .NUM_ROWS(PE_ROWS),
    .NUM_COLS(PE_COLS)
  ) u_shared_array (
    .CLK(CLK),
    . RESET(RESETn),
    .EN(1'b1),
    .W_EN(arr_w_en),
    .active_left(arr_active_left),
    .in_weight_above(arr_weight_above),
    .out_weight_final(),
    .out_sum_final(arr_out_sum)
  );

// L0 (conv1) is still 32x16
wire [PE_ROWS*8-1:0] l0_act;
wire                 l0_wen;
wire [127:0]          l0_w16;  

// L2 (PW) - 


// L2 (PW) - keep scheduler as 32x16
wire [PE_ROWS*8-1:0] l2_act;
wire                 l2_wen;
wire [127:0]          l2_w16;     // ? 改回 128b
wire [255:0]          l2_w32;     // ? 外部扩成 256b

assign l2_w32 = {128'd0, l2_w16}; // 或者用复制 {l2_w16, l2_w16}


// ----------------------------
// L0 scheduler is 32x16 => provides 16-col weights
// L2 scheduler is 32x32 => provides 32-col weights (if you set NUM_COLS=32)
// We unify array inputs to 32-col here.
// ----------------------------
wire [255:0] l0_w32 = {128'd0, l0_w16}; 


assign arr_active_left  =
  (le_state == LE_RUN_L0)  ? l0_act :
  (le_state == LE_RUN_L2P) ? l2_act :
  {(PE_ROWS*8){1'b0}};

assign arr_weight_above =
  (le_state == LE_RUN_L0)  ? l0_w32 :
  (le_state == LE_RUN_L2P) ? l2_w32 :
  {(PE_COLS*8){1'b0}};


assign arr_w_en =
  (le_state == LE_RUN_L0)  ? l0_wen :
  (le_state == LE_RUN_L2P) ? l2_wen :
  1'b0;


  // ============================================================
  // 3. Shared requantize16
  // ============================================================
  wire        l0_y_valid;
  wire [511:0]l0_y_data;
  wire        l2_y_valid;
  
wire [511:0] l2_y_data;   



  wire        q_en;
  wire [511:0]q_in;
  wire [127:0]q_out_data;
  wire        q_out_valid;

  assign q_en = (le_state == LE_RUN_L0)  ? l0_y_valid : 
                (le_state == LE_RUN_L2P) ? l2_y_valid :  1'b0;

  assign q_in = (le_state == LE_RUN_L0)  ? l0_y_data  :
                (le_state == LE_RUN_L2P) ? l2_y_data  : 512'd0;

  requantize16 u_shared_quant (
    . CLK             (CLK),
    .RESET           (RESETn),
    .en              (q_en),
    .in_acc          (q_in),
    .bias_in         (bias_vec),
    .cfg_mult_scalar (quant_M),
    .cfg_shift_scalar(quant_s),
    .cfg_symmetric   (1'b0),
    .cfg_zp_out      (quant_zp),
    .out_q           (q_out_data),
    .out_valid       (q_out_valid)
  );

  // ============================================================
  // 4. Feature buffer read/write mux (?APFC)
  // ============================================================
  // L1 read side
  wire       l1_feat_rd_en;
  wire [15:0]l1_feat_rd_addr;
  // L2 read side
  wire       l2_feat_rd_en;
  wire [15:0]l2_feat_rd_addr;
  // AP read side
  wire       ap_feat_rd_en;
  wire [15:0]ap_feat_rd_addr;
  // FC read side
  wire       fc_feat_rd_en;
  wire [15:0]fc_feat_rd_addr;

  // L0 write side
  wire       l0_feat_wr_en;
  wire [15:0]l0_feat_wr_addr;
  wire [127:0]l0_feat_wr_data;
  // L1 write side
  wire       l1_feat_wr_en;
  wire [15:0]l1_feat_wr_addr;
  wire [127:0]l1_feat_wr_data;
  // L2 write side
  wire       l2_feat_wr_en;
  wire [15:0]l2_feat_wr_addr;
  wire [127:0]l2_feat_wr_data;
  // AP write side
  wire       ap_feat_wr_en;
  wire [15:0]ap_feat_wr_addr;
  wire [127:0]ap_feat_wr_data;
  // FC write side (logits)
  wire       fc_feat_wr_en;
  wire [15:0]fc_feat_wr_addr;
  wire [127:0]fc_feat_wr_data;

  // Global write mux
  assign feat_wr_en         = (le_state == LE_RUN_L0)  ? l0_feat_wr_en   :
                              (le_state == LE_RUN_L1D) ? l1_feat_wr_en   :
                              (le_state == LE_RUN_L2P) ? l2_feat_wr_en   :
                              (le_state == LE_RUN_AP)  ? ap_feat_wr_en   : 
                              (le_state == LE_RUN_FC)  ? fc_feat_wr_en   : 1'b0;

  assign feat_wr_local_addr = (le_state == LE_RUN_L0)  ? l0_feat_wr_addr : 
                              (le_state == LE_RUN_L1D) ? l1_feat_wr_addr : 
                              (le_state == LE_RUN_L2P) ? l2_feat_wr_addr :
                              (le_state == LE_RUN_AP)  ? ap_feat_wr_addr : 
                              (le_state == LE_RUN_FC)  ? fc_feat_wr_addr : 16'd0;

  assign feat_wr_data       = (le_state == LE_RUN_L0)  ? l0_feat_wr_data :
                              (le_state == LE_RUN_L1D) ? l1_feat_wr_data :
                              (le_state == LE_RUN_L2P) ? l2_feat_wr_data : 
                              (le_state == LE_RUN_AP)  ? ap_feat_wr_data :
                              (le_state == LE_RUN_FC)  ? fc_feat_wr_data : 128'd0;

  // Global read mux
  assign feat_rd_en         = (le_state == LE_RUN_L1D) ? l1_feat_rd_en   :
                              (le_state == LE_RUN_L2P) ? l2_feat_rd_en   :
                              (le_state == LE_RUN_AP)  ? ap_feat_rd_en   : 
                              (le_state == LE_RUN_FC)  ? fc_feat_rd_en   : 1'b0;

  assign feat_rd_local_addr = (le_state == LE_RUN_L1D) ? l1_feat_rd_addr :
                              (le_state == LE_RUN_L2P) ? l2_feat_rd_addr :
                              (le_state == LE_RUN_AP)  ? ap_feat_rd_addr :
                              (le_state == LE_RUN_FC)  ? fc_feat_rd_addr : 16'd0;

  // ============================================================
  // 5. PW channel mux (L0 / L2 / FC)
  // ============================================================
  wire        l0_pw_req, l2_pw_req, fc_pw_req;
  wire [ADDR_W-1:0] l0_pw_base, l2_pw_base, fc_pw_base;
  wire [10:0]       l0_pw_count, l2_pw_count, fc_pw_count;

  assign pw_req   = (le_state == LE_RUN_L0)  ? l0_pw_req   : 
                    (le_state == LE_RUN_L2P) ? l2_pw_req   :
                    (le_state == LE_RUN_FC)  ? fc_pw_req   : 1'b0;

  assign pw_base  = (le_state == LE_RUN_L0)  ? l0_pw_base  : 
                    (le_state == LE_RUN_L2P) ? l2_pw_base  :
                    (le_state == LE_RUN_FC)  ? fc_pw_base  : w_base;

  assign pw_count = (le_state == LE_RUN_L0)  ? l0_pw_count :
                    (le_state == LE_RUN_L2P) ? l2_pw_count : 
                    (le_state == LE_RUN_FC)  ? fc_pw_count : 11'd0;

  // ============================================================
  // 6. L0: 3x3 CONV path
  // ============================================================
  wire        l0_win_req, l0_win_valid;
  wire [215:0]l0_win_flat;

  window_fetcher_pull_3x3x3_str2_int8 u_l0_fetch (
    .CLK        (CLK),
    .RESET      (RESETn),
    .start_frame(l0_start),
    .win_req    (l0_win_req),
    .win_valid  (l0_win_valid),
    .win_flat   (l0_win_flat),
    .frame_done ()
  );  
  
    conv1_scheduler_32x16 u_l0_sched (
    . CLK(CLK),
    .RESET(RESETn),
    .start(le_state == LE_RUN_L0),
    .done(),
    .w_base_in(w_base[ADDR_W-1:0]),

    .win_req(l0_win_req),
    .win_valid(l0_win_valid),
    .win_flat(l0_win_flat),

    .weight_req(l0_pw_req),
    .weight_grant(pw_grant),
    .weight_base(l0_pw_base),
    .weight_count(l0_pw_count),
    .weight_valid(pw_valid),
    .weight_data(pw_data),
    .weight_done(pw_done),

    .arr_W_EN(l0_wen),
    .in_weight_above(l0_w16),
    .active_left(l0_act),
    .out_sum_final(arr_out_sum_16),

    .y_valid(l0_y_valid),
    .y_data(l0_y_data),
    .y_tile_sel()
  );
  

  // L0 write-back
  reg [15:0] l0_wr_ptr_reg;
  reg        l0_done_reg;

  assign l0_feat_wr_en   = (le_state == LE_RUN_L0) && q_out_valid;
  assign l0_feat_wr_addr = l0_wr_ptr_reg;
  assign l0_feat_wr_data = q_out_data;
  assign l0_done         = l0_done_reg;

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      l0_wr_ptr_reg <= 16'd0;
      l0_done_reg   <= 1'b0;
    end else if (l0_start) begin
      l0_wr_ptr_reg <= 16'd0;
      l0_done_reg   <= 1'b0;
    end else if (l0_feat_wr_en && !l0_done_reg) begin
      if (l0_wr_ptr_reg == 16'd1024)
        l0_done_reg <= 1'b1;
      else
        l0_wr_ptr_reg <= l0_wr_ptr_reg + 16'd1;
    end
  end

  // ============================================================
  // 7. L2: 1x1 PW path
  // ============================================================
  // L2 feature fetcher
  wire l2_done_sched;

  pw_scheduler_32x16_pipelined#(
  .NUM_ROWS(PE_ROWS),
  .NUM_COLS(16),
  .ADDR_W(ADDR_W),
  .FAST_SIM_EN(FAST_SIM_EN),
  .FAST_COUT_SUBSAMPLE(FAST_PW_COUT_SUBSAMPLE),
  .FAST_PX_SUBSAMPLE(FAST_PW_PX_SUBSAMPLE)
) u_l2_sched (
    .CLK(CLK),
    .RESET(RESETn),
    .start(l2_start),
    .done(l2_done_sched),
    .cin(cin),
    .cout(cout),
    .img_w(img_w),
    .img_h(img_h),
    .w_base_in(w_base),

    .weight_req(l2_pw_req),
    .weight_grant(pw_grant),
    .weight_base(l2_pw_base),
    .weight_count(l2_pw_count),
    .weight_valid(pw_valid),
    .weight_data(pw_data),
    .weight_done(pw_done),

    .feat_rd_en(l2_feat_rd_en),
    .feat_rd_addr(l2_feat_rd_addr),
    .feat_rd_data(feat_rd_data),
    .feat_rd_valid(feat_rd_valid),

    .arr_W_EN(l2_wen),
    .in_weight_above(l2_w16),
    .active_left(l2_act),
    .out_sum_final(arr_out_sum_16),

    .y_valid(l2_y_valid),
    .y_data(l2_y_data),
    .y_tile_sel()
  );


   // ============================================================
  // 7. PW write-back (shared for ALL PW layers)
  // ============================================================
  reg [15:0] l2_wr_ptr_reg;
  reg        l2_done_reg;  // PW writeback event: driven by REAL quantized output valid
  wire pw_wr_event;
  assign pw_wr_event = ((le_state == LE_RUN_L2P) && q_out_valid);

  // PW done threshold (debug limit)
  localparam integer PW_FALLBACK_TOTAL_WORDS = 50176; // ??done?ν??DEBUG_PW_LIMIT_EN
  wire [15:0] pw_last_word_idx;

    assign pw_last_word_idx =
      (PW_WRITE_LIMIT_EN != 0) ? (PW_WRITE_LIMIT_WORDS[15:0] - 16'd1)
                              : (PW_FALLBACK_TOTAL_WORDS[15:0] - 16'd1);

  assign l2_feat_wr_en   = pw_wr_event;
  assign l2_feat_wr_addr = l2_wr_ptr_reg;
    assign l2_feat_wr_data = q_out_data;
  assign l2_done = (FAST_SIM_EN != 0) ? l2_done_sched : l2_done_reg;


  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      l2_wr_ptr_reg <= 16'd0;
      l2_done_reg   <= 1'b0;
    end else if (l2_start) begin
      l2_wr_ptr_reg <= 16'd0;
      l2_done_reg   <= 1'b0;
    end else if (pw_wr_event && !l2_done_reg) begin
      if (l2_wr_ptr_reg == pw_last_word_idx)
        l2_done_reg <= 1'b1;
      else
        l2_wr_ptr_reg <= l2_wr_ptr_reg + 16'd1;
    end
  end

  // ============================================================
  // 8. L1: DW 3x3 path  
  // ============================================================
  // DW weight cache
  wire        l1_cache_load_start, l1_cache_load_done;


  reg l1_weight_loaded;

always @(posedge CLK or negedge RESETn) begin
  if (!RESETn)
    l1_weight_loaded <= 1'b0;
  else if (dw_block_start)
    l1_weight_loaded <= 1'b0;
  else if (l1_cache_load_done)
    l1_weight_loaded <= 1'b1;
end

  assign l1_cache_load_start = (le_state == LE_RUN_L1D) && !l1_weight_loaded;
  
wire [ADDR_W-1:0] l1_dw_w_base =  w_base + (dw_block_idx * 11'd9); // ? block 9 weights
  wire [ADDR_W-1:0] dw_w_base_blk = w_base + (dw_block_idx * 11'd9);

wire        l1_w_valid;
wire [3:0]  l1_w_idx;
wire [127:0] l1_w_data;

dw_weight_cache #(
  .ADDR_W(ADDR_W)
) u_l1_dw_cache (
    .clk           (CLK),
    .rst_n         (RESETn),
    .load_start    (l1_cache_load_start),
    .base_addr     (dw_w_base_blk[ADDR_W-1:0]),
    .load_done     (l1_cache_load_done),

    .ldr_req       (dw_req),
    .ldr_grant     (dw_grant),
    .ldr_base_addr (dw_base),
    .ldr_count     (dw_count),
    .ldr_valid     (dw_valid),
    .ldr_data      (dw_data),
    .ldr_done_sig  (dw_done),

    .w_valid       (l1_w_valid),
    .w_idx         (l1_w_idx),
    .w_data        (l1_w_data)
);


  // L1 prefetch & scanner
  wire        l1_prefetch_start;
  wire [6:0]  l1_prefetch_tile_row;
  wire        l1_buffer_ready, l1_prefetch_done, l1_prefetch_busy;
  wire        l1_read_enable;
  wire [6:0]  l1_read_addr_internal;
  wire [767:0]l1_buffer_out;
  wire        l1_scanner_done, l1_scanner_busy;


// ------------------------------------------------------------
// Detect rising edge of l1_scanner_done (Verilog-2000 friendly)
// ------------------------------------------------------------
reg l1_scanner_done_d1;
wire l1_scanner_done_rise;

assign l1_scanner_done_rise = l1_scanner_done & ~l1_scanner_done_d1;

always @(posedge CLK or negedge RESETn) begin
  if (!RESETn)
    l1_scanner_done_d1 <= 1'b0;
  else
    l1_scanner_done_d1 <= l1_scanner_done;
end

// ------------------------------------------------------------
// DW block started latch:
// - set ONLY when dw_block_start pulse is issued
// - clear ONLY on rising edge of l1_scanner_done (one event per block)
// ------------------------------------------------------------
always @(posedge CLK or negedge RESETn) begin
  if (!RESETn) begin
    dw_block_started <= 1'b0;
  end else begin
    if (le_state != LE_RUN_L1D) begin
      dw_block_started <= 1'b0;
    end else if (l1_scanner_done_rise) begin
      dw_block_started <= 1'b0;
    end else if (dw_block_start) begin
      dw_block_started <= 1'b1;
    end
  end
end

assign dw_block_start =
  (le_state == LE_RUN_L1D) &&
  l1_weight_loaded &&
  bias_valid &&
  !dw_block_started;


  simple_column_scanner_pipeline u_l1_scanner (
    .clk           (CLK),
    .rst_n         (RESETn),
    .start         (dw_block_start),
    .prefetch_start(l1_prefetch_start),
    .prefetch_tile_row(l1_prefetch_tile_row),
    .prefetch_done (l1_prefetch_done),
    .prefetch_busy (l1_prefetch_busy),
    .buffer_ready  (l1_buffer_ready),
    .read_enable   (l1_read_enable),
    .read_addr     (l1_read_addr_internal),
    .busy          (l1_scanner_busy),
    .done          (l1_scanner_done),
    .current_col   ()
  );

  wire [14:0] l1_rd_addr_raw;
  assign l1_feat_rd_addr = {1'b0, l1_rd_addr_raw};

  prefetch_double_buffer u_l1_prefetch (
    .clk         (CLK),
    .rst_n       (RESETn),
    .prefetch_start(l1_prefetch_start),
    .tile_row    (l1_prefetch_tile_row),
    .mem_en      (l1_feat_rd_en),
    .mem_addr    (l1_rd_addr_raw),
    .mem_dout    (feat_rd_data),
    .mem_valid   (feat_rd_valid),
    .read_enable (l1_read_enable),
    .read_addr   (l1_read_addr_internal),
    .buffer_out  (l1_buffer_out),
    .buffer_ready(l1_buffer_ready),
    .prefetch_busy(l1_prefetch_busy),
    .prefetch_done(l1_prefetch_done)
  );

  wire [767:0] l1_column_data;
  wire         l1_column_valid;

  column_passthrough u_l1_col_pass (
    .clk           (CLK),
    .rst_n         (RESETn),
    .column_data_in(l1_buffer_out),
    .column_valid  (l1_read_enable && l1_buffer_ready),
    .column_data_out(l1_column_data),
    .out_valid     (l1_column_valid)
  );

// ------------------------------------------------------------
// DW block index (cin / 16)
// ------------------------------------------------------------
always @(posedge CLK or negedge RESETn) begin
  if (!RESETn) begin
    dw_block_idx <= 7'd0;
  end else begin
    if (le_state == LE_IDLE && start && is_dw_layer) begin
      dw_block_idx <= 7'd0;
    end else if (le_state == LE_RUN_L1D && l1_scanner_done) begin
      if (!dw_last_block)
        dw_block_idx <= dw_block_idx + 7'd1;
    end
  end
end



reg [3:0] wstream_cnt;
reg       weights_ready;

always @(posedge CLK or negedge RESETn) begin
  if (!RESETn) begin
    wstream_cnt   <= 0;
    weights_ready <= 0;
  end else begin
    if (dw_block_start) begin
      wstream_cnt   <= 0;
      weights_ready <= 0;
    end else if (l1_w_valid && !weights_ready) begin
      if (wstream_cnt == 8) begin
        weights_ready <= 1;
      end else begin
        wstream_cnt <= wstream_cnt + 1;
      end
    end
  end
end
  // L1 DW compute
  wire [2047:0] l1_dwc_out_sums;
  wire [63:0]   l1_dwc_out_valids;

dwc_pu u_l1_dwc (
    .clk        (CLK),
    .rst_n      (RESETn),
    .in_valid   (l1_column_valid && weights_ready),
    .column_data(l1_column_data),

    .w_valid    (l1_w_valid),
    .w_idx      (l1_w_idx),
    .w_data     (l1_w_data),

    .out_sums   (l1_dwc_out_sums),
    .out_valids (l1_dwc_out_valids)
);



  // L1 coord & valid mask
  reg [6:0] l1_current_col_d1, l1_current_col_d2, l1_current_col_d3;
  wire [6:0] l1_current_col;

  always @(posedge CLK) begin
    l1_current_col_d1 <= l1_read_addr_internal;
    l1_current_col_d2 <= l1_current_col_d1;
    l1_current_col_d3 <= l1_current_col_d2;
  end
  assign l1_current_col = l1_current_col_d3;

  wire signed [7:0] l1_col_with_pad;
  wire signed [7:0] l1_col_unpadded;
  wire [6:0] l1_output_row;
  wire [6:0] l1_output_col;
 // wire       l1_coord_valid;
  wire [63:0]l1_filtered_valids;
  wire       l1_block_idx_for_quant;

  assign l1_col_with_pad   = {1'b0, l1_current_col};
  assign l1_col_unpadded   = l1_col_with_pad - 8'sd1;
  
// ----------------------------
  // DW coord/stride handling
  // ? img_w/img_hlayer3: 112x112, stride=2
  // stride=2 ???? row/col 1
  // ----------------------------
  wire l1_in_coord_valid =
      (l1_prefetch_tile_row < img_h[6:0]) &&
      (l1_col_unpadded >= 8'sd0) &&
      (l1_col_unpadded < {1'b0, img_w});

  wire [6:0] l1_col_clamped =
      (l1_col_unpadded < 8'sd0) ? 7'd0 :
      (l1_col_unpadded >= {1'b0, img_w}) ? (img_w - 8'd1) :
      l1_col_unpadded[6:0];

  wire l1_take_row = !dw_stride2 || (l1_prefetch_tile_row[0] == 1'b0);
  wire l1_take_col = !dw_stride2 || (l1_col_clamped[0]       == 1'b0);

  wire l1_coord_valid = l1_in_coord_valid && l1_take_row && l1_take_col;

  assign l1_output_row = dw_stride2 ? (l1_prefetch_tile_row >> 1) : l1_prefetch_tile_row;
  assign l1_output_col = dw_stride2 ? (l1_col_clamped        >> 1) : l1_col_clamped;

  assign l1_filtered_valids = l1_coord_valid ? l1_dwc_out_valids : 64'd0;

  // layer1: cin=32 -> 2 blocks, quant  block_idx 1bit ?
  assign l1_block_idx_for_quant = dw_block_idx[0];

  
    wire use_l3_quant = (layer_id == 6'd3);
  wire [7:0] qcfg_out_w  = use_l3_quant ? 8'd56  : 8'd112;
wire [7:0] qcfg_out_h  = use_l3_quant ? 8'd56  : 8'd112;
wire [2:0] qcfg_blocks = use_l3_quant ? 3'd4   : 3'd2;
wire [6:0] q_tile_row = use_l3_quant ? {1'b0, l1_output_row[5:0]} : l1_output_row;
wire [6:0] q_col_idx  = use_l3_quant ? {1'b0, l1_output_col[5:0]} : l1_output_col;

// block idxL1?1bitL32bit
wire [1:0] q_blk_idx  = use_l3_quant ? dw_block_idx[1:0] : {1'b0, dw_block_idx[0]};

wire        dw_wr_en0, dw_wr_en1, dw_wr_en2, dw_wr_en3;
wire [15:0] dw_wr_addr0, dw_wr_addr1, dw_wr_addr2, dw_wr_addr3;
wire [127:0] dw_wr_data0, dw_wr_data1, dw_wr_data2, dw_wr_data3;

quant_l1_stream_4channel #(
  .UNIT_NUM(16),
  .OUT_W_MAX(112),
  .OUT_H_MAX(112),
  .BLOCKS_MAX(4),
  .ACC_W(32),
  .OUT_BITS(8)
) u_dw_quant_stream (
  .clk(CLK),
  .rst_n(RESETn),

  .cfg_out_w(qcfg_out_w),
  .cfg_out_h(qcfg_out_h),
  .cfg_blocks(qcfg_blocks),

  .dwc_sums(l1_dwc_out_sums),
  .dwc_valids(l1_filtered_valids),
  .tile_row(q_tile_row),
  .col_index(q_col_idx),
  .block_idx(q_blk_idx),

  .bias_vec(bias_vec),
  .cfg_mult_scalar(quant_M),
  .cfg_shift_scalar(quant_s),
  .cfg_symmetric(1'b0),
  .cfg_zp_out(quant_zp),

  .wr_en0(dw_wr_en0), .wr_addr0(dw_wr_addr0), .wr_data0(dw_wr_data0),
  .wr_en1(dw_wr_en1), .wr_addr1(dw_wr_addr1), .wr_data1(dw_wr_data1),
  .wr_en2(dw_wr_en2), .wr_addr2(dw_wr_addr2), .wr_data2(dw_wr_data2),
  .wr_en3(dw_wr_en3), .wr_addr3(dw_wr_addr3), .wr_data3(dw_wr_data3)
);


  // Pack into FIFO word
  wire [143:0] fifo0_din = {dw_wr_addr0, dw_wr_data0};
  wire [143:0] fifo1_din = {dw_wr_addr1, dw_wr_data1};
  wire [143:0] fifo2_din = {dw_wr_addr2, dw_wr_data2};
  wire [143:0] fifo3_din = {dw_wr_addr3, dw_wr_data3};

  wire [143:0] fifo0_dout, fifo1_dout, fifo2_dout, fifo3_dout;

  wire fifo0_full, fifo1_full, fifo2_full, fifo3_full;
  wire fifo0_empty, fifo1_empty, fifo2_empty, fifo3_empty;

  // Write enables (drop if full, same as your original)
  wire fifo0_wr_en = dw_wr_en0 && !fifo0_full;
  wire fifo1_wr_en = dw_wr_en1 && !fifo1_full;
  wire fifo2_wr_en = dw_wr_en2 && !fifo2_full;
  wire fifo3_wr_en = dw_wr_en3 && !fifo3_full;

  // Pop arbitration (only one pop per cycle)
  wire pop_fifo0 = !fifo0_empty;
  wire pop_fifo1 = fifo0_empty && !fifo1_empty;
  wire pop_fifo2 = fifo0_empty && fifo1_empty && !fifo2_empty;
  wire pop_fifo3 = fifo0_empty && fifo1_empty && fifo2_empty && !fifo3_empty;

  // ------------------------------------------------------------
  // XPM FIFO instances (BRAM)
  // READ_MODE="fwft" => dout always shows head element when not empty
  // FIFO_READ_LATENCY must be 0 in fwft
  // rst is ACTIVE HIGH, so use !RESETn
  // ------------------------------------------------------------

  xpm_fifo_sync #(
    .FIFO_MEMORY_TYPE("block"),
    .FIFO_WRITE_DEPTH(128),
    .WRITE_DATA_WIDTH(144),
    .READ_DATA_WIDTH(144),
    .READ_MODE("fwft"),
    .FIFO_READ_LATENCY(0),
    .ECC_MODE("no_ecc"),
    .DOUT_RESET_VALUE("0")
  ) u_dw_fifo0 (
    .wr_clk(CLK),
    .rst(!RESETn),
    .din(fifo0_din),
    .wr_en(fifo0_wr_en),
    .full(fifo0_full),

    .rd_en(pop_fifo0),
    .dout(fifo0_dout),
    .empty(fifo0_empty),

    .sleep(1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .prog_full(),
    .wr_data_count(),
    .almost_full(),
    .wr_rst_busy(),
    .rd_rst_busy(),
    .data_valid(),
    .underflow(),
    .overflow(),
    .almost_empty(),
    .prog_empty(),
    .rd_data_count(),
    .sbiterr(),
    .dbiterr()
  );

  xpm_fifo_sync #(
    .FIFO_MEMORY_TYPE("block"),
    .FIFO_WRITE_DEPTH(128),
    .WRITE_DATA_WIDTH(144),
    .READ_DATA_WIDTH(144),
    .READ_MODE("fwft"),
    .FIFO_READ_LATENCY(0),
    .ECC_MODE("no_ecc"),
    .DOUT_RESET_VALUE("0")
  ) u_dw_fifo1 (
    .wr_clk(CLK),
    .rst(!RESETn),
    .din(fifo1_din),
    .wr_en(fifo1_wr_en),
    .full(fifo1_full),

    .rd_en(pop_fifo1),
    .dout(fifo1_dout),
    .empty(fifo1_empty),

    .sleep(1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .prog_full(),
    .wr_data_count(),
    .almost_full(),
    .wr_rst_busy(),
    .rd_rst_busy(),
    .data_valid(),
    .underflow(),
    .overflow(),
    .almost_empty(),
    .prog_empty(),
    .rd_data_count(),
    .sbiterr(),
    .dbiterr()
  );

  xpm_fifo_sync #(
    .FIFO_MEMORY_TYPE("block"),
    .FIFO_WRITE_DEPTH(128),
    .WRITE_DATA_WIDTH(144),
    .READ_DATA_WIDTH(144),
    .READ_MODE("fwft"),
    .FIFO_READ_LATENCY(0),
    .ECC_MODE("no_ecc"),
    .DOUT_RESET_VALUE("0")
  ) u_dw_fifo2 (
    .wr_clk(CLK),
    .rst(!RESETn),
    .din(fifo2_din),
    .wr_en(fifo2_wr_en),
    .full(fifo2_full),

    .rd_en(pop_fifo2),
    .dout(fifo2_dout),
    .empty(fifo2_empty),

    .sleep(1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .prog_full(),
    .wr_data_count(),
    .almost_full(),
    .wr_rst_busy(),
    .rd_rst_busy(),
    .data_valid(),
    .underflow(),
    .overflow(),
    .almost_empty(),
    .prog_empty(),
    .rd_data_count(),
    .sbiterr(),
    .dbiterr()
  );

  xpm_fifo_sync #(
    .FIFO_MEMORY_TYPE("block"),
    .FIFO_WRITE_DEPTH(128),
    .WRITE_DATA_WIDTH(144),
    .READ_DATA_WIDTH(144),
    .READ_MODE("fwft"),
    .FIFO_READ_LATENCY(0),
    .ECC_MODE("no_ecc"),
    .DOUT_RESET_VALUE("0")
  ) u_dw_fifo3 (
    .wr_clk(CLK),
    .rst(!RESETn),
    .din(fifo3_din),
    .wr_en(fifo3_wr_en),
    .full(fifo3_full),

    .rd_en(pop_fifo3),
    .dout(fifo3_dout),
    .empty(fifo3_empty),

    .sleep(1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .prog_full(),
    .wr_data_count(),
    .almost_full(),
    .wr_rst_busy(),
    .rd_rst_busy(),
    .data_valid(),
    .underflow(),
    .overflow(),
    .almost_empty(),
    .prog_empty(),
    .rd_data_count(),
    .sbiterr(),
    .dbiterr()
  );

  // Unpack FIFO head words
  wire [15:0]  fifo0_addr_out = fifo0_dout[143:128];
  wire [127:0] fifo0_data_out = fifo0_dout[127:0];
  wire [15:0]  fifo1_addr_out = fifo1_dout[143:128];
  wire [127:0] fifo1_data_out = fifo1_dout[127:0];
  wire [15:0]  fifo2_addr_out = fifo2_dout[143:128];
  wire [127:0] fifo2_data_out = fifo2_dout[127:0];
  wire [15:0]  fifo3_addr_out = fifo3_dout[143:128];
  wire [127:0] fifo3_data_out = fifo3_dout[127:0];

  // Output selection (same priority as original)
  reg        l1_wr_en_out;
  reg [15:0] l1_wr_addr_out;
  reg [127:0] l1_wr_data_out;

  always @(*) begin
    l1_wr_en_out   = 1'b0;
    l1_wr_addr_out = 16'd0;
    l1_wr_data_out = 128'd0;

    if (!fifo0_empty) begin
      l1_wr_en_out   = 1'b1;
      l1_wr_addr_out = fifo0_addr_out;
      l1_wr_data_out = fifo0_data_out;
    end else if (!fifo1_empty) begin
      l1_wr_en_out   = 1'b1;
      l1_wr_addr_out = fifo1_addr_out;
      l1_wr_data_out = fifo1_data_out;
    end else if (!fifo2_empty) begin
      l1_wr_en_out   = 1'b1;
      l1_wr_addr_out = fifo2_addr_out;
      l1_wr_data_out = fifo2_data_out;
    end else if (!fifo3_empty) begin
      l1_wr_en_out   = 1'b1;
      l1_wr_addr_out = fifo3_addr_out;
      l1_wr_data_out = fifo3_data_out;
    end
  end

  // Connect to feature buffer write interface
  assign l1_feat_wr_en   = l1_wr_en_out;
  assign l1_feat_wr_addr = l1_wr_addr_out;
  assign l1_feat_wr_data = l1_wr_data_out;



  assign l1_done = l1_scanner_done && dw_last_block && dw_block_started;

 // 9. AP: Global Average Pooling (Layer 27)
  // ============================================================
  global_avg_pool #(
    . CHANNELS  (1024),
    .POOL_SIZE (7),
    .DATA_W    (8),
    .ACC_W     (32),
    .LANES     (16)
  ) u_gap (
    .clk               (CLK),
    .rst_n             (RESETn),
    .start             (ap_start),
    .feat_rd_en        (ap_feat_rd_en),
    .feat_rd_local_addr(ap_feat_rd_addr),
    .feat_rd_data      (feat_rd_data),
    .feat_rd_valid     (feat_rd_valid),
    .feat_wr_en        (ap_feat_wr_en),
    .feat_wr_local_addr(ap_feat_wr_addr),
    .feat_wr_data      (ap_feat_wr_data),
    .done              (ap_done)
  );


  // ============================================================
  // 10. FC:  Fully Connected (Layer 28)
  // ============================================================
  fc_layer #(
    .IN_FEATURES (1024),
    .OUT_CLASSES (1000),
    .DATA_W      (8),
    .ACC_W       (32),
    .LANES       (16),
    .ADDR_W      (ADDR_W)
  ) u_fc (
    .clk               (CLK),
    .rst_n             (RESETn),
    .start             (fc_start),
    .w_base            (w_base),
    .b_base            (b_base),
    .quant_M           (quant_M),
    .quant_s           (quant_s),
    .quant_zp          (quant_zp),
    .weight_req        (fc_pw_req),
    .weight_base       (fc_pw_base),
    .weight_count      (fc_pw_count),
    .weight_grant      (pw_grant),
    .weight_valid      (pw_valid),
    .weight_data       (pw_data),
    .weight_done       (pw_done),
    .bias_vec          (bias_vec),
    .bias_valid        (bias_valid),
    .bias_block_idx    (fc_bias_block_idx),
    .bias_rd_en        (),  // FSM
    .feat_rd_en        (fc_feat_rd_en),
    .feat_rd_local_addr(fc_feat_rd_addr),
    .feat_rd_data      (feat_rd_data),
    .feat_rd_valid     (feat_rd_valid),
    .out_valid         (fc_out_valid),
    .out_class_idx     (fc_out_class_idx),
    .out_logit         (fc_out_logit),
    .done              (fc_done)
  );

  assign fc_feat_wr_en   = 1'b0;
  assign fc_feat_wr_addr = 16'd0;
  assign fc_feat_wr_data = 128'd0;

endmodule