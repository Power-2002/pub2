`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/22 19:45:02
// Design Name: 
// Module Name: conv1_scheduler_32x16
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

//////////////////////////////////////////////////////////////////////////////////
// Conv1 Scheduler for 32×16 PE Array
// 
// Layer 0: 3×3 Conv, 3 input channels → 32 output channels
// 输入窗口:  3×3×3 = 27 个值
// 需要将 27 个值映射到 32 行激活输入
//////////////////////////////////////////////////////////////////////////////////
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Conv1 Scheduler for 32×16 PE Array
// 输入:  3×3×3 窗口 (27 bytes)
// 输出: 32 个通道的累加结果
//////////////////////////////////////////////////////////////////////////////////
module conv1_scheduler_32x16 #(
  parameter NUM_ROWS = 32,    // 改为 32
  parameter NUM_COLS = 16,
  parameter A_BITS   = 8,
  parameter W_BITS   = 8,
  parameter ACC_BITS = 32,
  parameter ADDR_W   = 19
)(
  input  wire                      CLK,
  input  wire                      RESET,
  input  wire                      start,
  output reg                       done,

  input  wire [ADDR_W-1: 0]         w_base_in,

  // Window input
  output reg                       win_req,
  input  wire                      win_valid,
  input  wire [27*A_BITS-1:0]      win_flat,

  // Weight interface
  output reg                       weight_req,
  input  wire                      weight_grant,
  output reg  [ADDR_W-1:0]         weight_base,
  output reg  [10: 0]               weight_count,
  input  wire                      weight_valid,
  input  wire [127:0]              weight_data,
  input  wire                      weight_done,

  // Array I/F (32×16)
  output wire                      arr_W_EN,
  output reg  [NUM_COLS*W_BITS-1:0] in_weight_above,
  output reg  [NUM_ROWS*A_BITS-1:0] active_left,    // 256 bits
  input  wire [NUM_COLS*ACC_BITS-1:0] out_sum_final,

  output reg                       y_valid,
  output reg  [NUM_COLS*ACC_BITS-1:0] y_data,
  output reg                       y_tile_sel
);

  localparam integer PE_LAT = NUM_ROWS - 1;  // 31 cycles

  // Weight loader
  reg         loader_start;
  reg  [ADDR_W-1:0] loader_base;
  reg         loader_busy;
  wire        loader_done;
  
  assign loader_done = weight_done;
  assign arr_W_EN    = weight_valid;
  
  always @(posedge CLK or negedge RESET) begin
    if (!RESET) begin
      weight_req   <= 1'b0;
      weight_base  <= 0;
      weight_count <= 11'd16;
      loader_busy  <= 1'b0;
    end else begin
      if (loader_start && ! loader_busy) begin
        weight_req   <= 1'b1;
        weight_base  <= loader_base;
        weight_count <= 11'd16;
        loader_busy  <= 1'b1;
      end
      if (weight_grant && weight_req)
        weight_req <= 1'b0;
      if (weight_done && loader_busy)
        loader_busy <= 1'b0;
    end
  end

  // Window register
  reg [27*A_BITS-1:0] win_reg;
  
  // 从窗口提取字节
  function [A_BITS-1:0] kbyte;
    input [27*A_BITS-1:0] bus;
    input integer k;
    begin
      kbyte = (k >= 0 && k < 27) ? bus[k*A_BITS +:  A_BITS] : 8'd0;
    end
  endfunction

  // 构建 32-channel 激活向量
  // Conv1: 3×3×3 = 27 个有效值，其余填 0
  wire [NUM_ROWS*A_BITS-1:0] vec0;  // 前 27 + 5个0
  
  genvar gi;
  generate
    for (gi = 0; gi < NUM_ROWS; gi = gi + 1) begin : GEN_VEC
      if (gi < 27)
        assign vec0[gi*A_BITS +:  A_BITS] = kbyte(win_reg, gi);
      else
        assign vec0[gi*A_BITS +:  A_BITS] = 8'd0;  // 填充 0
    end
  endgenerate

  // FSM
  localparam S_IDLE       = 4'd0;
  localparam S_REQ_WIN    = 4'd1;
  localparam S_WAIT_WIN   = 4'd2;
  localparam S_T0_LOAD    = 4'd3;
  localparam S_T0_WAITLD  = 4'd4;
  localparam S_T0_INJECT  = 4'd5;
  localparam S_T0_WAITLAT = 4'd6;
  localparam S_T0_CAPTURE = 4'd7;
  localparam S_T0_OUT     = 4'd8;
  localparam S_T1_LOAD    = 4'd9;
  localparam S_T1_WAITLD  = 4'd10;
  localparam S_T1_INJECT  = 4'd11;
  localparam S_T1_WAITLAT = 4'd12;
  localparam S_T1_CAPTURE = 4'd13;
  localparam S_T1_OUT     = 4'd14;
  localparam S_DONE       = 4'd15;

  reg [3:0] state, state_n;
  reg [5:0] wait_cnt;
  reg [4:0] cap_col;
  reg signed [ACC_BITS-1:0] psum [0:NUM_COLS-1];

  integer i;

  // State transition
  always @(*) begin
    state_n = state;
    case (state)
      S_IDLE:        state_n = start ? S_REQ_WIN : S_IDLE;
      S_REQ_WIN:    state_n = S_WAIT_WIN;
      S_WAIT_WIN:   state_n = win_valid ? S_T0_LOAD : S_WAIT_WIN;
      
      S_T0_LOAD:    state_n = S_T0_WAITLD;
      S_T0_WAITLD:  state_n = loader_done ? S_T0_INJECT : S_T0_WAITLD;
      S_T0_INJECT:   state_n = S_T0_WAITLAT;
      S_T0_WAITLAT: state_n = (wait_cnt >= PE_LAT) ? S_T0_CAPTURE : S_T0_WAITLAT;
      S_T0_CAPTURE: state_n = (cap_col == NUM_COLS-1) ? S_T0_OUT :  S_T0_CAPTURE;
      S_T0_OUT:     state_n = S_T1_LOAD;
      
      S_T1_LOAD:    state_n = S_T1_WAITLD;
      S_T1_WAITLD:  state_n = loader_done ? S_T1_INJECT : S_T1_WAITLD;
      S_T1_INJECT:   state_n = S_T1_WAITLAT;
      S_T1_WAITLAT: state_n = (wait_cnt >= PE_LAT) ? S_T1_CAPTURE :  S_T1_WAITLAT;
      S_T1_CAPTURE:  state_n = (cap_col == NUM_COLS-1) ? S_T1_OUT : S_T1_CAPTURE;
      S_T1_OUT:     state_n = S_DONE;
      
      S_DONE:        state_n = S_IDLE;
      default:      state_n = S_IDLE;
    endcase
  end

  // Sequential logic
  always @(posedge CLK or negedge RESET) begin
    if (!RESET) begin
      state <= S_IDLE;
      win_req <= 0;
      win_reg <= 0;
      loader_start <= 0;
      loader_base <= 0;
      in_weight_above <= 0;
      active_left <= 0;
      wait_cnt <= 0;
      cap_col <= 0;
      y_valid <= 0;
      y_data <= 0;
      y_tile_sel <= 0;
      done <= 0;
      for (i = 0; i < NUM_COLS; i = i + 1) psum[i] <= 0;
    end else begin
      state <= state_n;
      win_req <= 0;
      loader_start <= 0;
      active_left <= 0;
      y_valid <= 0;
      done <= 0;

      // Weight loading
      if (weight_valid)
        in_weight_above <= weight_data;

      case (state)
        S_IDLE:  begin
          if (start) win_req <= 1;
        end
        
        S_REQ_WIN: win_req <= 1;
        
        S_WAIT_WIN: begin
          win_req <= ! win_valid;
          if (win_valid) win_reg <= win_flat;
        end
        
        // Tile 0
        S_T0_LOAD: begin
          loader_base <= w_base_in;
          loader_start <= 1;
          for (i = 0; i < NUM_COLS; i = i + 1) psum[i] <= 0;
        end
        
        S_T0_INJECT: begin
          active_left <= vec0;
          wait_cnt <= 0;
          cap_col <= 0;
        end
        
        S_T0_WAITLAT:  begin
          wait_cnt <= wait_cnt + 1;
        end
        
        S_T0_CAPTURE: begin
          psum[cap_col] <= out_sum_final[cap_col*ACC_BITS +: ACC_BITS];
          cap_col <= cap_col + 1;
        end
        
        S_T0_OUT: begin
          for (i = 0; i < NUM_COLS; i = i + 1)
            y_data[i*ACC_BITS +: ACC_BITS] <= psum[i];
          y_tile_sel <= 0;
          y_valid <= 1;
        end
        
        // Tile 1
        S_T1_LOAD: begin
          loader_base <= w_base_in + 19'd16;
          loader_start <= 1;
          for (i = 0; i < NUM_COLS; i = i + 1) psum[i] <= 0;
        end
        
        S_T1_INJECT: begin
          active_left <= vec0;
          wait_cnt <= 0;
          cap_col <= 0;
        end
        
        S_T1_WAITLAT: begin
          wait_cnt <= wait_cnt + 1;
        end
        
        S_T1_CAPTURE:  begin
          psum[cap_col] <= out_sum_final[cap_col*ACC_BITS +: ACC_BITS];
          cap_col <= cap_col + 1;
        end
        
        S_T1_OUT: begin
          for (i = 0; i < NUM_COLS; i = i + 1)
            y_data[i*ACC_BITS +: ACC_BITS] <= psum[i];
          y_tile_sel <= 1;
          y_valid <= 1;
        end
        
        S_DONE: done <= 1;
      endcase
    end
  end

endmodule