`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:37:49
// Design Name: 
// Module Name: simple_column_scanner_pipeline
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

module simple_column_scanner_pipeline #(
  parameter integer OUT_W   = 112,
  parameter integer OUT_H   = 112,
  parameter integer TILE_H  = 6,
  parameter integer COUT    = 32,
  parameter integer UNIT_NUM= 16,
  parameter integer K       = 3,
  parameter integer PADDING = 1
)(
  input  wire clk,
  input  wire rst_n,

  // 批次启动信号：每个 batch 进来打一拍即可
  input  wire start,

  // Prefetch 控制
  output reg  prefetch_start,
  output reg  [$clog2(OUT_H)-1:0] prefetch_tile_row,
  input  wire prefetch_done,
  input  wire prefetch_busy,
  input  wire buffer_ready,

  // 读出控制
  output wire read_enable,
  output wire [$clog2(OUT_W+2*PADDING)-1:0] read_addr,

  // 状态指示
  output reg  busy,
  output reg  done,
  output reg  [$clog2(OUT_W+2*PADDING)-1:0] current_col
);

  // 几何参数
  localparam integer STRIDE    = TILE_H - K + 1;
  localparam integer PADDED_W  = OUT_W + 2 * PADDING;
  localparam integer PADDED_H  = OUT_H + 2 * PADDING;
  localparam integer NUM_TILES = (PADDED_H - TILE_H) / STRIDE + 1;

  // 状态机
  localparam IDLE          = 3'd0;
  localparam PREFETCH_FIRST= 3'd1;
  localparam SCAN          = 3'd2;
  localparam DONE_ST       = 3'd3;

  reg [2:0] state;

  reg [$clog2(NUM_TILES)-1:0] scan_tile_idx;
  reg [$clog2(PADDED_W)-1:0]  col_counter;

  // FSM
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= IDLE;
      scan_tile_idx    <= 0;
      col_counter      <= 0;
      prefetch_start   <= 0;
      prefetch_tile_row<= 0;
      busy             <= 0;
      done             <= 0;
      current_col      <= 0;
    end else begin
      // 默认清脉冲
      prefetch_start <= 0;
      done           <= 0;

      // start 作为高优先级软复位：无论当前在什么状态，来一个 start 就从头开始扫描
      if (start) begin
        busy             <= 1'b1;
        state            <= PREFETCH_FIRST;
        scan_tile_idx    <= 0;
        col_counter      <= 0;
        prefetch_tile_row<= 0;
        prefetch_start   <= 1'b1;   // 触发第一个 tile 的预取
      end else begin
        case (state)
          IDLE: begin
            busy <= 1'b0;
            // 不在这里等待 start，因为上面已经处理了 start 分支
          end

          PREFETCH_FIRST: begin
            // 等待 prefetch_double_buffer 把当前 tile 填满
            if (prefetch_done && buffer_ready) begin
              col_counter <= 0;
              state       <= SCAN;
            end
          end

          SCAN: begin
            // 一列一列地扫描当前 tile
            current_col <= col_counter;

            if (col_counter < PADDED_W-1) begin
              col_counter <= col_counter + 1'b1;
            end else begin
              // 当前 tile 的一行扫描完毕
              if (scan_tile_idx < NUM_TILES-1) begin
                // 还有下一 tile 行
                scan_tile_idx    <= scan_tile_idx + 1'b1;
                col_counter      <= 0;
                prefetch_tile_row<= prefetch_tile_row + STRIDE;
                prefetch_start   <= 1'b1;
                state            <= PREFETCH_FIRST;
              end else begin
                // 所有 tile 扫描完毕
                state <= DONE_ST;
              end
            end
          end

          DONE_ST: begin
            busy <= 1'b0;
            done <= 1'b1;   // 拉 done 一拍
            state <= IDLE;
          end

          default: begin
            state <= IDLE;
          end
        endcase
      end
    end
  end

  // 读使能 / 地址：只要在 SCAN 状态，且 buffer_ready，就可以从 prefetch buffer 读对应列
  assign read_enable = (state == SCAN) && buffer_ready;
  assign read_addr   = col_counter;

endmodule