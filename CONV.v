`timescale 1ns/10ps

module CONV(
        input       clk,
        input       reset,
        output reg busy,
        input    ready,

        output [11:0]iaddr,
        input [19:0]idata,
        output reg  cwr,
        output reg [11:0] caddr_wr,
        output reg  [19:0] cdata_wr,

        output      crd,
        output reg [11:0] caddr_rd,
        
        input   [19:0] cdata_rd,
        output reg [2:0] csel
        
        
        );

parameter [2:0] L0      = 3'b000,
                L0_K0   = 3'b001,
                L0_K1   = 3'b010,
                L1_K0   = 3'b011,
                L1_K1   = 3'b100,
                L2      = 3'b101;               

reg  [2:0] state;
reg  [2:0] next_state;
wire [19:0]out0;
wire [19:0]out1;
wire [19:0]kernel0[0:8];
wire [19:0]kernel1[0:8];
reg L0_done;
reg L0_K0_done;
reg L0_K1_done;
reg L1_K0_done;
reg L1_K1_done;
reg L2_done;
reg [12:0]count_all;
reg [19:0]linebuff[0:134];
reg count_buff_en;
reg [12:0]buff_limit;
reg [11:0]count_buff;
integer i;
reg count_all_4356;
reg [19:0]poolbuff[0:65];//for max pooling
reg [19:0]poolbuff_cnt;
wire poolbuff_cnt_4095;
reg L1_K0_start;
reg L1_K1_start;
reg [9:0]stride_cnt;
reg stride_cnt_en;
reg [9:0]delay1;

reg [19:0]compare[0:3];
reg [19:0]temp;
reg [11:0]poolbuff_cnt_limit;
reg  wr_pool;
reg [9:0]wr_pool_cnt;
reg [19:0]temp_poolbuff[0:1023];
reg wr_temp;
reg wr_pool_cnt_rst;

reg L2_start;

parameter [1:0] R_K0 = 2'b00,
                R_K1 = 2'b01,
                W_L2 = 3'b10;
reg       [1:0] st;
reg       [1:0] nst;
reg       [11:0] L2_read_buff_cnt;
reg       [10:0]flatten_cnt;
reg       [11:0] cnt_delay;

reg st_0;
reg st_1;
reg [19:0]k0_buff;
reg [19:0]k1_buff;

reg delay_L2_start;
reg delay2_L2_start;

wire wr_pool_cnt_1024;
assign wr_pool_cnt_1024 = wr_pool_cnt >= 12'd1024;


assign kernel0[0] = 20'h0A89E;
assign kernel0[1] = 20'h092D5;
assign kernel0[2] = 20'h06D43;
assign kernel0[3] = 20'h01004;
assign kernel0[4] = 20'hF8F71;
assign kernel0[5] = 20'hF6E54;
assign kernel0[6] = 20'hFA6D7;
assign kernel0[7] = 20'hFC834;
assign kernel0[8] = 20'hFAC19;

assign kernel1[0] = 20'hFDB55;
assign kernel1[1] = 20'h02992;
assign kernel1[2] = 20'hFC994;
assign kernel1[3] = 20'h050FD;
assign kernel1[4] = 20'h02F20;
assign kernel1[5] = 20'h0202D;
assign kernel1[6] = 20'h03BD7;
assign kernel1[7] = 20'hFD369;
assign kernel1[8] = 20'h05E68;

assign iaddr = count_buff;
assign poolbuff_cnt_4095 = poolbuff_cnt >= 13'd4097; //end of pooling reading value and starting put value into memory 


always @(posedge clk)begin
    if(reset)begin
        state <= L0_K0;
    end
    else begin
        state <= next_state;
    end
    
end

always @(*)begin
    next_state = state;
    case(state)
        L0_K0: next_state = L0_K0_done  ? L0_K1 : L0_K0;
        L0_K1: next_state = L0_K1_done  ? L1_K0 : L0_K1;
        L1_K0: next_state = L1_K0_done  ? L1_K1 : L1_K0;
        L1_K1: next_state = L1_K1_done  ? L2    : L1_K1;
        L2   : next_state = L2_done     ? L0_K0 : L2;
    endcase
end


always @(*) begin
    L0_K0_done = 1'b0;
    L0_K1_done = 1'b0;
    L1_K0_done = 1'b0;
    L1_K1_done = 1'b0;
    L1_K0_start = 1'b0;
    L1_K1_start = 1'b0;
    delay_L2_start = 1'b0;
    L2_done  = 1'b0;
    csel = 3'b000;
    wr_pool=1'b0;
    wr_pool_cnt_rst=1'b0;
    case(state)
        L0_K0:begin
            if(count_all >= 13'd134)begin
                csel = 3'b001;
            end
            if(caddr_wr==12'd4095)begin
                L0_K0_done = 1'b1;
            end
        end
        L0_K1:begin
            if(count_all >= 13'd134)begin
                csel = 3'b010;
            end
            if(caddr_wr==12'd4095)begin
                L0_K1_done = 1'b1;
            end
        end
        L1_K0:begin
            L1_K0_start = 1'b1;
            if(poolbuff_cnt<=13'd4097)begin
                csel = 3'b001;
            end
            else begin
                csel = 3'b011;
                wr_pool=1'b1;
            end
            if(poolbuff_cnt==13'd5122)begin
                L1_K0_done = 1'b1;
            end 
        end
        L1_K1:begin
            L1_K1_start = 1'b1;
            if(poolbuff_cnt <= 13'd4097)begin
                csel = 3'b010;
            end
            else begin 
                csel = 3'b100;
                wr_pool = 1'b1;
            end
            if(poolbuff_cnt>=13'd5122)begin
                L1_K1_done=1'b1;
            end
        end
        L2:begin
            delay_L2_start=1'b1;
            if(st_0)begin
                csel = 3'b011;
            end
            if(st_1)begin
                csel = 3'b100;
            end
            if(~st_0&&~st_1)begin
                csel = 3'b101;
            end
            
        end
    endcase
end

always@(posedge clk)begin
    if(reset)begin
        delay2_L2_start <= 1'b0;
        L2_start <= 1'b0;
    end
    else begin
        delay2_L2_start <= delay_L2_start;
        L2_start <= delay2_L2_start;
    end
end


always @(*)begin
    if(count_all>=13'd4357)begin
        if(csel==3'b001||csel==3'b010)begin
            count_all_4356 = 1'b1;
        end
        else begin
            count_all_4356 = 1'b0;
        end
    end
    else begin
        count_all_4356 = 1'b0;
    end
end

always @(posedge clk)begin
    if(reset||count_all_4356)begin
        count_all <= 13'b0;
    end
    else begin
        if(~count_all_4356)begin
            count_all <= count_all + 13'b1;
        end
        else begin
            count_all <= count_all + 13'b0;
        end 
    end
end



always @(posedge clk)begin
    if(reset||count_all_4356)begin
        count_buff_en <= 1'b0;
        buff_limit <= 13'd130;
    end
    else begin
        if(count_all == buff_limit || count_all == buff_limit+13'b1)begin
            count_buff_en <= 1'b0;
            if(count_all == buff_limit + 13'b1) begin
                count_buff_en <= 1'b0;
                buff_limit <= buff_limit + 32'd66;
            end
        end
        else begin
            if(count_all <= 13'd4288 && count_all >=13'd66)begin//the first row should be zero
                count_buff_en <= 1'b1;
            end
            else begin
                count_buff_en <= 1'b0;
            end
        end
    end
end


always @(posedge clk)begin
    if(reset||count_all_4356)begin
        count_buff <=12'b0;
    end
    else begin
        if(~count_all_4356)begin
            if(count_buff_en == 1'b0)begin
                count_buff <= count_buff + 12'b0;
            end
            else begin
                if(count_all != 13'd4288)begin
                    count_buff <= count_buff + 12'b1;
                end
                else begin
                    count_buff <= count_buff + 12'b0;
                end
            end
        end
    end
end
   

always @(posedge clk)begin
    if(reset||count_all_4356)begin
        for(i=0;i<135;i=i+1)begin
            linebuff[i] <= 20'b0;  
        end
    end
    else begin
        if(~count_all_4356)begin
            if(count_buff_en == 1'b0)begin
                linebuff[134] <= 20'b0;
                for(i=0;i<134;i=i+1)begin
                    linebuff[i] <= linebuff[i+1];
                end
            end
            else begin
                linebuff[134] <= idata;
                for(i=0;i<134;i=i+1)begin
                    linebuff[i] <= linebuff[i+1];
                end
            end
        end
    end
end


always @(posedge clk)begin
    if(reset)begin
        busy <= 1'b0;
    end
    else begin
        if(L2_read_buff_cnt<=12'd1024)begin
            busy <= 1'b1;
        end
        else begin
            busy <= 1'b0;
        end

    end
end

wire [39:0]partial_p_k0[0:8];
wire [39:0]partial_p_k1[0:8];
wire [19:0]partial_p_r_k0;
wire [19:0]partial_p_r_k1;


assign partial_p_k0[0] = $signed(linebuff[0])   * $signed(kernel0[0]);
assign partial_p_k0[1] = $signed(linebuff[1])   * $signed(kernel0[1]);
assign partial_p_k0[2] = $signed(linebuff[2])   * $signed(kernel0[2]);
assign partial_p_k0[3] = $signed(linebuff[66])  * $signed(kernel0[3]);
assign partial_p_k0[4] = $signed(linebuff[67])  * $signed(kernel0[4]);
assign partial_p_k0[5] = $signed(linebuff[68])  * $signed(kernel0[5]);
assign partial_p_k0[6] = $signed(linebuff[132]) * $signed(kernel0[6]);
assign partial_p_k0[7] = $signed(linebuff[133]) * $signed(kernel0[7]);
assign partial_p_k0[8] = $signed(linebuff[134]) * $signed(kernel0[8]);

assign partial_p_k1[0] = $signed(linebuff[0])   * $signed(kernel1[0]);
assign partial_p_k1[1] = $signed(linebuff[1])   * $signed(kernel1[1]);
assign partial_p_k1[2] = $signed(linebuff[2])   * $signed(kernel1[2]);
assign partial_p_k1[3] = $signed(linebuff[66])  * $signed(kernel1[3]);
assign partial_p_k1[4] = $signed(linebuff[67])  * $signed(kernel1[4]);
assign partial_p_k1[5] = $signed(linebuff[68])  * $signed(kernel1[5]);
assign partial_p_k1[6] = $signed(linebuff[132]) * $signed(kernel1[6]);
assign partial_p_k1[7] = $signed(linebuff[133]) * $signed(kernel1[7]);
assign partial_p_k1[8] = $signed(linebuff[134]) * $signed(kernel1[8]);

wire [39:0] partial_p_r_k040bit;



assign partial_p_r_k040bit = partial_p_k0[0]+partial_p_k0[1]+partial_p_k0[2]+
                             partial_p_k0[3]+partial_p_k0[4]+partial_p_k0[5]+
                             partial_p_k0[6]+partial_p_k0[7]+partial_p_k0[8];
wire [39:0] partial_p_r_k140bit;

assign partial_p_r_k140bit = partial_p_k1[0]+partial_p_k1[1]+partial_p_k1[2]+
                             partial_p_k1[3]+partial_p_k1[4]+partial_p_k1[5]+
                             partial_p_k1[6]+partial_p_k1[7]+partial_p_k1[8];


assign partial_p_r_k0 = partial_p_r_k040bit[35:16] + partial_p_r_k040bit[15];

assign partial_p_r_k1 = partial_p_r_k140bit[35:16] + partial_p_r_k140bit[15];

assign out0 = partial_p_r_k0 + 20'h01310;
assign out1 = partial_p_r_k1 + 20'hF7295;



reg [12:0] mem_cnt_limit;
reg mem_cnt_en;
reg [11:0] mem_cnt;
wire stop;
wire stop_negedge;

always @(posedge clk)begin
    if(reset||count_all_4356)begin
        mem_cnt_limit <= 13'd198;
        mem_cnt_en <= 1'b0;
    end
    else begin
        if(count_all == mem_cnt_limit||count_all == mem_cnt_limit + 13'b1)begin
                mem_cnt_en <= 1'b0;
                if(count_all == mem_cnt_limit + 13'b1)begin
                    mem_cnt_en <= 1'b0;
                    if(count_all <= 13'd4289)begin
                        mem_cnt_limit <= mem_cnt_limit + 13'd66;
                    end
                end
        end
        else begin
            if(count_all >= 13'd134)begin
                mem_cnt_en <= 1'b1;
            end
        end

    end
end


always @(posedge clk)begin
    if(reset||count_all_4356)begin
        mem_cnt <= 12'b0;
    end
    else begin
        if(mem_cnt_en)begin
            mem_cnt <= mem_cnt + 12'b1;
        end
        else begin
            mem_cnt <= mem_cnt + 12'b0;
        end
    end
end



always @(posedge clk)begin
    if(reset||count_all_4356||wr_pool_cnt_rst)begin
        caddr_wr <= 12'b0;
    end
    else begin
        if(~L2_start&&~L1_K1_done)begin
            if((csel == 3'b001 || csel == 3'b010)&&~L1_K0_start&&~L1_K1_start)begin
                caddr_wr <= mem_cnt;
            end
            if(csel == 3'b011)begin
                caddr_wr <= wr_pool_cnt;
            end
            if(csel == 3'b100)begin
                caddr_wr <= wr_pool_cnt;
            end
        end
        else begin
            if(csel==3'b101)begin
                caddr_wr<=flatten_cnt;
            end
        end

    end
end
reg [19:0]cnt;
always@(posedge clk)begin
    if(reset)begin
        cnt <= 20'b0;
    end
    else begin
        cnt <= cnt + 1'b1;
    end
end
assign stop = cnt>=20'd18962;
/*
reg [19:0]cnt_negedge;
always@(negedge clk)begin
    if(reset)begin
        cnt_negedge <= 20'b0;
    end
    else begin
        cnt_negedge <= cnt_negedge + 1'b1;
    end
end

assign stop_negedge = cnt_negedge == 20'd18964;
*/

reg cdata_wr_rst;
always@(posedge clk)begin
    if(reset)begin
        cdata_wr_rst <= 1'b0;
    end
    else begin
        if(~L1_K1_start)begin
            cdata_wr_rst <= 1'b1;
        end
        else begin
            cdata_wr_rst <= 1'b0;
        end
    end
end

always @(posedge clk)begin
    if(reset||count_all_4356||stop_negedge)begin
        cdata_wr <= 20'b0;
    end
    else begin
        if(~L2_start)begin
            if(mem_cnt_en)begin
                if(csel==3'b001)begin
                    if(out0[19]==1'b1)begin
                        cdata_wr <= 20'b0;
                    end
                    else begin
                        cdata_wr<=out0;
                    end
                end
                if(csel==3'b010)begin
                    if(out1[19]==1'b1)begin
                        cdata_wr <= 20'b0;
                    end
                    else begin
                        cdata_wr <= out1;
                    end
                end
            end
            if(csel==3'b011)begin
                cdata_wr <= temp_poolbuff[wr_pool_cnt];
            end
            if(csel==3'b100)begin
                cdata_wr <= temp_poolbuff[wr_pool_cnt];
            end
        end
        else begin
            
            if(csel == 3'b101&&flatten_cnt[0]==1'b0)begin
                cdata_wr <= k0_buff;
            end
            if(csel == 3'b101&&flatten_cnt[0]==1'b1)begin
                cdata_wr <= k1_buff;
            end
        end
    end
end


always @(*)begin
    if(~L2_start)begin
        if(caddr_wr<=12'd4095)begin
            cwr=1'b1;
        end
        else begin
            cwr=1'b0;
        end
    end
    else begin
        if(csel==3'b011||csel==3'b100)begin
            cwr=1'b0;
        end
        else begin
            cwr=1'b1;
        end
    end
end



always @(posedge clk) begin
    if(reset||L1_K0_done)begin
        poolbuff_cnt <= 20'b0;
    end
    else begin
        if(L1_K0_start||L1_K1_start)begin
            poolbuff_cnt <= poolbuff_cnt + 1'b1;
        end
    end
end

always@(*)begin
    if(poolbuff_cnt<128)begin
        poolbuff_cnt_limit = 10'd128;
    end
    else begin
        if(poolbuff_cnt_limit < 12'd3968)begin
            if(poolbuff_cnt == poolbuff_cnt_limit + 10'd65)begin
                poolbuff_cnt_limit = poolbuff_cnt_limit + 10'd128;
            end
        end
        else begin
            if(poolbuff_cnt == poolbuff_cnt_limit + 10'd65)begin
                poolbuff_cnt_limit = poolbuff_cnt_limit + 10'd127;
            end
        end
    end
end


always@(*)begin
    if(poolbuff_cnt<=10'd65)begin
        stride_cnt_en = 1'b0;
    end
    else if (poolbuff_cnt!=13'd4096)begin
        if(poolbuff_cnt>poolbuff_cnt_limit)begin
            stride_cnt_en = 1'b0;
        end
        else begin
            stride_cnt_en = 1'b1;
        end
    end
    else if(poolbuff_cnt==13'd4096)begin
        stride_cnt_en = 1'b1;
    end
    else begin
        stride_cnt_en = 1'b1;
    end
end


always@(*)begin
    if(~L2_start)begin
        caddr_rd = poolbuff_cnt;
        st_0=1'b0;
        st_1=1'b0;
    end
    else begin
        if(st==2'b00)begin/////////////////watch out latch!!!!!!!??
            caddr_rd = L2_read_buff_cnt;
            st_0=1'b1;
            st_1=1'b0;
        end
        if(st==2'b01)begin
            caddr_rd = L2_read_buff_cnt;
            st_1=1'b1;
            st_0=1'b0;
        end
        if(st==2'b10)begin
            st_0=1'b0;
            st_1=1'b0;
        end
    end
end

assign crd = L1_K0_start||L1_K1_start||L2_start;

always @(posedge clk)begin
    if(reset||L1_K0_done)begin
        for(i=0;i<66;i=i+1)begin
            poolbuff[i] <= 20'b0;
        end
    end
    else begin
        if(L1_K0_start||L1_K1_start)begin
            poolbuff[65] <= cdata_rd;
            for(i=0;i<65;i=i+1)begin
                poolbuff[i] <= poolbuff[i+1];
            end
        end
    end
end

always @(*) begin
    if(stride_cnt_en)begin
        temp = 1'b0;
        compare[0]=poolbuff[0];
        compare[1]=poolbuff[1];
        compare[2]=poolbuff[64];
        compare[3]=poolbuff[65];
        for(i=0;i<4;i=i+1)begin
            if(compare[i]>temp)begin
                temp=compare[i];
            end
        end
    end
    else begin
        temp = 20'd0;
    end
end

always@(posedge clk)begin
    if(reset||L1_K0_done)begin
        stride_cnt <=10'b0;
        delay1     <=10'b0;
    end
    else begin
        if(stride_cnt_en)begin
            delay1     <= stride_cnt+1'b1;
            stride_cnt <= delay1;
        end
        else begin
            stride_cnt <= stride_cnt +1'b0;
        end
    end
end

always@(posedge clk)begin
    if(reset||L1_K0_done)begin
        wr_temp <= 1'b0;
    end
    else begin
        if(poolbuff_cnt >= 12'd65)begin
            if(poolbuff_cnt[0]==1'b0)begin
                wr_temp <= 1'b0;
            end
            else begin
                wr_temp <= 1'b1;
            end
        end
    end
end

always @(posedge clk)begin
    if(reset||L1_K0_done)begin
        for(i=0;i<1024;i=i+1)begin
            temp_poolbuff[i] <= 20'b0;
        end
    end
    else begin
        if(wr_temp&&stride_cnt_en)begin
            temp_poolbuff[stride_cnt] <= temp;    
        end
    end
end

always@(posedge clk)begin
    if(reset||L1_K0_done)begin
        wr_pool_cnt <= 10'b0;
    end
    else begin
        if(wr_pool==1'b1)begin
            wr_pool_cnt <= wr_pool_cnt + 1'b1;
        end
        else begin
            wr_pool_cnt <= wr_pool_cnt + 1'b0;
        end
    end
end


always@(posedge clk)begin
    if(reset)begin
        st <= R_K0;
    end
    else begin
        if(L2_start)begin
            st <= nst;
        end
    end
end

always @(*)begin
    if(L2_start)begin
        nst = st;
        case (st)
            R_K0: nst = R_K1;
            R_K1: nst = W_L2;
            W_L2: nst = flatten_cnt[0]==1'b1 ? R_K0 : W_L2;
        endcase
    end
end

always @(posedge clk)begin
    if(reset)begin
        L2_read_buff_cnt <= 12'b0;
        cnt_delay <=12'b0;
    end
    else begin
        if(L2_start&&(st==2'b0||st==2'b01))begin
            cnt_delay <= L2_read_buff_cnt + 1'b1;
            L2_read_buff_cnt <= cnt_delay;
        end
    end
end

always @(posedge clk)begin
    if(reset)begin
        flatten_cnt <= 1'b0;
    end
    else begin
        if(L2_start && (st==2'b10))begin
            flatten_cnt <= flatten_cnt + 1'b1;
        end
    end
end


always@(posedge clk)begin
    if(reset)begin
        k0_buff<=1'b0;
    end
    else begin
        if(st_0)begin
            k0_buff <= cdata_rd;
        end
    end
end

always@(posedge clk)begin
    if(reset)begin
        k1_buff<=1'b0;
    end
    else begin
        if(st_1)begin
            k1_buff <= cdata_rd;
        end
    end
end
endmodule
