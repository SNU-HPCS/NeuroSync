import GlobalVars as GV
import heapq
import sys
from collections import deque


cdef class Switch:
    def __init__(self, int x, int y):
        self.cyc = 0
        self.timestep = 0

        # Activation flags
        self.is_active = False
        self.flag_deact = False
        self.in_activation_list = False
        self.in_deactivation_list = False

        # Switch position
        self.x = x
        self.y = y
        self.x_in_chip = self.x % GV.sim_params["max_core_x_in_chip"]
        self.y_in_chip = self.y % GV.sim_params["max_core_y_in_chip"]
        self.chip_x = self.x // GV.sim_params["max_core_x_in_chip"]
        self.chip_y = self.y // GV.sim_params["max_core_y_in_chip"]

        # Connected router
        self.router = None

        # Chip edge
        self.is_edge = [False for _ in range(<int>EL.DirectionIndex.direction_max)] # LEFT, BOT, TOP, RIGHT

        if self.x_in_chip == 0: self.is_edge[<int>EL.DirectionIndex.left] = True
        if self.y_in_chip == 0: self.is_edge[<int>EL.DirectionIndex.bot] = True
        if self.y_in_chip == GV.sim_params["max_core_y_in_chip"] - 1: self.is_edge[<int>EL.DirectionIndex.top] = True
        if self.x_in_chip == GV.sim_params["max_core_x_in_chip"] - 1: self.is_edge[<int>EL.DirectionIndex.right] = True

        # 4 switches around
        self.switches = [None, None, None, None]

        # Switch delay
        self.delay = [0, 0, 0, 0, 1]
        self.delay[<int>EL.DirectionIndex.left] = GV.timing_params["switch_EW_delay"]
        self.delay[<int>EL.DirectionIndex.bot] = GV.timing_params["switch_NS_delay"]
        self.delay[<int>EL.DirectionIndex.top] = GV.timing_params["switch_NS_delay"]
        self.delay[<int>EL.DirectionIndex.right] = GV.timing_params["switch_EW_delay"]

        # Input data
        self.i_data = [None, None, None, None, None]

        # Output data
        self.o_data = [None, None, None, None, None]
        self.o_data_next = [None, None, None, None, None]

        # Output buffer
        self.o_buf = [deque(), deque(), deque(), deque(), deque()]
        self.buf_pos = [0, 0, 0, 0, 0]


    cpdef put_buf(self, pck):
        cdef int dx
        cdef int dy
        cdef int direction

        if not pck:
            return

        dx = pck["dst_x"] - self.x
        dy = pck["dst_y"] - self.y

        if dx < 0:
            direction = 0
        elif dx > 0:
            direction = 3
        elif dy < 0:
            direction = 1
        elif dy > 0:
            direction = 2
        else: # dx, dy = 0
            direction = 4


        while len(self.o_buf[direction]) < self.delay[direction]:
            if len(self.o_buf[direction]) == self.delay[direction] - 1:
                self.o_buf[direction].append([None for _ in range(GV.timing_params["sw_to_sw_bw"])])
                self.buf_pos[direction] = 0
            else:
                self.o_buf[direction].append(None)

        if self.buf_pos[direction] == GV.timing_params["sw_to_sw_bw"]:
            self.o_buf[direction].append([None for _ in range(GV.timing_params["sw_to_sw_bw"])])
            self.buf_pos[direction] = 0

        self.o_buf[direction][-1][self.buf_pos[direction]] = pck
        self.buf_pos[direction] += 1



    cpdef put_activation_list(self, switch, activation_list):
        if not switch.in_activation_list:
            switch.in_activation_list = True
            activation_list.append(switch)


    cpdef put_deactivation_list(self, switch, deactivation_list):
        if not switch.in_deactivation_list:
            switch.in_deactivation_list = True
            deactivation_list.append(switch)

    cpdef advance(self, activation_list):
        cdef int direction
        cdef dict pck
        cdef int transfer_count

        for direction in range(4):
            self.o_data[direction] = self.o_data_next[direction]

        if self.o_data_next[4]:
            assert self.router
            transfer_count = 0
            for pck in self.o_data_next[4]:
                if not pck or transfer_count >= GV.timing_params["sw_to_core_bw"]:
                    break
                self.router.packet_in(pck)
                transfer_count += 1

        for direction in range(4):
            if self.o_data[direction] and not self.is_edge[direction]:
                self.put_activation_list(self.switches[direction], activation_list)


    cpdef calc_next(self, deactivation_list):
        for direction in range(4):
            if not self.is_edge[direction]:
                self.i_data[direction] = self.switches[direction].o_data[3-direction]
                self.switches[direction].o_data[3-direction] = None


        for direction in range(5):
            if self.i_data[direction]:
                for pck in self.i_data[direction]:
                    self.put_buf(pck)
            self.i_data[direction] = None

        self.flag_deact = True

        for direction in range(5):
            if self.o_buf[direction]:
                self.flag_deact = False
                self.o_data_next[direction] = self.o_buf[direction].popleft()
            else:
                self.o_data_next[direction] = None

        #self.flag_deact = False
        if self.flag_deact:
            self.put_deactivation_list(self, deactivation_list)

        self.cyc += 1


cdef class MergeSplit:
    def __init__(self, int chip_x, int chip_y, int direction):
        self.cyc = 0

        # Chip position
        self.chip_x = chip_x
        self.chip_y = chip_y

        # Direction
        self.direction = direction
        self.is_horizontal = True if direction == 0 or direction == 3 else False
        self.length = GV.sim_params["max_core_y_in_chip"] if self.is_horizontal else GV.sim_params["max_core_x_in_chip"]

        # Merging and spliting switches
        self.merging_switches = []

        # Neighbor MS block
        self.neighbor_block = None

        # Latency
        self.delay = GV.timing_params["switch_EW_delay"] if self.is_horizontal else GV.timing_params["switch_NS_delay"]
        if self.direction == <int>EL.DirectionIndex.left and self.chip_x % 4 == 0 \
        or self.direction == <int>EL.DirectionIndex.bot and self.chip_y % 8 == 0 \
        or self.direction == <int>EL.DirectionIndex.top and self.chip_y % 8 == 7 \
        or self.direction == <int>EL.DirectionIndex.right and self.chip_x % 4 == 3:
            self.chip_delay = GV.timing_params["board_delay"] - 1
        else:
            self.chip_delay = GV.timing_params["chip_delay"] - 1

        # Core-side data & buffer
        self.i_data = None
        self.o_data_next = [None for _ in range(self.length)]
        self.o_buf = [deque() for _ in range(self.length)]
        self.buf_pos = [0 for _ in range(self.length)]

        # Chip-side data & buffer
        self.chip_i_data = None
        self.chip_o_data = None
        self.chip_o_data_next = None
        self.chip_o_buf = deque()
        self.chip_latency_buf = deque()
        for _ in range(self.chip_delay):
            self.chip_latency_buf.append(None)

        self.off_chip_window = deque()
        for _ in range(GV.timing_params["off_chip_window_size"] - 1):
            self.off_chip_window.append(0)


    cpdef put_buf(self, pck, packet_pos):
        if pck is None:
            return

        if packet_pos >= 0: # switch to off chip
            pck["packet_pos"] = packet_pos

            self.chip_o_buf.append(pck)

        else: # off chip to switch
            packet_pos = pck["packet_pos"]

            while len(self.o_buf[packet_pos]) < self.delay:
                if len(self.o_buf[packet_pos]) == self.delay - 1:
                    self.o_buf[packet_pos].append([None for _ in range(GV.timing_params["sw_to_sw_bw"])])
                    self.buf_pos[packet_pos] = 0
                else:
                    self.o_buf[packet_pos].append(None)

            if self.buf_pos[packet_pos] == GV.timing_params["sw_to_sw_bw"]:
                self.o_buf[packet_pos].append([None for _ in range(GV.timing_params["sw_to_sw_bw"])])
                self.buf_pos[packet_pos] = 0

            self.o_buf[packet_pos][-1][self.buf_pos[packet_pos]] = pck
            self.buf_pos[packet_pos] += 1


    cpdef put_activation_list(self, block, activation_list):
        if not block.in_activation_list:
            block.in_activation_list = True
            activation_list.append(block)


    cpdef advance(self, activation_list):
        cdef int i

        for i in range(self.length):
            self.merging_switches[i].i_data[self.direction] = self.o_data_next[i]

        self.chip_o_data = self.chip_o_data_next

        for i in range(self.length):
            if self.o_data_next[i]:
                self.put_activation_list(self.merging_switches[i], activation_list)


    cpdef calc_next(self, deactivation_list):
        cdef int packet_pos
        cdef int i
        cdef Switch switch
        cdef dict pck

        for switch in self.merging_switches:
            self.i_data = switch.o_data[self.direction]
            switch.o_data[self.direction] = None

            if self.is_horizontal: packet_pos = switch.y_in_chip
            else : packet_pos = switch.x_in_chip
            
            if self.i_data:
                for pck in self.i_data:
                    self.put_buf(pck, packet_pos)

        if self.neighbor_block:
            self.chip_i_data = self.neighbor_block.chip_o_data
            self.neighbor_block.chip_o_data = None

        if self.chip_i_data:
            for pck in self.chip_i_data:
                self.put_buf(pck, -1)

        off_chip_packets = []

        for _ in range(GV.timing_params["off_chip_integer"]):
            if self.chip_o_buf:
                off_chip_packets.append(self.chip_o_buf.popleft())
            else: break

        if self.chip_o_buf and sum(self.off_chip_window) < GV.timing_params["off_chip_window_num"]:
            off_chip_packets.append(self.chip_o_buf.popleft())
            self.off_chip_window.append(1)
            self.off_chip_window.popleft()
        else:
            self.off_chip_window.append(0)
            self.off_chip_window.popleft()

        self.chip_latency_buf.append(off_chip_packets)
        self.chip_o_data_next = self.chip_latency_buf.popleft()

        for i in range(self.length):
            if self.o_buf[i]:
                self.o_data_next[i] = self.o_buf[i].popleft()
            else:
                self.o_data_next[i] = None


        self.cyc += 1


cdef class NoC:
    def __init__(self):
        self.cyc = 0

        self.max_x = GV.sim_params["max_core_x_in_total"]
        self.max_y = GV.sim_params["max_core_y_in_total"]

        # Initialize switch
        self.sw = [[Switch(x, y) for y in range(self.max_y)]\
                                for x in range(self.max_x)]

        # Switch connetion
        for x in range(GV.sim_params["max_core_x_in_total"]):
            for y in range(GV.sim_params["max_core_y_in_total"]):
                left_x = x - 1
                right_x = x + 1
                bot_y = y - 1
                top_y = y + 1

                if left_x == -1:
                    left_x = GV.sim_params["max_core_x_in_total"] - 1 #FIXME torus topology?

                if right_x == GV.sim_params["max_core_x_in_total"]:
                    right_x = 0

                if bot_y == -1:
                    bot_y = GV.sim_params["max_core_y_in_total"] - 1

                if top_y == GV.sim_params["max_core_y_in_total"]:
                    top_y = 0

                self.sw[x][y].switches[0] = self.sw[left_x][y]
                self.sw[x][y].switches[1] = self.sw[x][bot_y]
                self.sw[x][y].switches[2] = self.sw[x][top_y]
                self.sw[x][y].switches[3] = self.sw[right_x][y]


        # Initialize MS block
        self.merge_split = [[[MergeSplit(chip_x, chip_y, d) for d in range(4)]\
                                                            for chip_y in range(GV.sim_params["chip_y"])]\
                                                            for chip_x in range(GV.sim_params["chip_x"])]

        # Connect switches and MS blocks
        for x in range(self.max_x):
            for y in range(self.max_y):
                for d in range(4):
                    if self.sw[x][y].is_edge[d]:
                        switch = self.sw[x][y]
                        chip_x = switch.chip_x
                        chip_y = switch.chip_y
                        merging_msblock = self.merge_split[chip_x][chip_y][d]

                        merging_msblock.merging_switches.append(switch)
                        switch.switches[d] = merging_msblock

        # Connect neighbor MS block
        for chip_x in range(GV.sim_params["chip_x"]):
            for chip_y in range(GV.sim_params["chip_y"]):
                for d in range(4):
                    msblock = self.merge_split[chip_x][chip_y][d]

                    if d == <int>EL.DirectionIndex.left:
                        if chip_x - 1 >= 0:
                            msblock.neighbor_block = self.merge_split[chip_x-1][chip_y][<int>EL.DirectionIndex.right]
                    elif d == <int>EL.DirectionIndex.bot:
                        if chip_y - 1 >= 0:
                            msblock.neighbor_block = self.merge_split[chip_x][chip_y-1][<int>EL.DirectionIndex.top]
                    elif d == <int>EL.DirectionIndex.top:
                        if chip_y + 1 < GV.sim_params["chip_y"]:
                            msblock.neighbor_block = self.merge_split[chip_x][chip_y+1][<int>EL.DirectionIndex.bot]
                    elif d == <int>EL.DirectionIndex.right:
                        if chip_x + 1 < GV.sim_params["chip_x"]:
                            msblock.neighbor_block = self.merge_split[chip_x+1][chip_y][<int>EL.DirectionIndex.left]


        # Active switch list
        self.active_sw_list = []
        for chip_x in range(GV.sim_params["chip_x"]):
            for chip_y in range(GV.sim_params["chip_y"]):
                for d in range(4):
                    self.active_sw_list.append(self.merge_split[chip_x][chip_y][d])


    cpdef noc_advance(self):
        cdef list activation_list = []
        cdef list deactivation_list = []
        cdef object switch

        # Switch advance
        for switch in self.active_sw_list:
            switch.advance(activation_list)

        # Activate receiving switches
        for switch in activation_list:
            switch.in_activation_list = False
            self.activate_sw(switch)
        
        # Send spike messages to switch
        for core in GV.cores:
            self.send_pck_to_sw(core.router)
        
        # Switch calc_next
        for switch in self.active_sw_list:
            switch.calc_next(deactivation_list)
        
        # Deactivate empty switch
        for switch in deactivation_list:
            switch.in_deactivation_list = False
            self.deactivate_sw(switch)
        
        self.cyc += 1


    cpdef send_pck_to_sw(self, router):
        if not router.send_buf:
            return

        packet = [None for _ in range(GV.timing_params["core_to_sw_bw"])]

        x = router.x
        y = router.y

        for i in range(GV.timing_params["core_to_sw_bw"]):

            top_packet = router.send_buf[0]
            if top_packet.key <= self.cyc:
                packet[i] = heapq.heappop(router.send_buf).dct

        self.sw[x][y].i_data[4] = packet
        self.activate_sw(self.sw[x][y])


    cpdef activate_sw(self, switch):
        if not switch.is_active:
            switch.is_active = True
            switch.cyc = self.cyc
            self.active_sw_list.append(switch)


    cpdef deactivate_sw(self, switch):
        if switch.is_active:
            switch.is_active = False
            self.active_sw_list.remove(switch)
