import copy
from enum import Enum
import random
from collections import deque

import numpy as np
from scipy.ndimage.filters import minimum_filter, maximum_filter

def dilation(arr, ksize=3):                
    ret_arr = np.copy(arr)
    dilation_kernel = np.ones((ksize, ksize))
    
    if ksize % 2 == 0: 
        l = int(ksize / 2.0)
        r = int(ksize / 2.0)
    else:
        l = int(np.floor(ksize / 2.0))
        r = int(np.floor(ksize / 2.0))+1    
            
    for iy in range(l, arr.shape[0]-r):
        for ix in range(l, arr.shape[1]-r):
            if np.any(arr[iy, ix]==0):
                ret_arr[iy-l:iy+r, ix-l:ix+r] = 0                            
    return ret_arr

class Dungeon:
    def __init__(self, w, h, debug=False):
        super().__init__()
        """
        0: FREE
        1: WALL
        -1: GOAL
        -2: START
        """
        self.debug = debug         
        self.w = w
        self.h = h
        
        self.data = np.zeros((h, w), np.int) 
        self.room = np.zeros((h, w), dtype=bool)
        self.corridor = np.zeros((h, w), dtype=bool)        
        
        self.room_idx = {}
        self.goal_x = None
        self.goal_y = None
        self.start_x = None 
        self.start_y = None

    def __len__(self):
        return self.w * self.h
    
    def __str__(self):
        
        ret = ""
        for iy in range(self.h):
            for ix in range(self.w):
                if self.data[iy, ix] == 1:
                    ret += "W"
                elif self.data[iy, ix] == -1:
                    ret += "G"
                elif self.data[iy, ix] == -2:
                    ret += "S"                    
                else:
                    ret +=" "
            ret +="\n"
        return ret
                    
    def create_map_dungeon(self, num_col_rooms=3, 
                           num_row_rooms=2, 
                           corridor_width=1, 
                           min_room_size_ratio=0.3, 
                           max_room_size_ratio=0.8
                           ):
        """
        """
        self.data = np.ones(self.data.shape, np.int)
        
        # Random col and row number
        rand_col_idx = [ int(e)&~1 for e in np.linspace(0, self.w-1, num=int(num_col_rooms)+1)]
        rand_row_idx = [ int(e)&~1 for e in np.linspace(0, self.h-1, num=int(num_row_rooms)+1)]

        # Center of room
        room_center_x = np.zeros((num_row_rooms, num_col_rooms),np.int)
        room_center_y = np.zeros((num_row_rooms, num_col_rooms),np.int)
        
        # Max size of room
        room_max_size_x = np.zeros((num_row_rooms, num_col_rooms),np.int)
        room_max_size_y = np.zeros((num_row_rooms, num_col_rooms),np.int)

        # create rooms and corridors         
        rooms = np.ones(self.data.shape, np.int)
        corridors = np.ones(self.data.shape, np.int)
    
        # index of room
        room_idx = 0
        
        for ic in range(num_col_rooms):
            for ir in range(num_row_rooms):

                # room center pos
                cx = int(0.5 * (rand_col_idx[ic] + rand_col_idx[ic+1]))
                cy = int(0.5 * (rand_row_idx[ir] + rand_row_idx[ir+1]))
                room_center_x[ir, ic] = cx
                room_center_y[ir, ic] = cy
                self.room_idx[room_idx] = (cy, cx)
                room_idx += 1

                # room max size                 
                rmsx = int(rand_col_idx[ic+1] - rand_col_idx[ic])
                rmsy = int(rand_row_idx[ir+1] - rand_row_idx[ir])

                # room size at random
                w = np.random.randint(int(rmsx * min_room_size_ratio), int(rmsx * max_room_size_ratio))
                h = np.random.randint(int(rmsy * min_room_size_ratio), int(rmsy * max_room_size_ratio))
                rooms[cy-int(h/2.0):cy+int(h/2.0), cx-int(w/2.0):cx+int(w/2.0)] = 0
                
                # pos of each corridors at random
                try:
                    exit_x_up = np.random.randint(cx-int(w/2.0)+1, cx+int(w/2.0)-1)
                    exit_x_down = np.random.randint(cx-int(w/2.0)+1, cx+int(w/2.0)-1)
                except:
                    raise ValueError("Too many col room numbers")
                
                try:
                    exit_left = np.random.randint(cy-int(h/2.0)+1, cy+int(h/2.0)-1)
                    exit_right = np.random.randint(cy-int(h/2.0)+1, cy+int(h/2.0)-1)
                except:
                    raise ValueError("Too many row room numbers")

                # create corridors for each rooms
                rx = int(rmsx/2.0)
                ry = int(rmsy/2.0)
                
                if num_row_rooms == 1 and ic == 0:
                    corridors[exit_right, cx:cx+rx] = 0
                    
                elif num_row_rooms == 1 and ic == num_col_rooms-1:
                    corridors[exit_left, cx-rx:cx] = 0

                elif num_col_rooms == 1 and ir == 0:
                    corridors[cy:cy+ry, exit_x_down] = 0
                    
                elif num_col_rooms == 1 and ir == num_row_rooms-1:
                    corridors[cy-ry:cy, exit_x_up] = 0

                elif ir == 0 and ic == 0:
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0

                elif ir == num_row_rooms-1 and ic == num_col_rooms-1:
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0

                elif ir == 0 and ic == num_col_rooms-1:
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0

                elif ir == num_row_rooms-1 and ic == 0:
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0

                elif ir == num_row_rooms-1 and ic == 0:
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0

                elif ir == 0:
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0

                elif ir == num_row_rooms-1:
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0

                elif ic == 0:
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0

                elif ic == num_col_rooms-1:
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0
                            
                else:                
                    corridors[exit_left, cx-rx:cx] = 0
                    corridors[exit_right, cx:cx+rx] = 0
                    corridors[cy-ry:cy, exit_x_up] = 0
                    corridors[cy:cy+ry, exit_x_down] = 0
                    
        # connect corridors
        rand_col_idx = np.array(rand_col_idx)
        for col_idx in rand_col_idx[1:-1]:
            end_y_l = np.where(corridors[:, col_idx - 1]== 0 )
            end_y_r = np.where(corridors[:, col_idx + 1]== 0 )
            for idx in range(num_row_rooms):
                end_y = sorted([end_y_l[0][idx], end_y_r[0][idx]])
                corridors[end_y[0]:end_y[1]+1, col_idx] = 0                
        
        # connect corridors
        rand_row_idx = np.array(rand_row_idx)
        for row_idx in rand_row_idx[1:-1]:
            end_x_u = np.where(corridors[row_idx - 1, :]== 0 )
            end_x_d = np.where(corridors[row_idx + 1, :]== 0 )
            for idx in range(num_col_rooms):
                end_x = sorted([end_x_u[0][idx], end_x_d[0][idx]])
                corridors[row_idx, end_x[0]:end_x[1]+1] = 0                

        # expand corridor width
        if corridor_width > 1:            
            corridors = dilation(corridors, ksize=corridor_width)            

        # combine rooms and corridors
        self.data = np.logical_and(rooms, corridors).astype(np.int)
        self.rooms = rooms.astype(np.bool)
        self.corridors = corridors.astype(np.bool)

    def get_free_space(self, num=1):
        """
        if num == 1:
            return (y, x)
        else:
            return [(y1, x1), (y2, x2), ...]
        """
        idx = list(zip(*np.where(self.data==0)))
        yx = random.sample(idx, num)
        
        if num==1:
            yx = yx[0]
    
        return yx

    def get_local_data(self, mcy, mcx, size=16):
        """
        return arr, (left_margin_x, right_margin_x, up_margin_y, down_,margin_y)        
        """
        
        if not (0 <= mcy < self.h and 0 <= mcx < self.w):
            raise ValueError("Invalid (mcy, mcx)")
                        
        # mcx = int(cx/size)
        # mcy = int(cy/size)        
        rmdx = lmdx = umdy = dmdy = int(size/2)
        mdx2 = mdy2 = size
        
        if mcx - lmdx < 0:            
            lmdx = mcx - 0
            rmdx = mdx2 - lmdx
        elif mcx + rmdx > self.w:
            rmdx = self.w - mcx
            lmdx = mdx2 - rmdx
        if mcy - umdy < 0:
            umdy = mcy - 0
            dmdy = mdy2 - umdy
        if mcy + dmdy > self.h:
            dmdy = self.h - mcy
            umdy = mdy2 - dmdy
            
        return self.data[mcy-umdy:mcy+dmdy, mcx-lmdx:mcx+rmdx], (lmdx, rmdx, umdy, dmdy)

    def set_start_random(self):

        self.data = np.where(self.data == -2, 0, self.data)
        
        # get one pos from FREE at random
        idx = list(zip(*np.where(self.data==0)))
        (y, x) = random.sample(idx, 1)[0]

        # set GOAL
        self.data[y, x] = -2  # START
        self.start_x = x
        self.start_y = y
        
    def set_goal_random(self):
        
        # replace GOAL with FREE
        self.data = np.where(self.data == -1, 0, self.data)

        # get one pos from FREE at random
        idx = list(zip(*np.where(self.data==0)))
        (y, x) = random.sample(idx, 1)[0]

        # set GOAL
        self.data[y, x] = -1 # GOAL
        self.goal_x = x
        self.goal_y = y
 
    def search_shortest_path_dws(self, start, goal):
        """
        start = (y, x)
        goal = (y, x)
        """
        
        start_goal = np.zeros((self.h, self.w), dtype=int)
        cost = np.zeros((self.h, self.w), dtype=int) + 1E10
        done = np.zeros((self.h, self.w), dtype=bool)
        barrier = np.zeros((self.h, self.w), dtype=bool)
        path = np.zeros((self.h, self.w), dtype=int)
                
        for iy in range(self.h):
            for ix in range(self.w):
                if iy == start[0] and ix == start[1]:
                    cost[iy, ix] = 0
                    done[iy, ix] = True
                    start_goal[iy, ix] = -255
                    
                if iy == goal[0] and ix == goal[1]:                
                    start_goal[iy, ix] = 255
                    
                if self.data[iy, ix] == 1: # WALL
                    barrier[iy, ix] = True 
                    
        barrier[start[0], start[1]] = False
        barrier[goal[0], goal[1]] = False
    
        g = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]) 
        
        for i in range(1, 10000000):

            done_next = maximum_filter(done, footprint=g) * ~done
            cost_next = minimum_filter(cost, footprint=g) * done_next
            cost_next[done_next] += 1   
            cost[done_next] = cost_next[done_next]
            cost[barrier] = 10000000
            done[done_next] = done_next[done_next]
            done[barrier] = False

            if done[goal[0], goal[1]] == True:
                break

        point_now = goal
        cost_now = cost[goal[0], goal[1]]
        route = [goal]

        while cost_now > 0:
            try:
                if cost[point_now[0] - 1, point_now[1]] == cost_now - 1:
                    point_now = (point_now[0] - 1, point_now[1])
                    cost_now = cost_now - 1
                    route.append(point_now)
            except: pass
            try:
                if cost[point_now[0] + 1, point_now[1]] == cost_now - 1:
                    point_now = (point_now[0] + 1, point_now[1])
                    cost_now = cost_now - 1
                    route.append(point_now)
            except: pass
            try:
                if cost[point_now[0], point_now[1] - 1] == cost_now - 1:
                    point_now = (point_now[0], point_now[1] - 1)
                    cost_now = cost_now - 1
                    route.append(point_now)
            except: pass
            try:
                if cost[point_now[0], point_now[1] + 1] == cost_now - 1:
                    point_now = (point_now[0], point_now[1] + 1)
                    cost_now = cost_now - 1
                    route.append(point_now)
            except: pass

        route = route[::-1]
                
        for cell in route:
            ix = cell[1]
            iy = cell[0]
            path[iy, ix] = 1

        if self.debug:        

            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import matplotlib.colors as colors
        
            cmap = cm.jet
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0
            customized_jet = colors.ListedColormap(cmap_data)
            
            cmap = cm.magma
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0
            customized_magma = colors.ListedColormap(cmap_data)
            
            cmap = cm.cool
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0
            customized_cool = colors.ListedColormap(cmap_data)

            cmap = cm.gist_yarg
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0
            customized_gist_yarg = colors.ListedColormap(cmap_data)

            plt.scatter(start[1], start[0], s=2, c="blue", label="start")
            plt.scatter(goal[1], goal[0], s=2, c="red", label="goal")
            plt.imshow(barrier, cmap="gist_yarg")
            plt.legend()
            plt.show()
            
          
            plt.imshow(cost, cmap="jet", vmax=100, vmin=0, alpha=0.8)  
            plt.imshow(barrier, cmap=customized_gist_yarg)
            plt.scatter(start[1], start[0], s=2, c="blue", label="start")
            plt.scatter(goal[1], goal[0], s=2, c="red", label="goal")
            plt.legend()
            plt.show()

            plt.imshow(path, cmap=customized_cool)        
            plt.imshow(barrier, cmap=customized_gist_yarg)
            plt.scatter(start[1], start[0], s=2, c="blue", label="start")
            plt.scatter(goal[1], goal[0], s=2, c="red", label="goal")
            plt.legend()
            plt.show()
        
        return route

if __name__ == "__main__":

    width = 128
    height = 128
    dungeon = Dungeon(width, height, debug=True)
    dungeon.create_map_dungeon(num_col_rooms=7, num_row_rooms=7, corridor_width=1)
    dungeon.set_goal_random()
    dungeon.set_start_random()

    route = dungeon.search_shortest_path_dws((dungeon.start_y, dungeon.start_x), (dungeon.goal_y, dungeon.goal_x))
    lmd, _ = dungeon.get_local_data(dungeon.start_y, dungeon.start_x, size=21)
    
    print(dungeon)
    print(lmd)