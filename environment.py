"""
Sudoku grid environment.
"""

import logging

import sudoku
import numpy as np


SUDOKU_SIZE = 4


class Environment:
    def __init__(self,ex):
        """
        Sudoku solving environment.
        """
        self.num_actions = SUDOKU_SIZE**3
        self.ex_no=ex
        self.start_grid = self.new_grid()
        self.current_grid = self.start_grid.copy()
    
    def new_grid(self):
        """
        Initialise a new sudoku grid for this environment.

        :return: The new grid
        """
        new_puzzle = sudoku.generate_grid(self.ex_no,flat=True)
        #logging.debug("Creating new grid\n%s", self.start_grid)
        #self.current_grid = self.start_grid.copy()

        return new_puzzle

    def reset_grid(self):
       self.current_grid = self.start_grid.copy()
       #return self.current_grid

    def act(self, action):
        """
        Perform an action within the current grid.

        :param action: The action to perform
        :return: A (grid, reward, terminal) tuple containing the new grid,
                 the reward for the given action, and whether the game is now over.
        """
        new_grid = sudoku.unflatten(self.current_grid)
    #    print(new_grid)
    #    print('Unflattened Grid>>>',new_grid)
        row_idx = action // (SUDOKU_SIZE**2)
        col_idx = (action % (SUDOKU_SIZE**2)) // SUDOKU_SIZE
        entry = action % SUDOKU_SIZE + 1

        if new_grid[row_idx][col_idx] != 0:
            self.current_grid = sudoku.flatten(new_grid)
            # This square already contains an entry.
            #print('already filled')
            if np.min(new_grid) > 0:
                    # Have solved the grid.
                    self.current_grid = sudoku.flatten(new_grid)
                #    print("\nSudoku solved!\n")
                    reward=0
                    mistake=0
                    terminal = 1
            else:
                reward=-100
                mistake=1
                terminal = 0
        else:
            
            new_grid[row_idx][col_idx] = entry
            is_valid = sudoku.check_valid(new_grid)
        #    print('Unflattened Grid after action>>>',new_grid)
            if is_valid:
                if np.min(new_grid) > 0:
                    # Have solved the grid.
                    self.current_grid = sudoku.flatten(new_grid)
                #    print("\nSudoku solved!\n")
                    reward = 0
                    mistake=0
                    terminal = 1
                else:
                    self.current_grid = sudoku.flatten(new_grid)
                    reward = 0
                    mistake=0
                    terminal = 0
            else:
                new_grid[row_idx][col_idx] = 0
                self.current_grid = sudoku.flatten(new_grid)
                reward = -100
            #    print('Faulty>>>>')
                mistake=1
                terminal = 0
        
        return self.current_grid, reward,mistake, terminal
