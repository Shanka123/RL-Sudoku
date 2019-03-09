import numpy as np
quizzes = np.zeros((1000000, 81), np.int32)
solutions = np.zeros((1000000, 81), np.int32)
for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s
    print(i)
quizzes = quizzes.reshape((-1, 9, 9))
solutions = solutions.reshape((-1, 9, 9))
with open('/home/temp_siplab//shanka/temporary/rl_sudoku/sudoku9_grids.pkl', 'wb') as fp:
    pickle.dump(quizzes, fp)
with open('/home/temp_siplab/shanka/temporary/rl_sudoku/sudoku9_sols.pkl', 'wb') as fp:
    pickle.dump(solutions, fp)
