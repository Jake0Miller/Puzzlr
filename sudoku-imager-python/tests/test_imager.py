import pdb
import sys
sys.path.insert(0, '.')
import imager
from seeds import seeds

index = 0
run = 0
success = 0
for board, solution in zip(seeds.boards, seeds.solutions):
    index += 1
    for solved in imager.read_image(board):
        run += 1
        if solved == solution:
            print(f"Test {index} passed!")
            success += 1
        else:
            print(f"Test {index} failed!")

print(f"{run}/{len(seeds.boards)} tests ran!")
print(f"{success}/{len(seeds.boards)} succeeded!")
