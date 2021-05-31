import os
import shutil
from pathlib import Path

misclass_folder = Path("./misclassified")


def store_misclassified(path: str) -> None:
    files = next(os.walk(misclass_folder))[2]
    last = len(files)
    if last < 100:
        os.rename(str(path), str(last) + ".png")
        shutil.move(str(last) + ".png", str(misclass_folder) + "/" + str(last) + ".png")
