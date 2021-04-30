import shutil
from pathlib import Path
import os
from openpyxl import load_workbook
import pandas as pd


# Empty and create direcotory
def create_dir(dir):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass
    Path(dir).mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rotate(l, n):
    return l[n:] + l[:n]


def save_results(sheet_name, file, table, index=False, header=True):
    try:
        book = load_workbook(file)
        writer = pd.ExcelWriter(file)
        writer.book = book
    except FileNotFoundError:
        writer = pd.ExcelWriter(file)
    table.to_excel(writer, sheet_name=sheet_name, index=index, header=header)
    writer.save()
    writer.close()
