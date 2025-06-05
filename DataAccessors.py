import numpy as np
import pandas as pd
from datetime import datetime
import sys
import glob
import cv2
import os
import random
import xlrd
import time
import json

def getOutcome(idx_base, outcome_name, filename="outcomes/GBa2D_ECU_Breast_2001-2010_Updated-for_BMI_and_survival.xlsx", dtype=float):
    '''
    A no-error method to get an outcome
    '''
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    search = "start"

    target_row = -1
    target_col = -1

    for row in range(0,152):
        search = sheet.cell_value(row, 0)
        if idx_base in str(search): target_row = row
    for col in range(0,sheet.ncols):
        search = sheet.cell_value(3, col) # in row 4...
        if search == outcome_name: target_col = col

    if not (target_row > -1 and target_col > -1):
        # Case, patient not in db
        print ("!!! Failed to find patient index !!!")
        return dtype(-9)
    if (sheet.cell_value(target_row,target_col)==''):
        # Case, the cell is empty
        print ("!!! Emtpy cell !!!")
        return dtype(-9)
    try:
        # Can we convert?
        return_as_dtype = dtype(sheet.cell_value(target_row,target_col))
    except:
        # Case, improper datatype
        print ("!!! Couldn't convert to dtype !!")
        return_as_dtype = dtype(-99)
    # Successful conversion
    return return_as_dtype

def getGrade(idx_base, outcome_name, filename="outcomes/GBa2D_ECU_Breast_2001-2010_Updated-for_BMI_and_survival.xlsx", dtype=float):
    '''
    A no-error method to get an outcome
    '''
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    search = "start"

    target_row = -1
    target_col = -1

    for row in range(0,sheet.nrows):
        search = sheet.cell_value(row, 1)
        if idx_base in str(search): target_row = row
    for col in range(0,sheet.ncols):
        search = sheet.cell_value(0, col) # in row 4...
        if search == outcome_name: target_col = col

    if not (target_row > -1 and target_col > -1):
        # Case, patient not in db
        #print ("!!! Failed to find patient index !!!")
        return dtype(-9)
    if (sheet.cell_value(target_row,target_col)==''):
        # Case, the cell is empty
        #print ("!!! Emtpy cell !!!")
        return dtype(-9)
    try:
        # Can we convert?
        return_as_dtype = dtype(sheet.cell_value(target_row,target_col))
    except:
        # Case, improper datatype
        #print ("!!! Couldn't convert to dtype !!")
        return_as_dtype = dtype(-99)
    # Successful conversion
    return return_as_dtype

def getMRN(idx_base, outcome_name, filename="/raid/Neoadjuvant BC/nacpre.xlsx", dtype=float):
    '''
    A no-error method to get an outcome
    '''
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    search = "start"

    target_row = -1
    target_col = 0
    index_col  = 1

    for row in range(0,sheet.nrows):
        search = sheet.cell_value(row, index_col)
        if idx_base in str(search): target_row = row

    if not (target_row > -1 and target_col > -1):
        # Case, patient not in db
        #print ("!!! Failed to find patient index !!!")
        return dtype(-9)
    if (sheet.cell_value(target_row,target_col)==''):
        # Case, the cell is empty
        #print ("!!! Emtpy cell !!!")
        return dtype(-9)
    try:
        # Can we convert?
        return_as_dtype = dtype(sheet.cell_value(target_row,target_col))
    except:
        # Case, improper datatype
        #print ("!!! Couldn't convert to dtype !!")
        return_as_dtype = dtype(-99)
    # Successful conversion
    return return_as_dtype


def split_GHP_convention(idx_base, dtypes):
    idx_base = idx_base.replace('-','_')
    base_split = idx_base.split('_')
    try:
        return_dtypes = dtypes[0](base_split[0]), dtypes[1](base_split[1]), dtypes[2](base_split[2])
    except:
        return_dtypes = dtypes[0]('-99'), dtypes[1]('-99'), dtypes[2]('-99')
    return return_dtypes


def getClusterIndex(identifiers, outcome_name, filename="outcomes/GBa2D_ECU_Breast_2001-2010_Updated-for_BMI_and_survival.xlsx", dtypes=float, dtype=str):
    '''
    A no-error method to get an outcome
    '''
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    search = "start"

    target_row = -1
    target_col = -1

    for row in range(0,sheet.nrows):
        search = sheet.cell_value(row, 0)
        search_identifiers = split_GHP_convention(search, dtypes)

        if identifiers == search_identifiers:
            target_row = row
    for col in range(0,sheet.ncols):
        search = sheet.cell_value(1, col) # in row 2...
        if search == outcome_name:
            target_col = col

    if not (target_row > -1 and target_col > -1):
        # Case, patient not in db
        #print ("!!! Failed to find patient index !!!")
        return dtype(-9)
    if (sheet.cell_value(target_row,target_col)==''):
        # Case, the cell is empty
        #print ("!!! Emtpy cell !!!")
        return dtype(-1)
    try:
        # Can we convert?
        return_as_dtype = dtype(sheet.cell_value(target_row,target_col))
    except:
        # Case, improper datatype
        #print ("!!! Couldn't convert to dtype !!")
        return_as_dtype = dtype(-99)
    # Successful conversion
    return return_as_dtype
