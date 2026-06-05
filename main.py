import io
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
import cv2
import numpy as np
from fastapi import FastAPI
