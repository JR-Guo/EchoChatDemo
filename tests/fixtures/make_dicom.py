"""Synthesize tiny DICOM files for testing the pipeline.

We cannot ship real patient DICOMs; instead we synthesize minimal ones
with pydicom + numpy so tests stay fully in-tree and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _empty_ds(path: Path) -> FileDataset:
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Echo"
    ds.PatientID = "TEST0001"
    ds.Modality = "US"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def make_still_dicom(path: Path, shape=(64, 64)) -> Path:
    ds = _empty_ds(path)
    arr = (np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8))
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


def make_cine_dicom(path: Path, frames: int = 5, shape=(32, 32)) -> Path:
    ds = _empty_ds(path)
    arr = np.random.default_rng(1).integers(0, 255, size=(frames, *shape), dtype=np.uint8)
    ds.Rows, ds.Columns = shape
    ds.NumberOfFrames = frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


def make_non_dicom(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    return path
