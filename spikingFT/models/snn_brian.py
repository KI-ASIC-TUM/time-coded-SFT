#!/usr/bin/env python3
"""
Module containing a class for implementing the S-FT in Brian2
"""
# Standard libraries
# Local libraries
import spikingFT.models.snn

class SNNBrian(spikingFT.models.snn.FourierTransformSNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def run(self, data):
        return
