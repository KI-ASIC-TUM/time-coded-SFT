#!/usr/bin/env python3
"""
Module containing the abstract class defining the SNN classes API
"""
# Standard libraries
from abc import ABC, abstractmethod
# Local libraries


class SpikingFTSNN(ABC):
    """
    Abstract class defining the interface of the SNN implementations 

    Any SNN model that has to be run in the library shall be created as
    an instance of this class
    """
    def __init__(self):
        return

    @abstractmethod
    def run(self, data, *args):
        return data

    def __call__(self, data, *args):
        self.run(*args)
        return
