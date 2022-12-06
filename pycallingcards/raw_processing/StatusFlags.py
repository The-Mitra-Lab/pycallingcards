"""Enumerate Status bit flags which will be used to mark why a read passes/fails"""
from enum import IntFlag

__all__ = ['StatusFlags']

class StatusFlags(IntFlag):
    """A barcode failure is 0x0, a mapq failure is 0x1 and a insert seq failure 
    is 0x2. A read that fails both barcode and mapq for instance would have 
    status 3.
    """
    BARCODE            = 0x0
    MAPQ               = 0x1
    INSERT_SEQ         = 0x2
    FIVE_PRIME_CLIP    = 0x3
    UNMAPPED           = 0x4
    NOT_PRIMARY        = 0x5
    ALIGNER_QC_FAIL    = 0x6
    RESTRICTION_ENZYME = 0x7

    def flag(self):
        return 2**self.value