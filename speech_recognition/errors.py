
class Error(Exception):
    """Base class for other exceptions"""
    pass

class TotalSamplesNotAlignedSpeakerSamples(Error):
    pass

class ExitApp(Error):
    pass
    