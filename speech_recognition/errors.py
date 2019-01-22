
class Error(Exception):
    """Base class for other exceptions"""
    pass

class FeatureExtractionError(Error):
    pass

class ExitApp(Error):
    pass
    
class TotalSamplesNotAlignedSpeakerSamples(Error):
    pass
