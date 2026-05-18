from .download import download_all
from .organize import organize_all
from .processing import prepare_all

def full_pipeline() -> dict:
    download_result = download_all()
    organize_result = organize_all()
    prepare_result = prepare_all()
    
    return {
        "download": download_result,
        "organize": organize_result,
        "prepare": prepare_result
    }