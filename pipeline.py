from s3_save_data import df
from s5_run_models import  run_models
from s6_eval import run_eval


def run_pipeline():
    df
    run_models()
    run_eval()

if __name__ == "__main__":
    run_pipeline()