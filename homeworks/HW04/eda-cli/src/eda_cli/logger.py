import json
import uuid
import sys
from pathlib import Path
from typing import Dict,Any
from datetime import datetime


def setup_logger(log_file: str="logs/api.log"):
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)


    def log_request(
            endpoint:str,
            status:str,
            latency_ms:float,
            ok_for_model:bool,
            n_rows:int=0,
            n_cols:int=0,
            message:str="",
    ):
        """Логирование запроса в JSON-формате"""
        log_data={
            "endpoint":endpoint,
            "status":status,
            "latency_ms":round(latency_ms,3),
            "ok_for_model":ok_for_model,
            "n_rows":n_rows,
            "n_cols":n_cols,
            "timestamp":datetime.now().isoformat(),
            "request_id":str(uuid.uuid4()),
            "message":message,
        }

        with open(log_path,"a",encoding="utf-8") as f:
            f.write(json.dumps(log_data,ensure_ascii=False)+"\n")
        
        print(f"[LOGGER] [{status}] [{endpoint}] [ok_for_model:{ok_for_model}] - message:{message},timestamp:{log_data["timestamp"]},latency-ms:{latency_ms}")
        if status!="success":
            print(f"[LOGGER] - For more information read the file:{log_file}")

    return log_request

log_request=setup_logger()