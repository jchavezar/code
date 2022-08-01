
import multiprocessing
import os

import uvicorn


if __name__ == "__main__":
    workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
    max_workers_str = os.getenv("MAX_WORKERS")
    use_max_workers = None
    if max_workers_str:
        use_max_workers = int(max_workers_str)
    web_concurrency_str = os.getenv("WEB_CONCURRENCY")

    if not web_concurrency_str:
        cores = multiprocessing.cpu_count()
        workers_per_core = float(workers_per_core_str)
        default_web_concurrency = workers_per_core * cores
        web_concurrency = max(int(default_web_concurrency), 2)
        if use_max_workers:
            web_concurrency = min(web_concurrency, use_max_workers)
        os.environ["WEB_CONCURRENCY"] = str(web_concurrency)

    uvicorn.run("cpr_model_server:ModelServer", host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT")), factory=True)
