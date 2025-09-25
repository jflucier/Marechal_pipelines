import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from dpfold.multimer import parse_multimer_list_from_samplesheet
from dpfold.pipeline_conf import gen_conf
from dry_pipe.service import PipelineRunner

from web_gasket.routes import init_page_and_upload_routes, create_sub_api
from web_gasket.auth import SqliteAuthenticator


import logging
import logging.config


def init_logging():

    logging_conf = os.environ.get("LOGGING_CONF")

    if logging_conf is not None:

        if not Path(logging_conf).exists():
            raise Exception(f"logging config file '{logging_conf}' refered by env var LOGGING_CONF does not exist")

        with open(logging_conf, "r") as f:
            log_conf_json = json.load(f)

            handlers = log_conf_json["handlers"]

            for k, handler in handlers.items():
                handler_class = handler["class"]
                if handler_class == "logging.FileHandler":
                    filename = handler.get("filename")
                    if filename is None or filename == "":
                        raise Exception(f"logging.FileHandler '{k}' has no filename attribute in {logging_conf}")
                    if "$" in filename:
                        filename = os.path.expandvars(filename)
                        handler["filename"] = os.path.expandvars(filename)

        logger = logging.getLogger(__name__)
        logger.info("using logging config file '%s'", logging_conf)

    else:
        log_conf_json = {
          "version": 1,
          "formatters": {
            "simple": {
              "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
          },
          "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            }
          },
          "root": {
            "level": "DEBUG",
            "handlers": ["console"]
          },
          "web_gasket": {
              "level": "DEBUG",
              "handlers": ["console"],
              "propagate": 0
          },
          "dry_pipe": {
              "level": "DEBUG",
              "handlers": ["console"],
              "propagate": 0
          }
        }

    logging.config.dictConfig(log_conf_json)


def parse_permissions(user_email):

    ppf = os.environ.get("PIPELINE_PERMISSIONS_FILE")

    if ppf is None:
        raise RuntimeError("variable PIPELINE_PERMISSIONS_FILE not set")

    if not Path(ppf).exists():
        raise RuntimeError(f"file '{ppf}' refered by env var PIPELINE_PERMISSIONS_FILE does not exist")

    with open(ppf, "r") as f:
        c = 1
        for line in f.readlines():
            line = line.strip()
            if line == "":

                c += 1

            try:
                _user_email, pipelines, allocations = line.split("\t")
                _user_email = user_email.strip()

                if _user_email != user_email:
                    continue

                pipelines = pipelines.strip()
                allocations = allocations.strip()

                def vals(s):
                    return [s0.strip() for s0 in s.split(",")]

                return vals(pipelines), vals(allocations)

            except Exception as e:
                raise RuntimeError(f"invalid line '{line}' in PIPELINE_PERMISSIONS_FILE {ppf} {e}")

    return []



def init_app():

    init_logging()

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "Location",
            "Upload-Offset",
            "Tus-Resumable",
            "Tus-Version",
            "Tus-Extension",
            "Tus-Max-Size",
            "Upload-Expires",
            "Upload-Length",
        ]
    )

    WEB_GASKET_TEMP_FILE_UPLOAD_DIR = os.environ.get("WEB_GASKET_TEMP_FILE_UPLOAD_DIR")

    if WEB_GASKET_TEMP_FILE_UPLOAD_DIR is None:
        WEB_GASKET_TEMP_FILE_UPLOAD_DIR = "/tmp/ibio-reception-dir"

    Path(WEB_GASKET_TEMP_FILE_UPLOAD_DIR).mkdir(exist_ok=True, parents=True)

    pipeline_runner = PipelineRunner(gen_conf())

    api = create_sub_api(pipeline_runner)

    @api.get("/cc_allocations")
    async def cc_allocations(request: Request):

        sess = request.state.session

        user_email = sess.get("user_email")

        af = os.environ.get("CC_ALLOCATIONS_FILE")
        if af is None:
            raise Exception(f"missing env var CC_ALLOCATIONS_FILE")

        if not Path(af).exists():
            raise Exception(f"CC_ALLOCATIONS_FILE does not exist")

        with open(af, "r") as f:
            cc_allocs = json.load(f)

        return cc_allocs.get(user_email)


    @api.get("/dpFoldFilesStatus/{pid:path}")
    async def dp_files_status(pid: str):

        samplesheet = Path(f"/{pid}", "samplesheet.tsv")

        if samplesheet.exists():

            try:
                parse_multimer_list_from_samplesheet(samplesheet)
                samplesheet_parse_exception = None
            except Exception as e:
                samplesheet_parse_exception = e

            return {
                "name": "samplesheet.tsv",
                "exists": True,
                "isValid": samplesheet_parse_exception is None,
                "errors": None if samplesheet_parse_exception is None else str(samplesheet_parse_exception)
            }

        return {
            "name": "samplesheet.tsv",
            "exists": False
        }

    #@api.get("/dpFoldsStatus/{pid:path}")
    #async def dp_folds_status(pid: str):

    #    state_files = Path(f"/{pid}", ".drypipe", "samplesheet.tsv")


    user_auth_db = os.environ.get("USER_AUTH_DB")

    if user_auth_db is None:
        raise Exception(f"missing env var USER_AUTH_DB")

    web_artifacts_dir = Path(Path(__file__).parent.parent.parent, "web-ui", "build")

    def page_func(head="", bundle_js=None):

        return f"""
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8">        
                <link href="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.css" rel="stylesheet" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.tailwindcss.com"></script>
                <link rel="stylesheet" href="/style.css">
                {head}
            </head>
            <body>
                <div id="app"/>                
                <script type="module" src="/{bundle_js}" bundle></script>
            </body>
        </html>
        """


    session_key = os.environ.get("WEB_SESSION_KEY")

    if session_key is None:
        raise Exception(f"missing env var WEB_SESSION_KEY")

    authenticator = SqliteAuthenticator(
        user_auth_db,
        session_key,
        60*30
    )

    authenticator.init_routes(api, app, page_func, web_artifacts_dir)

    init_page_and_upload_routes(app, authenticator, page_func, web_artifacts_dir)

    app.mount("/api", api)

    return app


def run():

    init_logging()

    WEB_APP_PORT = os.environ.get("WEB_APP_PORT")

    port = 8000 if WEB_APP_PORT is None else int(WEB_APP_PORT)

    logger = logging.getLogger(__name__)
    logger.info(f"starting web app on port {port}")

    uvicorn.run(app="dpfold.server:init_app", host="0.0.0.0", port=port, workers=2)

if __name__ == '__main__':

    run()