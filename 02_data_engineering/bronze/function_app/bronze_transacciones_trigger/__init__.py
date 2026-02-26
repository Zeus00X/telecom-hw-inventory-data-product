import logging
import os
import azure.functions as func


def main(event: func.EventGridEvent):
    stop = os.getenv("STOP_PROCESSING", "0")

    if stop == "1":
        logging.warning("STOP_PROCESSING activo. Ejecución cancelada.")
        return

    logging.info("Evento recibido desde Event Grid.")
    logging.info(f"Subject: {event.subject}")