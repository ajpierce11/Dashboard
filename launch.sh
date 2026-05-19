#!/bin/bash
  streamlit run comparator.py \
    --server.port=$CDSW_APP_PORT \
    --server.address=127.0.0.1 \
    --browser.serverAddress=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false