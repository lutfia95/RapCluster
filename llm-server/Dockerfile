# RapCluster Clustering Assistant: Ollama with Qwen2.5 7B Instruct (~4.7 GB, 4-bit), downloaded while the image is built,
# so the model is part of the image and works without internet access at run time.
# Pin the Ollama release for reproducible builds, e.g. --build-arg OLLAMA_VERSION=0.12.6
ARG OLLAMA_VERSION=latest
FROM ollama/ollama:${OLLAMA_VERSION}

ARG RAPCLUSTER_ASSISTANT_MODEL=qwen2.5:7b
# Keep the model loaded (no reload delay before each answer); one answer at a time, others wait in a queue.
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_MAX_QUEUE=16

# Start the server briefly, pull the model into the image, stop the server.
RUN ollama serve > /tmp/ollama-build.log 2>&1 & server=$!; \
    for i in $(seq 1 60); do ollama list > /dev/null 2>&1 && break; sleep 1; done; \
    ollama pull "${RAPCLUSTER_ASSISTANT_MODEL}"; status=$?; \
    kill "$server"; \
    if [ "$status" -ne 0 ]; then cat /tmp/ollama-build.log; fi; \
    exit "$status"

EXPOSE 11434
