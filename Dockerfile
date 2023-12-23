FROM public.ecr.aws/lambda/python:3.9

COPY data ./data
COPY 348000_model.pth ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY tit.py ${LAMBDA_TASK_ROOT}
COPY transformer.py ${LAMBDA_TASK_ROOT}
COPY util.py ${LAMBDA_TASK_ROOT}
COPY model_tit.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]